import os
import logging
import time
from dotenv import load_dotenv
from typing import List
from consts import llm_model_type
from ai.ai_agents import get_agent_response, handle_multi_step_query
from ai.ai_chains import query_sql_db
from slack.slack_utils import get_random_thinking_message, send_slack_message_and_return_message_id, summarizeLongDocument, storeEmbeddings
from utils import extract_messages
from consts import demo_company_name, ai_name
from supabase_wrapper import write_message_log
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from ratelimit import limits, RateLimitException, sleep_and_retry


ONE_MINUTE = 60
TIER_1_RATELIMIT_PER_MINUTE = 1
TIER_2_RATELIMIT_PER_MINUTE = 20
TIER_3_RATELIMIT_PER_MINUTE = 50
TIER_4_RATELIMIT_PER_MINUTE = 100
TIER_5_RATELIMIT_PER_MINUTE = 500

# requires importing logging
logging.basicConfig(level=logging.INFO)

# Load .env variables
load_dotenv()

# LLM Initialization
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(max_retries=3, temperature=0.8,  # type: ignore
                 model=llm_model_type)

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        logging.info(token)

@sleep_and_retry
@limits(calls=TIER_3_RATELIMIT_PER_MINUTE, period=ONE_MINUTE)
def get_all_conversation_history(app, channel, cursor = None):
    response = app.client.conversations_history(
        channel=channel,
        limit=500,
        cursor=cursor)
    messages = []
    if 'messages' in response:
        messages += response['messages']
        logging.info('Messages for channel: %s', str(len(response['messages'])))
    if 'response_metadata' in response:
        meta = response["response_metadata"]
        if 'next_cursor' in meta:
            next_cursor = response["response_metadata"]["next_cursor"]
            if (next_cursor != "" and next_cursor != None):
                result = get_all_conversation_history(app, channel, next_cursor)
                if result != None :
                    messages += result
    return messages

@sleep_and_retry
@limits(calls=TIER_5_RATELIMIT_PER_MINUTE, period=ONE_MINUTE)
def get_message_permalink(app, channel, ts):
    link = ""
    try :
        response = app.client.chat_getPermalink(
            channel=channel,
            message_ts=ts)
        link = ""
        if 'permalink' in response:
            link = response['permalink']
    except:
        logging.info('ERROR retrieving permalink for : %s', str(ts))
        link = ""
    return link

@sleep_and_retry
@limits(calls=TIER_3_RATELIMIT_PER_MINUTE, period=ONE_MINUTE)
def get_all_thread_history(app, channel, ts, cursor = None):
    response = app.client.conversations_replies(
        channel=channel,
        limit=500,
        ts=ts,
        cursor=cursor)
    messages = []
    if 'messages' in response:
        messages += response['messages']
        logging.info('thread history msgs for %s: %s', str(ts), str(len(response['messages'])))
    if 'response_metadata' in response:
        meta = response["response_metadata"]
        if 'next_cursor' in meta:
            next_cursor = response["response_metadata"]["next_cursor"]
            if (next_cursor != "" and next_cursor != None):
                result = get_all_thread_history(app, channel, ts, next_cursor)
                if result != None :
                    messages += result
    return messages

@sleep_and_retry
@limits(calls=TIER_2_RATELIMIT_PER_MINUTE, period=ONE_MINUTE)
def get_all_user_list(app, cursor = None):
    user_response = app.client.users_list(
        limit=500,
        cursor=cursor)
    users = []
    #logging.info('USER RESPONSE: %s', user_response)
    if 'members' in user_response:
        users += user_response['members']
        logging.info('members retrieved: %s', str(len(user_response['members'])))
    if 'response_metadata' in user_response:
        meta = user_response["response_metadata"]
        if 'next_cursor' in meta:
            next_cursor = user_response["response_metadata"]["next_cursor"]
            if (next_cursor != "" and next_cursor != None):
                result = get_all_user_list(app, next_cursor)
                if result != None :
                    users += result
    return users

def get_channel_convos(app, channel):
    member_dict = {}
    #TODO : get and store this in DB so we don't need to get each time.  Also store in local memory for APP
    members = get_all_user_list(app)
    if (members == None): 
        members = []
    for member in members:
        member_dict[member['id']] = member['profile']

    # Write message log to Supabase
    # write_message_log(user_name=user_diplay_name, message=msg)
 
    messages = get_all_conversation_history(app, channel)
    if (messages == None):
        messages = []
    logging.info('Channel History Count: %s', len(messages))
    if (len(messages)>2):
        logging.info('Channel History last: %s', messages[2])
    document = ""
    all_convos = []
    all_convos_links = []
    numMessages = 0
    prevMsgTimestamp = 0.0
    last_ts = 0.0
    top_ts = 0.0 #this is the top ts of the convo to link to
    reset_top_ts = False
    lenConvo = numConvos = 0
    BORDER_TEXT = '=============================='
    logging.info('START PROCESSING ALL MESSAGES: %s', str(len(messages)))

    for message in reversed(messages):
        #skip messages that don't match message type
        if (message['type'] == 'message' and 'user' in message):
            name = message['user']
            if (message['user'] in member_dict):
                if 'real_name' in member_dict[message['user']]:
                    name = member_dict[message['user']]['real_name']
                elif 'display_name' in member_dict[message['user']]:
                    name = member_dict[message['user']]['display_name']
                else:
                    logging.info('NAME NOT FOUND: %s', message)
            # else:
            #     logging.info('NAME NOT FOUND: %s', message)

            logging.info('\n\nMESSAGE: %s', message)

            str_ts = message['ts']
            ts = float(message['ts'])
            last_ts = ts
            if (top_ts == 0) :
                top_ts = ts
            #this_date = time.ctime(ts) #friendly text datetime
            #current_ts = time.time()

            #check if this is a thread - if so treat as one convo for entire thread
            if 'thread_ts' in message:
                # if this is the parent to the thread - see if we need to cut off the convo from other parent messages
                # Note: I determined that a thread might start with normal channel message and then after another channel msg
                #  it continues in a thread - so this convo might start with channel messages and then continue with full thread.
                #  regardless, I am ending the convo with end of thread since it would be wierd to have 2 parrallel series of 
                #  messages (one in thread and one in channel) since those don't follow the same convo usually.
                if ts - prevMsgTimestamp > 86400.0:
                    link = get_message_permalink(app, channel, top_ts)
                    all_convos_links.append(link)
                    all_convos.append(document)
                    logging.info('END OF CONVO: URL: %s AND DOC LEN: %s', str(link), str(len(document)))
                    document = ""
                    top_ts = str_ts
                    numConvos += 1
                prevMsgTimestamp = ts

                thread_ts = message['thread_ts']
                thread_messages = get_all_thread_history(app, channel, thread_ts)
                if (thread_messages == None) :
                    thread_messages = []
                # loop through the thread and add all messages to one convo
                for thread_message in thread_messages:
                    if (thread_message['type'] == 'message' and 'user' in thread_message):
                        name = thread_message['user']
                        if (thread_message['user'] in member_dict):
                            if 'real_name' in member_dict[thread_message['user']]:
                                name = member_dict[thread_message['user']]['real_name']
                            elif 'display_name' in member_dict[thread_message['user']]:
                                name = member_dict[thread_message['user']]['display_name']
                        document += F"{name}: {thread_message['text']}\n\n"
                        numMessages += 1

                # now close the convo
                link = get_message_permalink(app, channel, thread_ts)
                all_convos_links.append(link)
                all_convos.append(document)
                logging.info('END OF CONVO: URL: %s AND DOC LEN: %s', str(link), str(len(document)))
                document = ""
                top_ts = thread_ts
                reset_top_ts = True
                numConvos += 1
            else:
                #this is not a thread - so just keep going normally.

                # check if this message is more than 24 hours since last and separate 
                # dealing with seconds (60x60x24)
                if ((ts - prevMsgTimestamp > 86400.0) and (document != "")):
                    #document += F"{BORDER_TEXT}\n\n"
                    #instead of making one large document with separators - lets add convos to array and join
                    link = get_message_permalink(app, channel, top_ts)
                    all_convos_links.append(link)
                    all_convos.append(document)
                    logging.info('END OF CONVO: URL: %s AND DOC LEN: %s', str(link), str(len(document)))
                    document = ""
                    top_ts = str_ts
                    numConvos += 1

                prevMsgTimestamp = ts
                if (reset_top_ts):
                    reset_top_ts = False
                    top_ts = str_ts
                #add the message to the convo with Name: Message format
                document += F"{name}: {message['text']}\n\n"
                numMessages += 1
        else:
            logging.info('UNHANDLED MESSAGE: %s', message)
    logging.info('DONE PROCESSING - NUM CONVOS: %s', str(len(all_convos)))

    #add the last convo to array
    if (document != "") :
        link = get_message_permalink(app, channel, top_ts)
        all_convos_links.append(link)
        all_convos.append(document)
        document = ""
        numConvos += 1

    return all_convos, all_convos_links, last_ts

def slack_store_channel(ack, app, say, body):
    logging.info('BODY: %s', body)
    channel = body["channel_id"]
    channel_name = body["channel_name"]
    ack_message_id = send_slack_message_and_return_message_id(
        app=app, channel=channel, message=get_random_thinking_message())

    reference_name = body['text']

    # If user didn't include a URL or URLs, then abort
    if (reference_name == "" or reference_name is None):
        say("Please enter the name to associate with this summary. This will be the official company name for client channels. example: /summary Medbridge")
        return
  

    (documents, links, last_ts) = get_channel_convos(app, channel)
    #logging.info('ALL CONVOS: %s', documents)


    #logging.info('summary: %s', document)
    (summary, longSummary) = storeEmbeddings(documents, links, last_ts, channel, channel_name, reference_name)

    replyTs = ack_message_id
    # Replace acknowledgement message with actual response
    # if (len(summary) > 3000) :
    #     app.client.chat_update(
    #         channel=channel,
    #         text="Longer Summary will be added below:",
    #         ts=ack_message_id
    #     )
    #     if (len(summary) < 40000) :
    #         thisPost = app.client.chat_postMessage(
    #             channel=channel,
    #             unfurl_links=False, 
    #             unfurl_media=False,
    #             text=summary
    #         )
    #         # if this message gets split into multiple this looks weird on the last one - so lets just put it on the ack_message
    #         # if ("ts" in thisPost):
    #         #     replyTs = thisPost["ts"]
    #     else : 
    #         app.client.chat_postMessage(
    #             channel=channel,
    #             text="Message too long : report this error"
    #         )
    # else :
    #     app.client.chat_update(
    #         channel=channel,
    #         text=summary,
    #         unfurl_links=False, 
    #         unfurl_media=False,
    #         ts=ack_message_id
    #     )
    #reply with longer summary
    app.client.chat_update(
        channel=channel,
        text="Channel Summary added as reply:",
        ts=ack_message_id
    )
    app.client.chat_postMessage(
        channel=channel,
        unfurl_links=False, 
        unfurl_media=False,
        thread_ts=replyTs,
        text="Longer version of the summary: " + longSummary
    )


def slack_summarize_channel(ack, app, say, body):
    msg = body['text']
    channel = body["channel_id"]
    ack_message_id = send_slack_message_and_return_message_id(
        app=app, channel=channel, message=get_random_thinking_message())

    (documents, links, last_ts) = get_channel_convos(app, channel)
    document = '==============================\n\n'.join(documents)


    #logging.info('summary: %s', document)
    summary = summarizeLongDocument(document, "")
    logging.info('summary: %s', summary)



def slack_respond_with_new_agent(agent, event, ack, app):
    channel = event["channel"]

    # Acknowledge user's message
    ack()
    ack_message_id = send_slack_message_and_return_message_id(
        app=app, channel=channel, message=get_random_thinking_message())
    
    user_query = event["text"]

    response = handle_multi_step_query(agent=agent, query=user_query, messages_history=[])
    #response = query_sql_db(user_query)

    if (len(response) > 3000) :
        app.client.chat_update(
            channel=channel,
            text="Longer Response will be added below:",
            ts=ack_message_id
        )
        app.client.chat_postMessage(
            channel=channel,
            text=f"```\n{response}\n```"
        )
    else :
        app.client.chat_update(
            channel=channel,
            text=response,
            ts=ack_message_id
        )


def slack_respond_with_agent(agent, event, ack, app):
    """
    This function takes a Slack message event and respond with a LLM-generated response
    """

    channel = event["channel"]

    # Acknowledge user's message
    ack()
    ack_message_id = send_slack_message_and_return_message_id(
        app=app, channel=channel, message=get_random_thinking_message())

    # Get the conversation history (last 5 messages)
    messages_history = []
    conversation_history = app.client.conversations_history(
        channel=channel, limit=5)
    messages_history.extend(extract_messages(conversation_history))

    # Give the bot context of about the user (changed from first name to display name so it unique for history)
    user_id = event["user"]
    user_real_name = app.client.users_info(
        user=user_id)['user']['profile']['real_name']  # type: ignore
    user_diplay_name = app.client.users_info(
        user=user_id)['user']['profile']['display_name']  # type: ignore
    messages_history.append(
        {"type": "user", "message": f"""My name is {user_real_name}"""})
    messages_history.append(
        {"type": "AI", "message": f"""I'm a knowledge assistant at {demo_company_name}. My name is {ai_name}. I only answer questions cheerfully about {demo_company_name} the company."""})

    # Write message log to Supabase
    write_message_log(user_name=user_diplay_name, message=event["text"])

    # Generate an LLM response using agent
    user_query = event["text"]

    response = get_agent_response(
        agent=agent, query=user_query, messages_history=messages_history)

    blocks = [
        {
            "type": "section",
            "text": {
                    "type": "mrkdwn",
                    "text": response
            },
        },
        {
            "type": "actions",
            "block_id": "actionblock789",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "👍"
                    },
                    "style": "primary",
                    "value": "feedback_good"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "👎"
                    },
                    "style": "primary",
                    "value": "feedback_bad"
                }
            ]
        }
    ]

    # Replace acknowledgement message with actual response
    app.client.chat_update(
        channel=channel,
        text=response,
        ts=ack_message_id,
        blocks=blocks
    )

    # Write message log to Supabase
    write_message_log("AI", response)
    
    
def slack_respond_with_general_agent(agent, ack, app, say, body):
    """
    This function takes a Slack message /bot query and responds with a LLM-generated response
    """

    # Acknowledge user's message
    msg = body['text']
    channel = body["channel_id"]
    ack_message_id = send_slack_message_and_return_message_id(
        app=app, channel=channel, message=get_random_thinking_message())

    # Get the conversation history (last 5 messages)
    messages_history = []
    conversation_history = app.client.conversations_history(
        channel=channel, limit=5)
    messages_history.extend(extract_messages(conversation_history))

    # Give the bot context of about the user (changed from first name to display name so it unique for history)
    user_id = body["user_id"]
    user_real_name = app.client.users_info(
        user=user_id)['user']['profile']['real_name']  # type: ignore
    user_diplay_name = app.client.users_info(
        user=user_id)['user']['profile']['display_name']  # type: ignore
    messages_history.append(
        {"type": "user", "message": f"""My name is {user_real_name}"""})
    messages_history.append(
        {"type": "AI", "message": f"""I'm a large language model chatbot assistant at {demo_company_name} trained by OpenAI. My name is {ai_name}. I answer questions about anything as factually as I can based on the information I know and will attempt to provide references to where my answer came from."""})

    # Write message log to Supabase
    write_message_log(user_name=user_diplay_name, message=msg)

    # Generate an LLM response using agent
    user_query = msg

    response = get_agent_response(
        agent=agent, query=user_query, messages_history=messages_history)

    blocks = [
        {
            "type": "section",
            "text": {
                    "type": "mrkdwn",
                    "text": response
            },
        },
        {
            "type": "actions",
            "block_id": "actionblock789",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "👍"
                    },
                    "style": "primary",
                    "value": "feedback_good"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "👎"
                    },
                    "style": "primary",
                    "value": "feedback_bad"
                }
            ]
        }
    ]

    # Replace acknowledgement message with actual response
    app.client.chat_update(
        channel=channel,
        text=response,
        ts=ack_message_id,
        blocks=blocks
    )

    # Write message log to Supabase
    write_message_log("AI", response)

def slack_respond_to_gpt_conversation(agent, ack, app, say, body):

    # Acknowledge user's message
    msg = body['text']
    channel = body["channel_id"]
    ack_message_id = send_slack_message_and_return_message_id(
        app=app, channel=channel, message=get_random_thinking_message())

    # Give the bot context of about the user (changed from first name to display name so it unique for history)
    user_id = body["user_id"]
    user_real_name = app.client.users_info(user=user_id)['user']['profile']['real_name']  # type: ignore
    user_diplay_name = app.client.users_info(user=user_id)['user']['profile']['display_name']  # type: ignore

    # Write message log to Supabase
    write_message_log(user_name=user_diplay_name, message=msg)
    
    
    my_messages = [
    	SystemMessage(content=f"You are large language model chatbot assistant at {demo_company_name} trained by OpenAI. Your name is {ai_name}. You answer questions about anything, as factually as you can based on the information you know, and will attempt to provide references to where your answers came from."),
    	HumanMessage(content=f"My name is {user_real_name}"),
    	HumanMessage(content=msg)
	]
    response = llm(my_messages)
    logging.info('llm results: %s', response)
    
    logging.info('llm results: %s', response.content)
    # Prompt
    chat_id = f"chat_history_{user_id}"
    prompt=ChatPromptTemplate.from_messages([
		SystemMessagePromptTemplate.from_template(
			"You are large language model chatbot assistant at {demo_company_name} trained by OpenAI. Your name is {ai_name}. You answer questions about anything, as factually as you can based on the information you know, and will attempt to provide references to where your answers came from."
		),
		HumanMessagePromptTemplate.from_template("My name is {user_real_name}"),
		HumanMessagePromptTemplate.from_template("{question}")
	])
	# MessagesPlaceholder(variable_name=chat_id),

    messages = prompt.format_messages(
		demo_company_name=demo_company_name,
		ai_name=ai_name,
		user_real_name=user_real_name,
		question=msg
	)
    logging.info('messages: %s', messages)

	# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
	# Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    # memory = ConversationBufferMemory(memory_key=chat_id,return_messages=True)
    conversation = LLMChain(
		llm=llm,
		prompt=prompt,
		verbose=True,
		memory=ConversationBufferMemory()
	)
	#memory=memory
	
	# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    # response = conversation.run({"question": msg})

    # logging.debug('RESPONSE ****: %s', response)
	
    blocks = [
        {
            "type": "section",
            "text": {
                    "type": "mrkdwn",
                    "text": response.content
            },
        },
        {
            "type": "actions",
            "block_id": "actionblock789",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "👍"
                    },
                    "style": "primary",
                    "value": "feedback_good"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "emoji": True,
                        "text": "👎"
                    },
                    "style": "primary",
                    "value": "feedback_bad"
                }
            ]
        }
    ]

    # Replace acknowledgement message with actual response
    app.client.chat_update(
        channel=channel,
        text=response.content,
        ts=ack_message_id,
        blocks=blocks
    )

    #write_message_log("AI", f'{response}')
