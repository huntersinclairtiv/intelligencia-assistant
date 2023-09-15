from ai.ai_agents import get_agent_response
from slack.slack_utils import get_random_thinking_message, send_slack_message_and_return_message_id
from utils import extract_messages
from consts import demo_company_name, ai_name
from supabase_wrapper import write_message_log


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

    # Give the bot context of about the user (changed from frist name to display name so it unique for history)
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

    # Give the bot context of about the user (changed from frist name to display name so it unique for history)
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

