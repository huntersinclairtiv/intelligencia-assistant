import os
import streamlit as st
import logging
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from ai.ai_functions import load_urls_and_overwrite_index
from ai.ai_agents import initialize_retrieval_agent, initialize_general_agent
from slack.slack_utils import is_dm
from slack.slack_utils import get_random_thinking_message
from slack.slack_functions import slack_respond_with_agent, slack_respond_with_general_agent, slack_respond_to_gpt_conversation

# logger in a global context
# requires importing logging
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# Slack App Initialization
bot_token = os.environ["SLACK_BOT_TOKEN"]
app_token = os.environ["SLACK_APP_TOKEN"]
bot_id = os.environ["SLACK_APP_ID"]
app = App(token=bot_token)

# Initialize agent
agent = initialize_retrieval_agent()
general_agent = initialize_general_agent()


# Handle incoming DMs
@app.event("message")
def handle_message_events(event, ack):
    if (is_dm(event)):
        slack_respond_with_agent(agent=agent, ack=ack, app=app, event=event)
    return


@app.event("app_mention")
def handle_mention(event, ack):
    slack_respond_with_agent(agent=agent, ack=ack, app=app, event=event)


@app.command("/upload-new-doc")
def handle_document_upload(body, say, ack):
    ack()
    value = body['text']

    # If user didn't include a URL or URLs, then abort
    if (value == "" or value is None):
        say("Please enter a valid URL to the document!")
        return

    say("I'm uploading a new document! :arrow_up:")

    # Load the URLs into vectorstore
    load_urls_and_overwrite_index(value)

    say("I'm done uploading the document! :white_check_mark:")

@app.command("/ai")
def handle_ai_query(body, say, ack):
    ack()
    value = body['text']

    # If user didn't include a URL or URLs, then abort
    if (value == "" or value is None):
        say("Please enter a query for ChatGPT.")
        return

    slack_respond_to_gpt_conversation(agent=general_agent, ack=ack, app=app, say=say, body=body)

@app.command("/bot")
def handle_bot_query(body, say, ack):
    ack()
    value = body['text']

    # If user didn't include a URL or URLs, then abort
    if (value == "" or value is None):
        say("Please enter a query for ChatGPT.")
        return

    slack_respond_with_general_agent(agent=general_agent, ack=ack, app=app, say=say, body=body)


def run_slack_app():
    SocketModeHandler(app, app_token).start()
