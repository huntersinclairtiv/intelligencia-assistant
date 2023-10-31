import os
import sys
import streamlit as st
from dotenv import load_dotenv
from slack.slack import run_slack_app

#st.set_page_config(page_title="Home", page_icon="🦜️🔗")

# Load .env variables
#load_dotenv()

st.header("Welcome to Intelligencia! 👋")

st.markdown(
    """
   ##### Web Search
    * A sample app for web search queries using LangChain and Serper API.
    * References: Blog | [Source Code](https://github.com/alphasecio/langchain-examples/blob/main/search) | [Python Notebook](https://github.com/alphasecio/langchain-examples/blob/main/search/langchain_search.ipynb)
    * *Note: This search app has been modified to use Serper API instead of SerpApi.*

    ##### URL Summary
    * A sample app for summarizing URL content using LangChain and OpenAI.
    * References: [Blog](https://alphasec.io/blinkist-for-urls-with-langchain-and-openai) | [Source Code](https://github.com/alphasecio/langchain-examples/blob/main/url-summary)

    ##### Text Summary
    * A sample app for summarizing text using LangChain and OpenAI.
    * References: [Blog](https://alphasec.io/summarize-text-with-langchain-and-openai) | [Source Code](https://github.com/alphasecio/langchain-examples/blob/main/text-summary) | [Python Notebook](https://github.com/alphasecio/langchain-examples/blob/main/text-summary/langchain_text_summarizer.ipynb)

    ##### Document Query
    * A sample app for querying against documents using LangChain, OpenAI and Supabase.
    * References: [Blog](https://alphasec.io/summarize-documents-with-langchain-and-chroma) | [Source Code](https://github.com/alphasecio/langchain-examples/blob/main/chroma-summary) | [Python Notebook](https://github.com/alphasecio/langchain-examples/blob/main/chroma-summary/langchain_doc_summarizer.ipynb)

    ##### News Summary
    * A sample app for Google news search and summaries using LangChain and Serper API.
    * References: [Blog](https://alphasec.io/summarize-google-news-results-with-langchain-and-serper-api) | [Source Code](https://github.com/alphasecio/langchain-examples/blob/main/news-summary)

    ##### SQL DB Query
    * A sample app for querying against a SQL DB using LangChain, OpenAI and Supabase.
    """
)
slackHandler = None
if __name__ == "__main__":
    slackHandler = run_slack_app()

if st.button("Quit App"):
    if (slackHandler != None):
        slackHandler.close()
    sys.exit("Exited the App")