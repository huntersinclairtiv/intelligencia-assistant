import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

# Load .env variables
load_dotenv()

# Set API keys from session state
# openai_api_key = st.session_state.openai_api_key
# serper_api_key = st.session_state.serper_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Streamlit app
st.subheader('Web Search')
search_query = st.text_input("Enter Search Query")

# If the 'Search' button is clicked
if st.button("Search"):
    # Validate inputs
    if not openai_api_key or not serper_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not search_query.strip():
        st.error("Please provide the search query.")
    else:
        try:
            with st.spinner('Please wait...'):
              # Initialize the OpenAI module, load the Google Serper API tool, and run the search query using an agent
              llm = OpenAI(temperature=0, openai_api_key=openai_api_key, verbose=True)
              tools = load_tools(["google-serper"], llm, serper_api_key=serper_api_key)
              agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
              result = agent.run(search_query)
              st.success(result)
        except Exception as e:
            st.exception(e)
