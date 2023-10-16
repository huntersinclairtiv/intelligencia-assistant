import os
import streamlit as st, tiktoken
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

# Load .env variables
load_dotenv()

# Set API keys from session state
# openai_api_key = st.session_state.openai_api_key
# serper_api_key = st.session_state.serper_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Streamlit app
st.subheader('News Summary')
num_results = st.number_input("Number of Search Results", min_value=3, max_value=10) 
search_query = st.text_input("Enter Search Query")
col1, col2 = st.columns(2)

# If the 'Search' button is clicked
if col1.button("Search"):
    # Validate inputs
    if not openai_api_key or not serper_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not search_query.strip():
        st.error("Please provide the search query.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Show the top X relevant news articles from the previous week using Google Serper API
                search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=serper_api_key)
                result_dict = search.results(search_query)

                if not result_dict['news']:
                    st.error(f"No search results for: {search_query}.")
                else:
                    for i, item in zip(range(int(num_results)), result_dict['news']):
                        st.success(f"Title: {item['title']}\n\nLink: {item['link']}\n\nSnippet: {item['snippet']}")
        except Exception as e:
            st.exception(e)

# If 'Search & Summarize' button is clicked
if col2.button("Search & Summarize"):
    # Validate inputs
    if not openai_api_key or not serper_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not search_query.strip():
        st.error("Please provide the search query.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Show the top X relevant news articles from the previous week using Google Serper API
                search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=serper_api_key)
                result_dict = search.results(search_query)

                if not result_dict['news']:
                    st.error(f"No search results for: {search_query}.")
                else:
                    # Load URL data from the top X news search results
                    for i, item in zip(range(int(num_results)), result_dict['news']):
                        loader = UnstructuredURLLoader(urls=[item['link']])
                        data = loader.load()

                        # Initialize the ChatOpenAI module, load and run the summarize chain
                        llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=openai_api_key)
                        chain = load_summarize_chain(llm, chain_type="map_reduce")
                        summary = chain.run(data)

                        st.success(f"Title: {item['title']}\n\nLink: {item['link']}\n\nSummary: {summary}")
        except Exception as e:
            st.exception(e)
