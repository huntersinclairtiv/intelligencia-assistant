import os
import validators, streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# Load .env variables
load_dotenv()

# Set API keys from session state
# openai_api_key = st.session_state.openai_api_key
# serper_api_key = st.session_state.serper_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Streamlit app
st.subheader('URL Summary')
url = st.text_input("Enter Source URL")

# If 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not url:
        st.error("Please provide the URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Please wait..."):
                # Load URL data
                loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                data = loader.load()
                
                # Initialize the ChatOpenAI module, load and run the summarize chain
                llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=openai_api_key)
                prompt_template = """Write a summary of the following in 200-250 words:
                    
                    {text}

                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(data)

                st.success(summary)
        except Exception as e:
            st.exception(e)
