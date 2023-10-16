import os, tempfile
from typing import List
import streamlit as st
import logging
import langchain
import sqlalchemy
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory.buffer import ConversationBufferMemory
from supabase.client import Client, create_client
from supabase.lib.client_options import ClientOptions
from langchain.prompts import PromptTemplate
from streamlit_chat import message
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from sqlalchemy import text
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


langchain.debug = True
langchain.verbose = True

# requires importing logging
logging.basicConfig(level=logging.INFO)

# Load .env variables
load_dotenv()

# Set API keys from session state
# openai_api_key = st.session_state.openai_api_key
# serper_api_key = st.session_state.serper_api_key
openai_api_key = os.getenv("OPENAI_API_KEY") 
serper_api_key = os.getenv("SERPER_API_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_user = os.environ.get("SUPABASE_USER")
supabase_password = os.environ.get("SUPABASE_PASSWORD")
supabase_host = os.environ.get("SUPABASE_HOST")
model_name = os.environ.get("MODEL_NAME")

st.set_page_config(
    page_title="Thinktiv Chat - Demo",
    page_icon=":robot:"
)

client_options = ClientOptions(postgrest_client_timeout=0)
supabase: Client = create_client(supabase_url, supabase_key, options=client_options) # type: ignore
# Initialize the OpenAI module, load and run the summarize chain
llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model=model_name) # type: ignore
#toolkit = SQLDatabaseToolkit(db=supabase, llm=llm)
db_url = f'postgresql://{supabase_user}:{supabase_password}@{supabase_host}:5432/postgres'
db = SQLDatabase.from_uri(db_url)


##########
# Example using Alchemy for Querying
##########
# engine = sqlalchemy.create_engine(db_url)
# metadata = sqlalchemy.MetaData()

# table = sqlalchemy.Table(
#     'test_table', 
#     metadata, 
#     autoload_with=engine
# )

# rs = []
# with engine.connect() as con:
#     rs = con.execute(text('SELECT * FROM test_table'))

# for row in rs:
#     logging.info('SQL results: %s', row)

@st.cache_resource
def init_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return memory

# Streamlit app
st.subheader('Query DB')
search_query = st.text_input("Enter Search Query")

if st.button("Query"):
    # Validate inputs
    if not openai_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not search_query:
        st.error("Please provide a question.")
    else:
        #try:
        with st.spinner('Please wait...'):
            db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
            db_query_results = db_chain.run(search_query)
            logging.info('DB SQL results: %s', db_query_results)
            st.success(db_query_results.replace("Final answer here: ", ""))
