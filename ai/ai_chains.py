import os, tempfile
from typing import List, Dict, Any
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
from supabase.client import Client, create_client, Timeout
from supabase.lib.client_options import ClientOptions
from langchain.prompts import PromptTemplate
from streamlit_chat import message
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from sqlalchemy import text
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain


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
model_name = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo"


client_options = ClientOptions(postgrest_client_timeout=Timeout(None))
supabase: Client = create_client(supabase_url, supabase_key, options=client_options) # type: ignore
# Initialize the OpenAI module, load and run the summarize chain
llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model=model_name)
embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key, chunk_size=3500)
#toolkit = SQLDatabaseToolkit(db=supabase, llm=llm)
db_url = f'postgresql://{supabase_user}:{supabase_password}@{supabase_host}:5432/postgres'
db = SQLDatabase.from_uri(db_url)
vectordb = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name= "documents")

@st.cache_resource
def init_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return memory

def query_sql_db(query: str):
    # Validate inputs
    if not query:
        return("Please provide a question.")
    else:
        #try:
        db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True, top_k=20)
        db_query_results = db_chain.run(query)
        logging.info('DB SQL results: %s', db_query_results)
        return db_query_results.replace("Final answer here: ", "")
    
def storeVectorsInDB(vectors, batch, ids):
  result = vectordb.add_vectors(vectors, batch, ids)
  return result

def storeSummaryInDB(full_summary, last_ts, channel: str, channel_name: str, reference_name: str):
    rows: List[Dict[str, Any]] = [
        {
            "summary": full_summary,
            "last_ts": last_ts,
            "channel": channel,
            "channel_name": channel_name,
            "reference_name": reference_name,
        }
    ]

    logging.info('ADDING SUMMARY TO DB : %s', str(len(rows)))

    table_name = "slack_channel_summaries"
    db_result = supabase.from_(table_name).upsert(rows).execute()  # type: ignore

    if len(db_result.data) == 0:
        raise Exception("Error inserting summary: No rows added")
    logging.info('db_result: %s', db_result)
    return db_result

