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
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


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
llm_conv=ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model=model_name)
embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key, chunk_size=3500)
#toolkit = SQLDatabaseToolkit(db=supabase, llm=llm)
db_url = f'postgresql://{supabase_user}:{supabase_password}@{supabase_host}:5432/postgres'
db = SQLDatabase.from_uri(db_url)
vectordb = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents")

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
    
def storeVectorsInDB(vectors, batch, ids, embeddings2):
  vectordb2 = SupabaseVectorStore(client=supabase, embedding=embeddings2, table_name="documents")
  result = vectordb2.add_vectors(vectors, batch, ids)
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

def getDocumentConversationChain():
    # set the retirever to use similarity and return 4 result chunks
    vectordb_retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )
    # search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}

    qa_template = """Use the following CONTEXT to answer the question at the end to the best of your ability. You should think critically about the question and make inferences as needed from related context or conversation. 
    If the question asked has no relevant context available, you should attempt to answer the question based on the information you have been trained on as a LLM, but if you are not confident on the answer, just respond with nothing for the answer, don't try to make up an answer.\n
    ------------\n
    CONTEXT: \n{context}\n
    ------------\n
    Question: {question}\n
    Answer:"""
    QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=qa_template
    )

    condense_template = ("Given the following conversation and a ""Follow Up Question"", rephrase the follow up question to be a standalone question, in its original language.  If the follow up question does not directly reference the conversation history, then just use the original follow up question as the new standalone question.\n\n"
        "______________________\n"
        "Chat History:\n"
        "{chat_history}\n"
        "Follow Up Question: {question}\n"
        "Standalone question:"
        "______________________\n"
        "Then, once you have the standalone question and given the context provided below, please reformulate the standalone question, considering the different ways team members might naturally discuss topics related to the query in a workplace setting. When reformulating, think about direct and indirect references, synonyms, related concepts, and any colloquial or jargon terms they might use. The goal is to produce a revised standalone question that captures the essence of the original standalone question but is optimized for retrieving relevant conversations from a vector database of natural language discussions using similarity search.\n\n"
        "Context: The vector database contains Slack conversations among team members at our company. These conversations are diverse and cover various aspects of multiple projects. The language used is natural, conversational, and can range from casual to technical discussions, often sprinkled with workplace jargon.\n\n"
        "Reformulated Standalone question:"
    )
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
    #"Finally, once you have reformulated the standalone question, you should respond with the Standalone Question appended to the Follow up question as your final answer.\n\n"

    combine_template = (
        "Given the following extracted summaries from multiple documents or conversations, answer the final question.\n"
        "You should disregard any parts of the summaries that say ""I don't know"" or state something similar to ""based on the conversations there is no information provided about the question.\n"
        "Prioritize summaries which help answer the question over ones that do not.\n"
        "______________________\n"
        "Summaries:\n{summaries}"
        "______________________\n"
        "Human: {question}\n"
        "AI: "
    )
    COMBINE_PROMPT = PromptTemplate(input_variables=["summaries", "question"], template=combine_template)


    memory = init_memory()

    # Create the custom chain
    doc_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_conv, retriever=vectordb_retriever, memory=memory,
        get_chat_history=lambda h : h, return_source_documents=False,
        chain_type="map_reduce", condense_question_prompt=CONDENSE_QUESTION_PROMPT, rephrase_question=True,
        combine_docs_chain_kwargs={'question_prompt': QA_PROMPT, "combine_prompt": COMBINE_PROMPT}, verbose=True)
    
    return doc_retrieval_chain
