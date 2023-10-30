import os
import langchain
import logging
import streamlit as st
from dotenv import load_dotenv
from typing import List
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from consts import llm_model_type
from ai.ai_tools import tool_describe_skills, tool_retrieve_company_info, tool_calculate_stock_options
from ai.ai_chains import query_sql_db, getDocumentConversationChain
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain import hub
from supabase.client import Client, create_client
from supabase.lib.client_options import ClientOptions
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain, SQLDatabaseSequentialChain
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import SystemMessage


langchain.debug = True
langchain.verbose = True

# requires importing logging
logging.basicConfig(level=logging.INFO)

# Load .env variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_user = os.environ.get("SUPABASE_USER")
supabase_password = os.environ.get("SUPABASE_PASSWORD")
supabase_host = os.environ.get("SUPABASE_HOST")
model_name = os.environ.get("MODEL_NAME") or "gpt-3.5-turbo"

#DB Init
client_options = ClientOptions(postgrest_client_timeout=0)
supabase: Client = create_client(supabase_url, supabase_key, options=client_options) # type: ignore
# Initialize the OpenAI module, load and run the summarize chain
#toolkit = SQLDatabaseToolkit(db=supabase, llm=llm)
db_url = f'postgresql://{supabase_user}:{supabase_password}@{supabase_host}:5432/postgres'
db = SQLDatabase.from_uri(db_url)

# LLM Initialization
llm = ChatOpenAI(max_retries=3, temperature=0.1, model=model_name)
llm_simple = ChatOpenAI(max_retries=3, temperature=0.8, model=model_name)

@st.cache_resource
def init_memory():
    memory = ConversationBufferMemory(memory_key='query_history', return_messages=True, output_key='answer')
    return memory

def initialize_conversational_agent(tools: List = [], is_agent_verbose: bool = True, max_iterations: int = 3, return_thought_process: bool = False):
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Initialize agent
    agent = initialize_agent(
        tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=is_agent_verbose, max_iterations=max_iterations, return_intermediate_steps=return_thought_process, memory=memory)

    return agent


def get_agent_response(agent, query: str, messages_history: List):
    """
    This function takes a query, a list of tools, and some optional parameters, initializes a
    conversational LangChain agent, and returns the agent's response to the query.
    """

    # Add messages to memory
    for message_dict in messages_history:
        message_value = message_dict['message']
        sender = message_dict['type']
        if (sender == 'user'):
            agent.memory.chat_memory.add_user_message(message_value)
        if (sender == 'AI'):
            agent.memory.chat_memory.add_ai_message(message_value)

    # Get results from chain
    try:
        result = agent({"input": query})
        answer = result["output"]

        # Debug agent's answer to console
        print(answer)

        # Clear the agent's memory after generating response so they don't bring that to another person
        agent.memory.chat_memory.clear()

        return answer

    except Exception:
        return "Sorry, I had trouble answering your question. Please ask again 🥹"


def initialize_general_agent(tools: List = [], is_agent_verbose: bool = True, max_iterations: int = 3, return_thought_process: bool = False):
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Initialize agent
    agent = initialize_agent(
        tools, llm_simple, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=is_agent_verbose, max_iterations=max_iterations, return_intermediate_steps=return_thought_process, memory=memory)

    return agent

def initialize_retrieval_agent():
    tools = [tool_retrieve_company_info(),
             tool_describe_skills(),
             tool_calculate_stock_options()
             ]
    return initialize_conversational_agent(tools=tools)


def initialize_basic_agent(is_agent_verbose: bool = True, max_iterations: int = 30, return_thought_process: bool = False):
    #search = SerpAPIWrapper(search_engine="google")

    #TODO: need to figure out how to allow limit in query to override top_k
    db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True, top_k=1000)

    # tools = load_tools(["google-serper"], llm, serper_api_key=serper_api_key)
    # tools[0].description = (
    #     "A Google Search API."
    #     "Useful for when you need to answer questions about current events, the current state of the world, or public information about companies and people."
    #     "Input should be a search query."
    # )

    document_chain = getDocumentConversationChain()

    conversation_chain = ConversationChain(
		llm=llm_simple,
		verbose=True,
        memory=ConversationBufferMemory())
    
    search_tools = load_tools(["google-serper"], llm, serper_api_key=serper_api_key)
    search_tools[0].description = (
        "A Google Search API."
        "Useful for when you need to answer questions about current events, the current state of the world, or public information about companies and people."
        "Input should be a search query."
    )
    search_agent = initialize_agent(search_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=30, handle_parsing_errors=True)

    tools = [
        Tool(
            name = "Current Search",
            func=search_agent.run,
            description=(
                "A Google Search API." 
                "Useful for when you need to answer questions about current events, the current state of the world, or public information about companies and people."
                "Input should be a search query."
            )
        ),
        Tool(
            name = "Slack Channel Query",
            func=lambda e: document_chain({"question": e}),
            description= (
                "Useful for when you need to answer questions about employee conversations related to a company, program or within a slack channel. You should pass the original question in its entirety to this tool."
            )
        ),
        Tool(
            name = "Conversation Query",
            func=conversation_chain.run,
            description= (
                "Useful for when you need to answer general questions or have a more general conversation. Also should be used as default tool if no other tool is appropriate. "
            )
        ),
    ]
    #func=lambda e: document_chain({"question": e}),
    # Tool(
    #     name = "Company Database Query",
    #     func=db_chain.run,
    #     description= (
    #         "Useful for when you need to answer questions about structured company data. "
    #         "Examples include queries about client companies, programs, projects, or people. "
    #         "The database only contains limited information and may require use of another tool to expand the metadata related to each. "
    #         "Input should be the Human's natural language query. Input should not attempt to use SQL. "
    #     )
    # ),
    memory = ConversationBufferMemory(memory_key="chat_history")

    #system_message = SystemMessage(content="You are a helpful AI chat assistant for employees at a company named Thinktiv.  We communicate with you within chat channels of the Slack app. Questions will most commonly be about programs/projects which Thinktiv has done for other client companies. Sometimes a I may ask a question assuming you know the the related company or project simply because the question is asked within a channel dedicated to that company or project. If I ask a question which lacks necessary specifics like references to 'this program', 'this company', 'this client', or 'this project', you should ask the me to specify the missing details before proceeding.")
    PREFIX = """You are a helpful AI chat assistant for employees at a company named Thinktiv.  Employees communicate with you within chat channels of the Slack app. Questions will most commonly be about programs/projects which Thinktiv has done for other client companies. Sometimes they may ask a question assuming you know the the related company or project simply because the question is asked within a channel dedicated to that company or project. If your are ask a question which lacks necessary specifics like references to 'this program', 'this company', 'this client', or 'this project', you should ask the employee to specify the missing details before proceeding.\n
    Answer the following questions as best you can. You have access to the following tools:"""

     # Initialize agent
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=is_agent_verbose, 
        max_iterations=max_iterations, 
        return_intermediate_steps=return_thought_process, 
        memory=memory,
        agent_kwargs={'system_message_prefix':PREFIX},
        handle_parsing_errors=True)

    return agent
    

def handle_multi_step_query(agent, query: str, messages_history: List):
    params = {'input': query, 'chat_history': messages_history}
    result = agent.run(params)

    logging.info('QUERY results: %s', result)

    # prompt = hub.pull("hwchase17/react-chat")
    # prompt = prompt.partial(
    #     tools=render_text_description(tools),
    #     tool_names=", ".join([t.name for t in tools]),
    # )
    # return db_query_results.replace("Final answer here: ", "")

    return result
