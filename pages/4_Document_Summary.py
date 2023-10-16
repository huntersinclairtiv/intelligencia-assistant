import os, tempfile
from typing import List
import streamlit as st
import logging
import langchain
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory.buffer import ConversationBufferMemory
from supabase.client import Client, create_client, Timeout
from supabase.lib.client_options import ClientOptions
from langchain.prompts import PromptTemplate
from streamlit_chat import message

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
model_name = os.environ.get("MODEL_NAME")

st.set_page_config(
    page_title="Thinktiv Chat - Demo",
    page_icon=":robot:"
)

client_options = ClientOptions(postgrest_client_timeout=Timeout(None))
supabase: Client = create_client(supabase_url, supabase_key, options=client_options) # type: ignore
logging.info('supabase: %s', supabase)


@st.cache_resource
def init_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return memory


#TESTING
# test_content = ["This is a test.", "This is line 2.", "This is line 3 with a new line.\nThis is line 3 with a new ~~ line2.\n", "This is line ~~ 4 with double new line\n\nThis is line 4 with double  ~~ new line2\n\nThis is line 4 ~~ with double new line3\n\nThis is line 4 with single new line4\nThis is line 4 with double new line5\n\n", "This is line 5 with single \n", "This is no return line."]
# test_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     separators=['~~','==============================\n\n','\n\n','\n',' '],
#     chunk_size = 20,
#     chunk_overlap  = 5,
#     encoding_name='cl100k_base',
# )
# test_texts = test_text_splitter.create_documents(test_content)
# logging.info('Docs: %s', test_texts)



# Streamlit app
st.subheader('Document Summary')
source_doc = st.file_uploader("Upload Source Document", type="pdf")
st.subheader('Document URL')
doc_url = st.text_input("Enter Document URL")
st.subheader('Query Document')
search_query = st.text_input("Enter Search Query")

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not source_doc:
        st.error("Please provide the source document.")
    else:
        #try:
        with st.spinner('Please wait...'):
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)
            for page in pages:
                page.metadata["source"] = source_doc.name
                if doc_url :
                    page.metadata["source_url"] = doc_url
            logging.info('llm pages: %s', pages)
            
            # Create embeddings for the pages and insert into Chroma database
            embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
            # vectordb = Chroma.from_documents(pages, embeddings)
            # vectordb = SupabaseVectorStore.from_documents(pages, embeddings, client=supabase)
            vectordb = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name= "documents")
            #  FOR SPECIFYING IF DIFFERENT table and db query :  table_name="documents", query_name="match_documents",
            chunk_size = 50 #maybe increase this - default is 500 for JS
            id_list: List[str] = []
            for i in range(0, len(pages), chunk_size):
                chunk = pages[i : i + chunk_size]
                result = vectordb.add_documents(chunk)  # type: ignore

                if len(result) == 0:
                    raise Exception("Error inserting: No rows added")

                id_list.extend(result)
            logging.info('id_list: %s', id_list)

            # Initialize the OpenAI module, load and run the summarize chain
            llm=ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model=model_name)
            chain = load_summarize_chain(llm, chain_type="stuff")
            search = vectordb.similarity_search(query="What is the Summary?", k=5, filter={"source":source_doc.name})
            summary = chain.run(input_documents=search, question="Write a summary within 200 words.")

            st.success(summary)

        #except Exception as e:
        #    st.exception(f"An error occurred: {e}")

if st.button("Ask Only"):
    # Validate inputs
    if not openai_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not search_query:
        st.error("Please provide a question.")
    else:
        #try:
        with st.spinner('Please wait...'):
            # Create embeddings for the pages and insert into Chroma database
            embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

            vectordb_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name= "documents")
            #  FOR SPECIFYING IF DIFFERENT table and db query :  table_name="documents", query_name="match_documents",

            # set the retirever to use similarity and return 5 result chunks
            vectordb_retriever = vectordb_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 5}
            )
            # search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            # matched_docs = vectordb_retriever.
            # matched_docs = vectordb_store.similarity_search(search_query, 10)

            # Initialize the OpenAI module, load and run the summarize chain
            llm=ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model=model_name)

            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum and keep the answer as concise as possible. 
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
            #Summaries: {summaries}

            # Initialize the OpenAI module, load and run the chain to answer question
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb_retriever, chain_type="stuff", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)
            # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vectordb_retriever, chain_type="stuff", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            query_results = qa_chain({"query": search_query})
            # query_results = qa_chain({"question": search_query})
            
            st.success(query_results["result"])
            logging.info('llm results: %s', query_results)

        #except Exception as e:
        #    st.exception(f"An error occurred: {e}")

if st.button("Ask Only 2"):
    # Validate inputs
    if not openai_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not search_query:
        st.error("Please provide a question.")
    else:
        #try:
        with st.spinner('Please wait...'):
            # Create embeddings for the pages and insert into Chroma database
            embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

            vectordb_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name= "documents")
            #  FOR SPECIFYING IF DIFFERENT table and db query :  table_name="documents", query_name="match_documents",

            # set the retirever to use similarity and return 5 result chunks
            vectordb_retriever = vectordb_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 5}
            )
            # search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            # matched_docs = vectordb_retriever.
            # matched_docs = vectordb_store.similarity_search(search_query, 10)

            # Initialize the OpenAI module, load and run the summarize chain
            llm=ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model=model_name)

            template = """Use the following pieces of context and conversation history to answer the question at the end as truthfully as possible. 
            If questions are asked where there is no relevant context available, you should attempt to answer the question based on the information you have been trained on as a LLM, but if you don't know the answer, just respond with "I don't know", don't try to make up an answer. 
            Keep the answer as concise as possible. 
            Context: {context}

            
            {chat_history}
            Human: {question}
            Assistant:"""
            QA_CHAIN_PROMPT = prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"], template=template
            )

            memory = init_memory()

            # Create the custom chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=vectordb_retriever, memory=memory,
                get_chat_history=lambda h : h, return_source_documents=True,
                combine_docs_chain_kwargs={'prompt': prompt}, verbose=True)

            # Initialize the OpenAI module, load and run the chain to answer question
            # qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb_retriever, chain_type="stuff", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)
            # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vectordb_retriever, chain_type="stuff", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            # query_results = qa_chain({"query": search_query})
            query_results = chain({"question": search_query})
            
            logging.info('llm results: %s', query_results)
            # st.success(query_results["result"])
            st.success(query_results["answer"])

            for i in range(len(memory.chat_memory.messages)-1, -1, -1):
                logging.info('messages: %s', memory.chat_memory.messages[i].type) 
                #note this will error if it is identical to any other message - need to figure out a way to make ids different
                message(message=memory.chat_memory.messages[i].content, is_user=(memory.chat_memory.messages[i].type == 'human'))
        #except Exception as e:
        #    st.exception(f"An error occurred: {e}")
