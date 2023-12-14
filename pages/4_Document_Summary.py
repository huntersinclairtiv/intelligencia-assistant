import os, tempfile
import sys
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
from langchain.chains import LLMChain, SequentialChain, RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory.buffer import ConversationBufferMemory
from supabase.client import Client, create_client, Timeout
from supabase.lib.client_options import ClientOptions
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from streamlit_chat import message
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.document_loaders import UnstructuredAPIFileLoader
from langchain.document_loaders import GoogleDriveLoader




st.set_page_config(
    page_title="Thinktiv Chat - Demo",
    page_icon=":robot:"
)

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
unstructured_key = os.environ.get("UNSTRUCTURED_KEY")


client_options = ClientOptions(postgrest_client_timeout=Timeout(None))
supabase: Client = create_client(supabase_url, supabase_key, options=client_options) # type: ignore
logging.info('supabase: %s', supabase)


@st.cache_resource
def init_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    return memory

@st.cache_resource
def init_convo_memory():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return memory

@st.cache_resource
def init_convo_memory_main():
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='result')
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
st.subheader('Document Load')
source_doc2 = st.file_uploader("Upload Source Document")
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

# If the 'Summarize' button is clicked
if st.button("Summarize New"):
    # Validate inputs
    if not openai_api_key:
        st.error("Please provide the missing API keys in Settings.")
    elif not source_doc2:
        st.error("Please provide the source document2.")
    else:
        #try:
        with st.spinner('Please wait...'):
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            split_tup = os.path.splitext(source_doc2.name)
            # extract the file name and extension
            file_name = split_tup[0]
            # file_extension = split_tup[1]
            # with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir="temp") as tmp_file:
            #     tmp_file.write(source_doc2.read())

            # dirname, basename = os.path.split(tmp_file.name)
            # filename = "temp/" + basename
            # logging.info('FILE PATH: %s', filename)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir="temp")
            with open(tmp_file.name, 'w') as f:
                f.write(os.getenv("GOOGLE_AUTH_JSON"))

            dirname, basename = os.path.split(tmp_file.name)
            filename = "temp/" + basename
            logging.info('FILE PATH: %s', filename)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = filename

            loader2 = GoogleDriveLoader(
                folder_id="111l24JCZxaufVOaFjy0UAeVUA_lIPlpH",
                token_path="tokens/google_token.json",
                credentials_path=filename,
                # Optional: configure whether to recursively fetch files from subfolders. Defaults to False.
                recursive=False,
            )

            pages = loader2.load()
            logging.info('LOADER Key: %s',pages)

            # loader2 = UnstructuredAPIFileLoader(
            #     file_path=filename,
            #     api_key=unstructured_key,
            # )
            logging.info('LOADER Key: %s', loader2.api_key)

            # pages = loader2.load()
            logging.info('PARSED DOCS: %s', len(pages))
            logging.info('PARSED DOC: %s', pages[0])
            #Document(page_content='Lorem ipsum dolor sit amet.', metadata={'source': 'example_data/fake.docx'})

            #joined_content = '==============================\n\n'.join(content)

            # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            #     separators=['==============================\n\n','\n\n','\n',' '],
            #     chunk_size = 2500,
            #     chunk_overlap  = int(2500/30),
            #     encoding_name='cl100k_base',
            # )

            # pages = loader2.load_and_split(text_splitter=text_splitter)
            os.remove(tmp_file.name)


            
            # for page in pages:
            #     page.metadata["source"] = source_doc2.name
            #     if doc_url :
            #         page.metadata["source_url"] = doc_url
            # logging.info('LLM PAGES: %s', pages[:2])
            
            # # Create embeddings for the pages and insert into vector database
            # embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
            # # vectordb = SupabaseVectorStore.from_documents(pages, embeddings, client=supabase)
            # vectordb = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name= "documents")
            # #  FOR SPECIFYING IF DIFFERENT table and db query :  table_name="documents", query_name="match_documents",
            # chunk_size = 250 #maybe increase this - default is 500 for JS
            # id_list: List[str] = []
            # for i in range(0, len(pages), chunk_size):
            #     chunk = pages[i : i + chunk_size]
            #     result = vectordb.add_documents(chunk)  # type: ignore

            #     if len(result) == 0:
            #         raise Exception("Error inserting: No rows added")

            #     id_list.extend(result)
            # logging.info('id_list: %s', id_list)

            # # Initialize the OpenAI module, load and run the summarize chain
            # llm=ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model="gpt-4-1106-preview")
            # chain = load_summarize_chain(llm, chain_type="stuff")
            # summary = chain.run(input_documents=pages, question="Write a comprehensive summary.")

            # st.success(summary)

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
            If questions are asked where there is no relevant context available, you should attempt to answer the question based on the information you have been trained on as a LLM, but if you are not confident on the answer, just respond with "I don't know", don't try to make up an answer. 
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
if st.button("Ask Only 3"):
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
            # Initialize the OpenAI module, load and run the summarize chain
            llm=ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key, model=model_name)

            # set the retirever to use similarity and return 5 result chunks
            vectordb_retriever = vectordb_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}
            )
            # search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
            # matched_docs = vectordb_retriever.
            # matched_docs = vectordb_store.similarity_search(search_query, 10)


            metadata_field_info=[
                AttributeInfo(
                    name="source",
                    description="The source of the content to search. Can be Slack or a file name for example.",
                    type="string or list[string]",
                ),
                AttributeInfo(
                    name="channel",
                    description="Conversations within this Slack channel ID.  Useful for any questions referring to a specific channel id.",
                    type="string",
                ),
                AttributeInfo(
                    name="source_type",
                    description="The type of content from the source.  Can be Conversation, Document, or Deliverable",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number of the source document.",
                    type="integer"
                ),
                AttributeInfo(
                    name="source_url",
                    description="The URL of the original source document or conversation.",
                    type="string"
                ),
                AttributeInfo(
                    name="channel_name",
                    description="Conversations within this Slack channel name. Useful for any questions referring to a specific channel name.",
                    type="string",
                ),
                AttributeInfo(
                    name="reference_name_ucase",
                    description="An ALL capitalized version of the company name related to the content or conversation. Useful only when comparing to a 'company name' value that is all capital letters. Any comparision value should be converted to uppercase before comparing.",
                    type="string"
                ),
                AttributeInfo(
                    name="reference_name_cap",
                    description="A capitalized version of the company name related to the content or conversation. Useful only when comparing to a 'company name' value that has only the first letter capitalized.",
                    type="string",
                ),
                AttributeInfo(
                    name="reference_name_lcase",
                    description="An all lowercase version of the company name related to the content or conversation. Useful only when comparing to a 'company name' value that is all lowercase letters.  Any comparision value should be converted to lowercase before comparing. This can be the default company name comparison as long as you convert the value of the company name to all lowercase before doing the comparison filter.",
                    type="string",
                ),
            ]
            document_content_description = "Summary of a document or conversation."
            #vectordb_retriever = SelfQueryRetriever.from_llm(llm, vectordb_store, document_content_description, metadata_field_info, verbose=True)

            convo_template = """Use the following CONTEXT and conversation history to answer the question at the end to the best of your ability. You should think critically about the question and make inferences as needed from related context or conversation. 
            If the question asked has no relevant context available, you should attempt to answer the question based on the information you have been trained on as a LLM, but if you are not confident on the answer, just respond with "I don't know", don't try to make up an answer. 
            Keep the answer as concise as possible.\n 
            ------------\n
            CONTEXT: \n{context}\n
            ------------\n
            
            {chat_history}\n
            Human: {question}\n
            AI:"""
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "chat_history", "question"], template=convo_template
            )

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

            refine_template = (
                "I have more context below which can be used "
                "(only if needed) to update your previous answer.\n"
                "------------\n"
                "{context_str}\n"
                "------------\n"
                "Given the new context, update the previous answer to better "
                "answer my previous query if the new context is useful."
                "If the previous answer remains the same, repeat it verbatim. "
                "Never reference the new context or my previous query directly."
            )
            CHAT_REFINE_PROMPT = ChatPromptTemplate.from_messages(
                [("human", "{question}"), ("ai", "{existing_answer}"), ("human", refine_template)]
            )

            rephrase_template = (
                "Instruction: Given the context provided below, please reformulate the Human's query at the end. Use the Chat History if the query references prior information that should be included in the reformulated query. Consider the different ways team members might naturally discuss topics related to the query in a workplace setting. Think about direct and indirect references, synonyms, related concepts, and any colloquial or jargon terms they might use. The goal is to produce a revised query that captures the essence of the Human's query but is optimized for retrieving relevant conversations from a vector database of natural language discussions using similarity search.\n\n"
                "Context: The vector database contains Slack conversations among team members at our company. These conversations are diverse and cover various aspects of multiple projects. The language used is natural, conversational, and can range from casual to technical discussions, often sprinkled with workplace jargon.\n\n"
                "Chat History: \n{chat_history}\n\n"
                "Human: Original Query: ""{original_query}""\n\n"
                "AI: Revised Query: "
            )
            CHAT_REPHRASE_PROMPT = PromptTemplate(input_variables=["chat_history", "original_query"], template=rephrase_template)

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
            #rephrase_chain = LLMChain(llm=llm, prompt=CHAT_REPHRASE_PROMPT, output_key="question", memory=memory, verbose=True)



            # Create the custom chain
            #doc_retrieval_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb_retriever, input_key="question", chain_type="map_reduce", chain_type_kwargs={"question_prompt": QA_PROMPT, "combine_prompt": COMBINE_PROMPT}, return_source_documents=True, verbose=True)
            doc_retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=vectordb_retriever, memory=memory,
                get_chat_history=lambda h : h, return_source_documents=False,
                chain_type="map_reduce", condense_question_prompt=CONDENSE_QUESTION_PROMPT, rephrase_question=True,
                combine_docs_chain_kwargs={'question_prompt': QA_PROMPT, "combine_prompt": COMBINE_PROMPT}, verbose=True)
            
            #    combine_docs_chain_kwargs={'question_prompt': QA_CHAIN_PROMPT, 'refine_prompt': CHAT_REFINE_PROMPT}, verbose=True)
            #chain_type="refine" "stuff" "map_reduce"
            #'initial_response_name': '','document_variable_name': 'context', 
            # Initialize the OpenAI module, load and run the chain to answer question
            # qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb_retriever, chain_type="stuff", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)
            # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=vectordb_retriever, chain_type="stuff", chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
            # query_results = qa_chain({"query": search_query})

            # overall_chain = SequentialChain(
            #     chains=[rephrase_chain, doc_retrieval_chain],
            #     input_variables=["original_query"],
            #     # Here we return multiple variables
            #     output_variables=["question", "result"],
            #     verbose=True)

            
            #query_results = overall_chain({"original_query": search_query})
            #query_results = doc_retrieval_chain({"question": search_query})
            query_results = doc_retrieval_chain.run(search_query)
            
            logging.info('llm results: %s', query_results)
            #memory.chat_memory.add_ai_message(query_results["result"])

            st.success(query_results)
            #st.success(query_results["answer"])
            #st.success(query_results["result"])
if st.button("Quit App"):
    sys.exit("Exited the App")