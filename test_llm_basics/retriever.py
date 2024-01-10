import os

from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

import supabase_docstore as custom_docstore

from dotenv import load_dotenv
load_dotenv()


llm_model_type = "gpt-4-1106-preview"
llm = ChatOpenAI(max_retries=3, model=llm_model_type)
documents_table = os.environ.get('SUPABASE_DOCUMENTS_TABLE', '')
embeddings_table = os.environ.get('SUPABASE_EMBEDDINGS_TABLE', '')
match_function = os.environ.get('SUPABASE_QUERY_NAME', 'match_documents')


def get_supabase_vectorstore(table_name, match_function):
    """
    Returns a supabase vector store instance 
    """
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)
    return SupabaseVectorStore(
        embedding=OpenAIEmbeddings(),
        client=supabase,
        table_name=table_name,
        query_name=match_function
    )


def process_queries(query_list):
    """
    Accepts a list of queries and writes the responses generated, under the outputs directory
    """
    store = custom_docstore.SupabaseDocstore(documents_table)
    vectorstore = get_supabase_vectorstore(embeddings_table, match_function)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20),
        parent_splitter=RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200)
    )
    directory_path = 'outputs'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    for ques in query_list:
        qa = RetrievalQA.from_llm(
            llm=llm, retriever=retriever, return_source_documents=True)
        with get_openai_callback() as cb:
            result = qa({"query": ques})
            with open(f'{directory_path}/{ques}.txt', 'w') as f:
                f.write(result['result'])
            print("answered ", ques)


if __name__ == "__main__":

    query_list = [
        "What was the reason for decline of Sales?"
        # Add or remove questions from here and the corresponding output file will be generated under the 'outputs' folder
    ]

    process_queries(query_list)
