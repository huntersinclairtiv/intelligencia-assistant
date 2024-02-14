import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

import custom_embeddings

from dotenv import load_dotenv
load_dotenv()


class SupabaseVectorStore:

    def __init__(self, table_name, use_local_embedding=False) -> None:
        load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.embeddings = custom_embeddings.CustomSupabaseEmbeddings(
        ) if use_local_embedding else OpenAIEmbeddings()
        self.table_name = table_name or os.environ.get('SUPABASE_EMBEDDINGS_TABLE', '')

    def create_supabase_vectorstore(self, docs, ids=None):
        """
        Creates a vectorstore on supabase with <docs> using the <table_name> table 
        """
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)

        vectorstore = SupabaseVectorStore.from_documents(
            docs,
            self.embeddings,
            client=supabase,
            table_name=self.table_name, # Provide option to override here?
            ids=ids
        )
        return vectorstore

    def get_supabase_vectorstore(self):
        vectorstore = SupabaseVectorStore.from_documents(
            self.embeddings,
            client=self.supabase,
            table_name=self.table_name,
        )
        return vectorstore
    
    def delete_from_vectorstore(self, ids):
        self.get_supabase_vectorstore().delete(ids)
