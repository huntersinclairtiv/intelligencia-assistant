import sys

from langchain.docstore.document import Document
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class ChromaDB:
    default_embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    default_namespace = 'chroma/documents'
    default_persist_dir = './chroma_db'
    cleanup_mode = 'incremental'
    source_id_key = 'source'

    def __init__(self, persist_dir=default_persist_dir, embeddings_function=default_embeddings, namespace=default_namespace) -> None:
        self.persist_dir = persist_dir
        self.embeddings = embeddings_function
        self.namespace = namespace
        self.vectorstore = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings_function)
        self.record_manager = SQLRecordManager(
            namespace, db_url="sqlite:///record_manager_cache.sql")
        self.record_manager.create_schema()

    def _clear(self, source_path):
        """
        This function will help to remove all the indexing which have `source_path` as their `source`
        # AS OF NOW THIS IS DONE BY ADDING AN EMPTY STRING AS THE DOCUMENT
        """
        indexing_result = index(
            [Document(page_content='', metadata={'source': source_path})],
            self.record_manager,
            self.vectorstore,
            cleanup='incremental',
            source_id_key=self.source_id_key
        )
        print('Summary for document roll back: ', indexing_result)

    def create_persistent_vector_database(self, docs):
        """
        Loads Documents in the incremental mode. This reduces duplicacy
        """
        indexing_result = index(
            docs,
            self.record_manager,
            self.vectorstore,
            cleanup=self.cleanup_mode,
            source_id_key=self.source_id_key,
        )
        print('Summary for document indexing: ', indexing_result)
        return self.vectorstore

    def get_persistent_vector_database(self):
        return self.vectorstore

    def delete_indexes_from_database(self, ids):
        try:
            self.get_persistent_vector_database()._collection.delete(ids=ids)
        except Exception as e:
            print('Failed Deleting the indexes')
