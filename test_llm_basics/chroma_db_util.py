import sys
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
company_handbook_chroma_path = './chroma_db'
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def create_persistent_vector_database(docs, embeddings_function=embeddings, persist_dir=company_handbook_chroma_path):
    return Chroma.from_documents(docs, embeddings_function, persist_directory=persist_dir)


def get_persistent_vector_database(embeddings_function=embeddings, persist_dir=company_handbook_chroma_path):
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings_function)
