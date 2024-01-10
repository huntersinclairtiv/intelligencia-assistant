import tensorflow as tf
import numpy as nf
import tensorflow_hub as hub
from langchain_core.embeddings import Embeddings


class CustomFaissEmbeddings(Embeddings):
    """
    class to utilise UniversalSentenceEncoder for embeddings
    """
    MODEL_URL = "./models/UniversalSentenceEncoder"

    def load_model(self):
        model = hub.load(self.MODEL_URL)
        return model

    def embed_query(self, input):
        model = self.load_model()
        return model([input])[0]

    def embed_documents(self, input):
        model = self.load_model()
        return model(input)


class CustomSupabaseEmbeddings(Embeddings):
    """
    class to utilise UniversalSentenceEncoder for embeddings
    """
    MODEL_URL = "./models/UniversalSentenceEncoder"

    def load_model(self):
        model = hub.load(self.MODEL_URL)
        return model

    def embed_query(self, input):
        model = self.load_model()
        return model([input])[0].numpy().tolist()

    def embed_documents(self, input):
        model = self.load_model()
        embedding_lists = [tensor.numpy().tolist() for tensor in model(input)]
        return embedding_lists
