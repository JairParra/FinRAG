########################################################################################
# Part 0: Imports
from langchain_openai import OpenAIEmbeddings

########################################################################################
# Part 1: Embeddings Utility

def get_openai_embeddings() -> OpenAIEmbeddings:
    """
    Returns an instance of the OpenAI embeddings using the pre-trained model.
    """
    return OpenAIEmbeddings()


