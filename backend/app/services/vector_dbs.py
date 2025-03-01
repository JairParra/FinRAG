########################################################################################
# Part 0: Imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

########################################################################################
# Part 1: Vector Store Functions

def build_vector_store(documents: list) -> any:
    """
    Create a vector store by embedding document chunks using OpenAI embeddings.
    Returns a retriever object.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    return vectorstore.as_retriever()

