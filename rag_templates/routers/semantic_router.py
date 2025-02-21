########################################################################################
# Part 0: Setup

import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<langchain-api-key>'
os.environ['OPENAI_API_KEY'] = '<open-ai-api-key>'

########################################################################################
# Part 1: Imports 

# Data Retrieval and Processing
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import text splitter for chunking the document

# Vector Database 
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Chroma 

# Prompting 
from langchain_openai import ChatOpenAI # Interacts with OpenAI LLMs & generates embeddings
from langchain_core.prompts import PromptTemplate # Plain strinf formatting without chat history - Completion-based 
from langchain_core.prompts import ChatPromptTemplate # Used to generate prompts for the LLM - ChatBased models (multi-turn with roles)
from langchain_core.pydantic_v1 import BaseModel, Field # Pydantic model for type checking --> used to define router
from langchain_core.output_parsers import StrOutputParser # Output Parser 

# Chain   
from operator import itemgetter  # Extracts specific fields from a dictionary
from langchain_core.runnables import RunnablePassthrough  # Pass-through function for chaining
from langchain_core.runnables import RunnableLambda # Lambda function for chaining custom functions

# Langchain Tools 
from langchain import hub  # Fetches prebuilt LangChain prompts/chains
from langchain.load import dumps, loads # to convert from and to LangChain objects
from langchain.utils.math import cosine_similarity # Cosine similarity function


########################################################################################
# Part 2: ROUTER CREATION 

# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts 
embeddings = OpenAIEmbeddings()  # Load embeddings 
prompt_templates = [physics_template, math_template] # Possible prompts to choose from to adapt the question. 
prompt_embeddings = embeddings.embed_documents(prompt_templates) # Embed prompts into vectors 

# Route question to prompt 
def prompt_router(input):

    # Embed question
    query_embedding = embeddings.embed_query(input["query"])

    # Compute similarity between query and prompts 
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]

    # Chosen prompt + report 
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


########################################################################################
# Part 3: FINAL ROUTER CHAIN  

# Create the promnpt-router chain 
chain = (
    {"query": RunnablePassthrough()} # Input to the chain 
    | RunnableLambda(prompt_router) # Router function
    | ChatOpenAI() # OpenAI LLM 
    | StrOutputParser() # Output parser
)

print(chain.invoke("What's a black hole"))

