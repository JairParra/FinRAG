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
from langchain_core.prompts import ChatPromptTemplate # Used to generate prompts for the LLM 
from langchain_core.pydantic_v1 import BaseModel, Field # Pydantic model for type checking --> used to define router
from langchain_core.output_parsers import StrOutputParser # Output Parser 
from langchain_openai import ChatOpenAI # Interacts with OpenAI LLMs & generates embeddings

# Chain   
from operator import itemgetter  # Extracts specific fields from a dictionary
from langchain_core.runnables import RunnablePassthrough  # Pass-through function for chaining
from langchain_core.runnables import RunnableLambda # Lambda function for chaining custom functions

# Langchain Tools 
from langchain import hub  # Fetches prebuilt LangChain prompts/chains
from langchain.load import dumps, loads # to convert from and to LangChain objects


########################################################################################
# Part 2: ROUTER CREATION 

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)  
structured_llm = llm.with_structured_output(RouteQuery) # Ensure output is structured as per RouteQuery model

# Prompt 
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""


# Create a prompt template with system instructions and a placeholder for the user's question
routing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define query router 
router = routing_prompt | structured_llm


########################################################################################
# Part 3: ROUTER INVOCATION

# Example question with Python snippet, to be send to the router 
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

# Send quetion to router, which will associate the correct datasource with the question
result = router.invoke({"question": question})


########################################################################################
# Part 4: FINAL ROUTER CHAIN 

# function to assign the correct datasource based on the result from the router
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here 
        return "chain for js_docs"
    else:
        ### Logic here 
        return "golang_docs"

# Chain for the router 
full_chain = router | RunnableLambda(choose_route)

# Example invocation of the full chain 
full_chain.invoke({"question": question})

