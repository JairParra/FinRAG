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
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import text splitter for chunking the document

# Vector Database 
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Chroma 

# Prompting 
from langchain.prompts import ChatPromptTemplate  # Multi Query: Different Perspectives
from langchain_core.output_parsers import StrOutputParser # Output Parser 
from langchain_openai import ChatOpenAI # Interacts with OpenAI LLMs & generates embeddings

# Chain   
from operator import itemgetter  # Extracts specific fields from a dictionary
from langchain_core.runnables import RunnablePassthrough  # Pass-through function for chaining

# Langchain Tools 
from langchain import hub  # Fetches prebuilt LangChain prompts/chains
from langchain.load import dumps, loads # to convert from and to LangChain objects


########################################################################################
# Part 2: INDEXING 

## 1.1: Data Retrieval and Preprocessing 

# Define a web loader to fetch the blog post content from the given URL
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(  # Extract only specific sections from the HTML
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

# Load the blog content as a document
blog_docs = loader.load()

# Initialize a text splitter with a chunk size of 300 tokens and an overlap of 50 tokens
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)

# Split the blog content into smaller document chunks
splits = text_splitter.split_documents(blog_docs)


## 1.2: Vector Database 

# Create a vector store by embedding the document chunks using OpenAI embeddings
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

# Convert the vector store into a retriever object for semantic search
retriever = vectorstore.as_retriever() # no kwargs={'k':1} -> no limit in number of queries


########################################################################################
# Part 3: PROMPTING  

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)


# Define the RAG chain pipeline to generate queries
generate_queries = (
    prompt_perspectives  # The prompt for rephrasing queries
    | ChatOpenAI(temperature=0)  # Uses OpenAI's model (GPT) to generate query variations
    | StrOutputParser()  # Converts the output into a clean string
    | (lambda x: x.split("\n"))  # Splits the generated text into a list of five queries
)


def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is task decomposition for LLM agents?"

# generate queries -> retrieve for each generated query & return list of generated docs -> and get unique union
retrieval_chain = (generate_queries | # multi-query generator chain 
                   retriever.map() | # retrieve embeddings for each generated query 
                   get_unique_union) # filter unique union of retrieved documents
docs = retrieval_chain.invoke({"question":question})



########################################################################################
# Part 4: FINAL RAG  

# Define a prompt template for answering questions using retrieved context
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

# Create a LangChain chat prompt template using the above template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the OpenAI model (GPT-3.5/GPT-4) with temperature=0 (deterministic response)
llm = ChatOpenAI(temperature=0)

# Create a processing pipeline for answering questions using retrieved documents
final_rag_chain = (
    {
        "context": retrieval_chain,  # Uses the retrieval pipeline to get relevant documents
        "question": itemgetter("question")  # Extracts the "question" key from input dictionary
    } 
    | prompt  # Formats the retrieved context and question into a structured prompt
    | llm  # Passes the formatted prompt to OpenAI's LLM to generate an answer
    | StrOutputParser()  # Ensures that the LLM output is returned as a clean string
)

# Execute the chain with a user-provided question (can be multiple?)
response = final_rag_chain.invoke({"question": question})

# Print the final response generated by the LLM
print(response)
