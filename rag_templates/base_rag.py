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

## 1. Load Documents 
loader = WebBaseLoader(  # Initialize a web document loader
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),  # URL of the webpage to load
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(  # Extract only specific parts of the webpage
            class_=("post-content", "post-title", "post-header")  # Filter by these HTML classes
        )
    ),
)

# Load the webpage content into documents
docs = loader.load()


## 2. Text Cleaning and Chunking 
# Split the documents into smaller text chunks
text_splitter = RecursiveCharacterTextSplitter(  # Initialize a text splitter
    chunk_size=1000,  # Each chunk will have 1000 characters
    chunk_overlap=200  # 200 characters of overlap between consecutive chunks
)

# Create text splits from the loaded documents
splits = text_splitter.split_documents(docs)


## 3. Vector Indexing	
# Initialize a vector database for storing document embeddings
vectorstore = Chroma.from_documents(  # Create a Chroma vector store from document splits
    documents=splits, 
    embedding=OpenAIEmbeddings()  # Use OpenAI's embedding model for vectorization
)

# Convert the vector store into a retriever for semantic search
retriever = vectorstore.as_retriever()

########################################################################################
# Part 3: RETRIEVAL AND GENERATION  

# 1. Prompt
# This is a simple prompt that asks the model to generate a response to a question.
prompt = hub.pull("rlm/rag-prompt")

# 2. LLM
# This is the OpenAI LLM that will generate the response to the question. 
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 3. Post-processing
# This function formats the retrieved documents into a single string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Chain
# Define the retrieval-augmented generation (RAG) pipeline
rag_chain = (
    {
        "context": retriever | format_docs,  # Retrieve relevant documents and format them as input context
        "question": RunnablePassthrough()    # Passes the user's query directly without modification
    }
    | prompt  # Apply the retrieved context and question to the RAG prompt template
    | llm     # Send the formatted prompt to the OpenAI model for response generation
    | StrOutputParser()  # Ensure the output is returned as a clean string
)

# Question
rag_chain.invoke("What is Task Decomposition?")

