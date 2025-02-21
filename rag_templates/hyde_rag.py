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
from langchain.prompts import ChatPromptTemplate  # Prompt template generator 
from langchain.prompts import FewShotChatMessagePromptTemplate  # Few-shot prompt template generator
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
retriever = vectorstore.as_retriever() # no arg k -> no limit in number of queries


########################################################################################
# Part 3: HyDE (Hypothethical Document Embedding) PROMPT 

# Define a prompt template for generating a hypothetical document (HyDE)
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

# Create a ChatPromptTemplate from the template
prompt_hyde = ChatPromptTemplate.from_template(template)


########################################################################################
# Part 4: HyDE DOC GENERATION  

# Define a chain to generate a hypothetical document based on the input question
generate_docs_for_retrieval = (
    prompt_hyde  # Format the question using the HyDE prompt
    | ChatOpenAI(temperature=0)  # Use OpenAI's model to generate a hypothetical document
    | StrOutputParser()  # Convert the model's output into a plain string
)

# Define the user question
question = "What is task decomposition for LLM agents?"

# Invoke the chain to generate a hypothetical document
print(generate_docs_for_retrieval.invoke({"question": question}))


########################################################################################
# Part 4: HyDE DOC RETRIEVAL

# Define a retrieval pipeline that uses the generated hypothetical document as the query
retrieval_chain = generate_docs_for_retrieval | retriever  

# Retrieve documents from the vector database using the generated passage
retrieved_docs = retrieval_chain.invoke({"question": question})

# Display the retrieved documents
print(retrieved_docs)


########################################################################################
# Part 5: FINAL HyDE RAG 

# Define a prompt template for the final RAG response
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

# Create a ChatPromptTemplate from the template
prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(temperature=0)

# Define the final RAG chain
final_rag_chain = (
    prompt  # Format the retrieved context and question into the template
    | llm  # Pass the formatted prompt to the language model for response generation
    | StrOutputParser()  # Convert the LLM's output into a plain string format
)

# Invoke the final RAG chain using the retrieved documents
final_rag_chain.invoke({"context": retrieved_docs, "question": question})

