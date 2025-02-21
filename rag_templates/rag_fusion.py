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
retriever = vectorstore.as_retriever() # no arg k -> no limit in number of queries


########################################################################################
# Part 3: PROMPTING  

# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


# Define a pipeline to generate multiple query variations for RAG Fusion
generate_queries = (
    prompt_rag_fusion  # Initial prompt for query generation
    | ChatOpenAI(temperature=0)  # Use OpenAI’s chat model with zero temperature (deterministic output)
    | StrOutputParser()  # Convert the LLM's response into a plain string format
    | (lambda x: x.split("\n"))  # Split the string output into a list of queries based on newline characters
)

########################################################################################
# Part 4: Retrieval Chain RAG-Fusion

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):

            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)

            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]

            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

# Define a pipeline to retrieve documents using RAG Fusion and Reciprocal Rank Fusion 
retrieval_chain_rag_fusion = (
    generate_queries |  # multi-query RAG-Fusion generator
    retriever.map() | # vector store retriever object (mapped to multiple queries)
    reciprocal_rank_fusion) # RRF to fuse and rerank the retrieved documents

# Example question for RAG Fusion
question = "What is task decomposition for LLM agents?"

# Retrieve documents using RAG Fusion and Reciprocal Rank Fusion with example question 
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)

########################################################################################
# Part 5: FINAL RAG  

# Define the RAG prompt template
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

# Create a ChatPromptTemplate using the predefined template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the OpenAI model (GPT-3.5/GPT-4) with temperature=0 (deterministic response)
llm = ChatOpenAI(temperature=0)

# Define the final RAG pipeline as a chain of operations
final_rag_chain = (
    {
        "context": retrieval_chain_rag_fusion,  # Retrieve relevant documents using RAG Fusion
        "question": itemgetter("question")  # Extract the "question" field from input data
    } 
    | prompt  # Format the retrieved context and question into the prompt template
    | llm  # Pass the prompt to the language model for response generation
    | StrOutputParser()  # Parse the model's output into a string
)

# Invoke the final RAG pipeline with a user question
final_rag_chain.invoke({"question": question})
