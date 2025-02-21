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
# Part 2: DECOMPOSITION PROMPTING  

# (Query) Decomposition Prompt Template
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

# Create a ChatPromptTemplate from the template to be used for generating sub-questions
prompt_decomposition = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(temperature=0)

# Chain
generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

# Run
question = "What are the main components of an LLM-powered autonomous agent system?"
questions = generate_queries_decomposition.invoke({"question":question})


########################################################################################
# Part 3: RECURSIVE ANSWER RAG 

## 1. Prompt Template for Query Decomposition 

# Define a prompt template for query decomposition
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

# Create a ChatPromptTemplate from the defined template
decomposition_prompt = ChatPromptTemplate.from_template(template)


## 3. LLM
# Initialize the language model (LLM) with GPT-3.5-Turbo and deterministic responses (temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

## 4. Recursive Answer RAG Chain 

def format_qa_pair(question, answer):
    """Format a question-answer pair into a structured string"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"  # Concatenate Q&A into a formatted block
    return formatted_string.strip()  # Remove trailing whitespace


# Initialize an empty string to store previously generated question-answer pairs
q_a_pairs = ""

# Iterate over a list of questions, processing each one
for q in questions:
    
    # Define a RAG-based chain for query decomposition and response generation
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,  # Retrieve relevant context from retreiever vector dbs based on the question
            "question": itemgetter("question"),  # Extract the question field from input data
            "q_a_pairs": itemgetter("q_a_pairs")  # Extract previous Q&A pairs to provide additional context
        } 
        | decomposition_prompt  # Format the extracted data into the structured prompt
        | llm  # Pass the formatted prompt to the language model for response generation
        | StrOutputParser()  # Convert the LLM output into a plain string format
    )

    # Invoke the chain with the current question and accumulated Q&A pairs
    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})

    # Format the generated answer as a Q&A pair
    q_a_pair = format_qa_pair(q, answer)

    # Append the new Q&A pair to the existing list, separated by "---"
    q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

# check the retrieved questions 
print(q_a_pairs)

########################################################################################
# Part 4: FINAL ANSWER AGGREGATION RAG 

def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

# Format the retrieved question-answer pairs into a structured string 
context = format_qa_pairs(questions, answers)

# Prompt
template = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

# Create a ChatPromptTemplate from the provided template
prompt = ChatPromptTemplate.from_template(template)

# Define the final RAG chain for generating responses
final_rag_chain = (
    prompt  # Format the input data using the defined prompt template
    | llm  # Pass the formatted prompt to the language model for response generation
    | StrOutputParser()  # Convert the LLM's output into a plain string format
)

# Invoke the RAG chain with the provided context and question
final_rag_chain.invoke({"context": context, "question": question})