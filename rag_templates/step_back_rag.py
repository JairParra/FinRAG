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
# Part 2: STEP-BACK PROMPT   

# Define few-shot examples to demonstrate step-back prompting
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",  # A specific question
        "output": "What can the members of The Police do?",  # A broader, step-back question
    },
    {
        "input": "Jan Sindel was born in what country?",  # Specific factual question
        "output": "What is Jan Sindel's personal history?",  # More general question
    },
]

# Define a prompt template for structuring example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),  # The original user question
        ("ai", "{output}"),  # The model's step-back reformulation
    ]
)

# Create a few-shot prompt template using the provided examples
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,  # Format the examples as conversational exchanges
    examples=examples,  # Use the defined step-back examples
)

# Define the main step-back prompting template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and 
            paraphrase a question to a more generic step-back question, which is easier to answer. 
            Here are a few examples:""",
        ),
        # Insert few-shot examples into the prompt
        few_shot_prompt,
        # User's actual question that needs to be rephrased
        ("user", "{question}"),
    ]
)


########################################################################################
# Part 3: STEP-BACK QUERY GENERATION 

# Generate step-back queries by passing the question through the step-back prompt
generate_queries_step_back = (prompt  # Use the previously defined step-back prompt template
                              | ChatOpenAI(temperature=0)  # Use an OpenAI LLM with deterministic output (temperature=0)
                              | StrOutputParser())  # Convert the model's response into a plain string format

# Define a user question
question = "What is task decomposition for LLM agents?"

# Invoke the step-back query generation process with the given question
generate_queries_step_back.invoke({"question": question})



########################################################################################
# Part 4: STEP-BACK RAG CHAIN 

# Define a response prompt template that incorporates both normal and step-back retrieval contexts
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. 
Your response should be comprehensive and not contradict the following context if relevant. 
Otherwise, ignore them if they are not relevant.

# {normal_context}  # Retrieved context based on the original question
# {step_back_context}  # Retrieved context based on the step-back question

# Original Question: {question}
# Answer:"""

# Create a ChatPromptTemplate from the response template
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# Define the full response generation chain
chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,

        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        
        # Pass on the original question
        "question": lambda x: x["question"],
    }
    | response_prompt  # Format all retrieved data into the final response prompt
    | ChatOpenAI(temperature=0)  # Generate an answer using OpenAI’s LLM
    | StrOutputParser()  # Convert the model’s response into a plain string format
)

# Invoke the chain with the given question
chain.invoke({"question": question})





