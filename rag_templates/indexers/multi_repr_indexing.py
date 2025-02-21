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
from langchain_core.documents import Document # 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import text splitter for chunking the document

# Vector Database & In-Memory ByteStore
from langchain_openai import OpenAIEmbeddings  
from langchain.storage import InMemoryByteStore # In-memory storage for parent documents
from langchain_community.vectorstores import Chroma  # Vector store for storing embeddings 
from langchain.retrievers.multi_vector import MultiVectorRetriever  # Hierarchical retrieval mechanism

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

# Other tools 
import uuid # use uuid4() to generate a random UUID


########################################################################################
# Part 2: INDEXING 

# Retrieve two pages from the web and load them as documents
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())

########################################################################################
# Part 3: SUMMARY CHAIN 

 # Create a chain that summarizes the page content of each document
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)
    | StrOutputParser()
)

# Processes multiple documents in parallel, with a maximum of 5 concurrent LLM calls
summaries = chain.batch(docs, {"max_concurrency": 5})

########################################################################################
# Part 4: INDEXING 

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries",
                     embedding_function=OpenAIEmbeddings()) # document embeddings stored in Chroma

# The storage layer for the parent documents
store = InMemoryByteStore()  # Stores full documents in memory rather than a vector database
id_key = "doc_id"            # Each document is linked to its summary using a unique identifier

# Retriever object
## 1. vectorstore: VectorStore to retrieve summaries 
## 2. byte_store: ByteStore to retrieve full documents 
retriever = MultiVectorRetriever(
    vectorstore=vectorstore, # Retrieve summaries using vector search
    byte_store=store,        # Retrieve full documents from in-memory storage
    id_key=id_key,            # Unique ID key linking summaries to full documents
)
doc_ids = [str(uuid.uuid4()) for _ in docs] # Generarte unique IDs for each document

# Docs linked to summaries
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})  # Creates summary documents with their corresponding doc_id
    for i, s in enumerate(summaries)
]

# Add corresponding documents to the retriever vectorstore and docstore
retriever.vectorstore.add_documents(summary_docs) # Embeds the summaries and stores them in Chroma for efficient retrieval.
retriever.docstore.mset(list(zip(doc_ids, docs))) # Stores full documents in InMemoryByteStore with their unique doc_id.


########################################################################################
# Part 5: USAGE

# Example 1: Retrieve a summary of the first document 
query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query,k=1)
sub_docs[0]

# Example 2: Retrieve a summary of the second document 
retrieved_docs = retriever.get_relevant_documents(query,n_results=1)
retrieved_docs[0].page_content[0:500]

