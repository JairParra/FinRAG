########################################################################################
# Part 0: Setup & Imports
import os
import bs4
from operator import itemgetter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

# Import the vector store functionality from the new module
from .vector_dbs import build_vector_store

# Assume credentials (LANGCHAIN_API_KEY, OPENAI_API_KEY, etc.) are set as global env variables

########################################################################################
# Part 1: Utility Functions

def generate_queries(question: str) -> list:
    """
    Generate multiple query variations from a single input question using a prompt chain.
    """
    template = (
        "You are a helpful assistant that generates multiple search queries based on a single input query.\n"
        "Generate multiple search queries related to: {question}\n"
        "Output (4 queries):"
    )
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    
    # Chain: prompt -> ChatOpenAI -> StrOutputParser -> lambda to split output by newline
    chain = prompt_rag_fusion | ChatOpenAI(temperature=0) | StrOutputParser()
    queries_str = chain.invoke({"question": question})
    queries = queries_str.split("\n")
    return queries

def reciprocal_rank_fusion(results: list, k: int = 60) -> list:
    """
    Fuse multiple lists of ranked documents using Reciprocal Rank Fusion (RRF).
    Each element in 'results' is assumed to be a list of documents.
    Returns a list of tuples: (document, fused_score).
    """
    fused_scores = {}
    # Iterate through each list of ranked documents
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)  # Serialize the document to use as a key
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    # Convert back to document objects and sort by fused score descending
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

########################################################################################
# Part 2: Retrieval and RAG Pipeline Functions

def retrieval_chain_rag_fusion(question: str, retriever: any) -> list:
    """
    For a given question, generate multiple query variants,
    retrieve documents for each query via the retriever,
    and then fuse the results using Reciprocal Rank Fusion.
    """
    queries = generate_queries(question)
    all_results = []
    for q in queries:
        # Retrieve documents relevant to each query (using .get_relevant_documents)
        results = retriever.get_relevant_documents(q)
        all_results.append(results)
    
    fused_results = reciprocal_rank_fusion(all_results)
    return fused_results

def final_rag_response(question: str, retriever: any) -> str:
    """
    Execute the final RAG pipeline:
      1. Retrieve and fuse documents using RAG-Fusion.
      2. Format the context and question into a prompt.
      3. Generate the final answer using an OpenAI chat model.
    """
    fused_results = retrieval_chain_rag_fusion(question, retriever)
    # Extract context from the fused documents (each document is expected to have a 'page_content' attribute)
    context = "\n".join([doc.page_content for doc, _ in fused_results if hasattr(doc, "page_content")])
    
    template = (
        "Answer the following question based on this context:\n\n"
        "{context}\n\n"
        "Question: {question}\n"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0)
    
    final_prompt = prompt.format(context=context, question=question)
    answer = llm(final_prompt)
    return answer
