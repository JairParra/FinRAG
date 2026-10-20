"""
Enhanced RAG service with intelligent retrieval and conversation memory
"""
import os
from typing import List, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from backend.app.config import OPENAI_API_KEY

# Ensure API key is set
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""

########################################################################################
# Part 1: Intelligent Query Router

def should_use_rag(question: str, conversation_history: List[dict]) -> Tuple[bool, str]:
    """
    Determine if RAG retrieval is needed or if the question can be answered 
    from conversation history or is a general query.
    
    Returns:
        Tuple[bool, str]: (should_use_rag, reasoning)
    """
    # Keywords that indicate document-specific queries
    document_keywords = [
        "report", "document", "pdf", "financial", "statement", "revenue", 
        "profit", "loss", "income", "balance", "cash flow", "asset", 
        "liability", "metric", "quarter", "year", "fiscal", "company",
        "risk", "growth", "investment", "dividend", "shareholder", "debt",
        "equity", "ratio", "margin", "expense", "cost", "sales"
    ]
    
    # Keywords that indicate conversational/follow-up queries
    conversational_keywords = [
        "what do you think", "can you explain", "tell me more", 
        "what about", "how about", "summarize", "clarify",
        "you mentioned", "you said"
    ]
    
    question_lower = question.lower()
    
    # Check for document-related keywords
    has_document_keywords = any(keyword in question_lower for keyword in document_keywords)
    
    # Check for conversational keywords
    has_conversational_keywords = any(keyword in question_lower for keyword in conversational_keywords)
    
    # Check if question is short (likely follow-up)
    is_short_question = len(question.split()) < 5
    
    # Check if there's recent context in conversation
    has_recent_context = len(conversation_history) > 0
    
    # Decision logic
    if has_document_keywords and not (is_short_question and has_recent_context):
        return True, "Document-specific query requiring RAG retrieval"
    
    if has_conversational_keywords and has_recent_context:
        return False, "Conversational follow-up using context"
    
    if is_short_question and has_recent_context:
        return False, "Short follow-up question using conversation history"
    
    # Default to RAG for ambiguous cases
    return True, "Default to RAG for comprehensive answer"


########################################################################################
# Part 2: Enhanced Query Generation

def generate_enhanced_queries(question: str, conversation_context: str = "") -> List[str]:
    """
    Generate multiple optimized query variations for better retrieval.
    Considers conversation context if available.
    """
    template = (
        "You are an expert at generating search queries for financial document retrieval.\n"
        "{context}"
        "Given the user's question, generate 3 diverse search queries that will help find relevant information.\n"
        "Make queries specific, use financial terminology, and cover different aspects.\n\n"
        "User question: {question}\n\n"
        "Generate 3 specific search queries (one per line):"
    )
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model="gpt-5.1")
    chain = prompt | llm | StrOutputParser()
    
    queries_str = chain.invoke({
        "question": question,
        "context": f"Previous context: {conversation_context}\n\n" if conversation_context else ""
    })
    queries = [q.strip() for q in queries_str.split("\n") if q.strip()]
    
    # Always include the original question
    if question not in queries:
        queries.insert(0, question)
    
    return queries[:3]  # Limit to 3 queries for faster performance


def generate_suggested_prompts(context_text: str = "") -> List[str]:
    """
    Generate 3-4 short, relevant suggested questions based on document content.
    """
    template = (
        "You are a helpful financial analyst assistant.\n"
        "Based on the following summary or content from financial documents, generate 3 short, specific, and interesting questions a user might ask.\n"
        "The questions should be diverse (e.g., one about metrics, one about trends, one about risks).\n"
        "Keep them under 10 words each.\n\n"
        "Document Context: {context}\n\n"
        "Generate 3 questions (one per line):"
    )
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.7, model="gpt-5.1")
    chain = prompt | llm | StrOutputParser()
    
    try:
        # If context is too long, truncate it
        safe_context = context_text[:4000] if context_text else "Financial reports and statements."
        
        response = chain.invoke({"context": safe_context})
        questions = [q.strip() for q in response.split("\n") if q.strip()]
        return questions[:3]
    except Exception as e:
        print(f"Error generating suggested prompts: {e}")
        return [
            "What are the key financial metrics highlighted in the 2023 report?",
            "How has the company's revenue trended over the past quarter?",
            "What risks are mentioned in the financial statements?"
        ]


########################################################################################
# Part 3: Reciprocal Rank Fusion

def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List[Tuple]:
    """
    Fuse multiple lists of ranked documents using Reciprocal Rank Fusion.
    """
    fused_scores = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results


########################################################################################
# Part 4: Enhanced RAG Response with Context

def generate_rag_response(
    question: str,
    retriever,
    conversation_history: List[dict] = None
) -> str:
    """
    Generate response using RAG with conversation history awareness.
    """
    # Build conversation context
    conv_context = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 exchanges
        conv_context = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in recent_history
        ])
    
    # Generate enhanced queries
    queries = generate_enhanced_queries(question, conv_context)
    
    # Retrieve documents for each query
    all_results = []
    for query in queries:
        try:
            results = retriever.invoke(query)
            all_results.append(results)
        except Exception as e:
            print(f"Error retrieving for query '{query}': {e}")
            continue
    
    # Fuse results
    if not all_results:
        return "I couldn't retrieve relevant information from the documents. Please try rephrasing your question."
    
    fused_results = reciprocal_rank_fusion(all_results)
    
    # Extract context from top results
    context_docs = fused_results[:4]  # Top 4 documents for faster processing
    context = "\n\n".join([
        f"[Source: {doc[0].metadata.get('source', 'Unknown')} - Page {doc[0].metadata.get('page', 'N/A')}]\n{doc[0].page_content}"
        for doc in context_docs
    ])
    
    # Check if this is a visualization proposal request
    is_viz_proposal = any(keyword in question.lower() 
                          for keyword in ['propose visualization', 'suggest visualization', 
                                        'recommend visualization', 'what visualizations',
                                        'propose useful', 'suggest graphs', 'recommend charts'])
    
    # Build enhanced prompt with conditional instructions
    if is_viz_proposal:
        template = """You are a professional financial analyst AI assistant. The user is asking you to propose visualizations based on the financial documents.

**IMPORTANT:** You must EXTRACT AND PROVIDE the actual data from the documents in a structured format. Do NOT just describe what charts to create - PROVIDE THE DATA!

**Conversation History:**
{conversation_history}

**Retrieved Financial Information:**
{context}

**Current Question:** {question}

**CRITICAL INSTRUCTIONS FOR VISUALIZATION PROPOSALS:**
1. EXTRACT all relevant numerical data from the documents
2. STRUCTURE the data in clear tables or lists with:
   - Column names (e.g., Year, Revenue, EBITDA, etc.)
   - Actual values with units (e.g., ₹71,777 million)
3. For EACH dataset, provide:
   - Dataset name and description
   - The actual data points in tabular format
   - Why this data would make a good visualization
4. Include AT LEAST 2-3 concrete datasets with real numbers
5. Format data like this:

**Dataset 1: Revenue Trend (2019-2023)**
Year | Revenue (₹ millions)
2019 | 62,000
2020 | 62,621
2021 | 71,777
2022 | 85,336
2023 | 98,450

- This shows clear growth trend
- Best visualized as: Line chart

DO NOT say "you could visualize" or "I recommend" - PROVIDE THE ACTUAL DATA so the system can create the visualization automatically!

**Answer:**"""
    else:
        template = """You are a professional financial analyst AI assistant. Answer the user's question based on the provided financial documents.

**Conversation History:**
{conversation_history}

**Retrieved Financial Information:**
{context}

**Current Question:** {question}

**Instructions:**
- **LATEX FORMATTING**: Use `$$` for block equations and `$` for inline equations. DO NOT use `\\[ ... \\]` or `\\( ... \\)`.
- **DATA PRESENTATION**: ALWAYS present numerical data in clear **Markdown Tables** with proper headers and alignment.
- **VISUALIZATIONS**: The system will AUTOMATICALLY generate interactive charts from your tables. DO NOT create ASCII art graphs or provide Python/Excel code.
- **TABLE FORMAT**: When presenting data over time or comparisons, use this format:
  ```
  Year | Metric
  2020 | 12.5
  2021 | 15.3
  ```
- **COMPLEX QUESTIONS**: If the user asks for calculations, show your work step-by-step using the data found.
- **CITATIONS**: Cite specific metrics with their source and page number (e.g., "Revenue was $10M (Annual Report 2021, p. 5)").
- **ACCURACY**: If multiple documents are relevant, synthesize information comprehensively. If information is missing, state it clearly.
- **FORMATTING**: Use professional financial terminology and format numbers clearly (e.g., $1.2M, 15.3%).
- **NO CODE**: Never provide code snippets (Python, R, Excel formulas) unless explicitly requested. Just provide the data in tables.

**Answer:**"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model="gpt-5.1")
    
    # Format conversation history
    conv_history_text = conv_context if conv_context else "No previous conversation."
    
    # Generate response
    formatted_prompt = prompt.format(
        conversation_history=conv_history_text,
        context=context,
        question=question
    )
    
    response = llm.invoke(formatted_prompt)
    
    # Extract source information from context_docs
    sources = []
    for doc, score in context_docs:
        source_info = {
            "source": doc.metadata.get('source', 'Unknown'),
            "page": doc.metadata.get('page', 'N/A'),
            "score": round(score, 3),
            "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        sources.append(source_info)
    
    response_text = response.content if hasattr(response, 'content') else str(response)
    return response_text, sources


########################################################################################
# Part 5: Conversational Response (without RAG)

def generate_conversational_response(
    question: str,
    conversation_history: List[dict]
) -> str:
    """
    Generate response using only conversation history (no document retrieval).
    """
    # Build conversation context
    conv_context = "\n".join([
        f"User: {msg['user']}\nAssistant: {msg['assistant']}"
        for msg in conversation_history[-5:]  # Last 5 exchanges
    ])
    
    template = """You are a helpful financial analyst AI assistant. Continue the conversation naturally.

**Conversation History:**
{conversation_history}

**Current Question:** {question}

**Instructions:**
- Answer based on the conversation context
- Be concise and relevant
- If you need document information to answer properly, politely suggest the user ask a document-specific question
- Maintain professional tone

**Answer:**"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0.3, model="gpt-5.1")
    
    formatted_prompt = prompt.format(
        conversation_history=conv_context,
        question=question
    )
    
    response = llm.invoke(formatted_prompt)
    
    return response.content if hasattr(response, 'content') else str(response)


########################################################################################
# Part 6: Main Chat Function

def chat_with_documents(
    question: str,
    retriever,
    conversation_history: List[dict] = None
) -> Tuple[str, bool, List[dict]]:
    """
    Main chat function that intelligently decides whether to use RAG or conversation context.
    
    Returns:
        Tuple[str, bool, List[dict]]: (response, used_rag, sources)
    """
    if conversation_history is None:
        conversation_history = []
    
    # Decide whether to use RAG
    use_rag, reasoning = should_use_rag(question, conversation_history)
    
    print(f"Query routing decision: {reasoning}")
    
    if use_rag and retriever is not None:
        response, sources = generate_rag_response(question, retriever, conversation_history)
        return response, True, sources
    else:
        response = generate_conversational_response(question, conversation_history)
        return response, False, []
