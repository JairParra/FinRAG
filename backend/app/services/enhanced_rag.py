"""
Enhanced RAG service with intelligent retrieval, conversation memory,
and robust prompt engineering optimized for high-end reasoning models
(gpt-6-luna) while providing strict structural guidance for smaller models.
"""
import os
from typing import List, Tuple, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from backend.app.config import OPENAI_API_KEY, get_chat_llm

# Ensure API key is set
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""

########################################################################################
# Part 1: Intelligent Query Router

def should_use_rag(question: str, conversation_history: List[dict]) -> Tuple[bool, str]:
    """
    Determine if RAG retrieval is needed or if the question can be answered 
    from conversation history or is a general conversational query.
    
    Returns:
        Tuple[bool, str]: (should_use_rag, reasoning)
    """
    # Keywords that indicate document-specific queries
    document_keywords = [
        "report", "document", "pdf", "financial", "statement", "revenue", 
        "profit", "loss", "income", "balance", "cash flow", "asset", 
        "liability", "metric", "quarter", "year", "fiscal", "company",
        "risk", "growth", "investment", "dividend", "shareholder", "debt",
        "equity", "ratio", "margin", "expense", "cost", "sales", "ebitda",
        "guidance", "table", "data", "numbers", "statistics"
    ]
    
    # Keywords that indicate conversational/follow-up queries
    conversational_keywords = [
        "what do you think", "can you explain", "tell me more", 
        "what about", "how about", "summarize", "clarify",
        "you mentioned", "you said", "thank you", "thanks", "hello", "hi"
    ]
    
    question_lower = question.lower().strip()
    
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
    
    # Default to RAG for safety and comprehensiveness
    return True, "Default to RAG for comprehensive answer"


########################################################################################
# Part 2: Enhanced Query Generation

def generate_enhanced_queries(question: str, conversation_context: str = "") -> List[str]:
    """
    Generate multiple targeted query variations for better vector retrieval.
    Deconstructs complex queries into keyword-rich financial sub-queries.
    """
    template = (
        "You are an expert financial information retrieval assistant.\n"
        "Your task is to decompose the user's question into 3 diverse, precise search queries "
        "designed to retrieve relevant sections from corporate financial reports, balance sheets, "
        "income statements, and footnotes.\n\n"
        "{context}"
        "CRITICAL RULES:\n"
        "1. Output EXACTLY 3 search queries, one per line.\n"
        "2. Do NOT number the queries (no '1.', '2.', '3.').\n"
        "3. Do NOT include bullet points, hyphens, quotation marks, or preamble.\n"
        "4. Query 1: Focus on exact financial metrics, line items, and specific numbers.\n"
        "5. Query 2: Focus on contextual commentary, executive discussion, or trends.\n"
        "6. Query 3: Focus on broader fiscal year, comparison, or operational factors.\n\n"
        "User question: {question}\n\n"
        "Search queries:"
    )
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_chat_llm(model="gpt-6-luna", reasoning_effort="medium")
    chain = prompt | llm | StrOutputParser()
    
    try:
        queries_str = chain.invoke({
            "question": question,
            "context": f"Prior Conversation Context:\n{conversation_context}\n\n" if conversation_context else ""
        })
        
        # Clean up output lines
        raw_lines = [q.strip() for q in queries_str.split("\n") if q.strip()]
        clean_queries = []
        for line in raw_lines:
            # Strip numbering like '1.', '1)', '-', '*'
            cleaned = line.lstrip("0123456789.-*•) ").strip("\"' ")
            if cleaned:
                clean_queries.append(cleaned)
        
        # Always prioritize original question at position 0
        if question not in clean_queries:
            clean_queries.insert(0, question)
        
        return clean_queries[:3]
    except Exception as e:
        print(f"Error in generate_enhanced_queries: {e}")
        return [question]


def generate_suggested_prompts(context_text: str = "") -> List[str]:
    """
    Generate 3 short, relevant suggested questions based on uploaded document content.
    """
    template = (
        "You are a professional financial analyst.\n"
        "Review the excerpt from the financial document and suggest 3 insightful, concise questions "
        "that an investor or financial analyst would ask about this specific data.\n\n"
        "RULES:\n"
        "1. Output EXACTLY 3 questions, one per line.\n"
        "2. Keep each question concise (under 12 words).\n"
        "3. Do NOT include numbering, prefixes, or conversational filler.\n"
        "4. Focus on key numbers, revenue/profit trends, and financial risks mentioned in the text.\n\n"
        "Document Context:\n{context}\n\n"
        "Suggested Questions:"
    )
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_chat_llm(model="gpt-6-luna", reasoning_effort="medium")
    chain = prompt | llm | StrOutputParser()
    
    try:
        safe_context = context_text[:4000] if context_text else "Financial reports and statements."
        response = chain.invoke({"context": safe_context})
        raw_questions = [q.strip() for q in response.split("\n") if q.strip()]
        
        clean_questions = []
        for q in raw_questions:
            cleaned = q.lstrip("0123456789.-*•) ").strip("\"' ")
            if cleaned:
                clean_questions.append(cleaned)
                
        return clean_questions[:3] if len(clean_questions) >= 3 else [
            "What are the key financial highlights and revenue figures?",
            "How did profit margins and operating income perform year-over-year?",
            "What material risks and outlook are discussed in the report?"
        ]
    except Exception as e:
        print(f"Error generating suggested prompts: {e}")
        return [
            "What are the key financial highlights and revenue figures?",
            "How did profit margins and operating income perform year-over-year?",
            "What material risks and outlook are discussed in the report?"
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
) -> Tuple[str, List[dict]]:
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
        return "I could not retrieve relevant information from the uploaded documents. Please try rephrasing your question or uploading additional documents.", []
    
    fused_results = reciprocal_rank_fusion(all_results)
    
    # Extract context from top results
    context_docs = fused_results[:5]  # Top 5 documents for comprehensive synthesis
    context = "\n\n".join([
        f"[Source: {doc[0].metadata.get('source', 'Unknown')} - Page {doc[0].metadata.get('page', 'N/A')}]\n{doc[0].page_content}"
        for doc in context_docs
    ])
    
    # Check if this is a visualization proposal request
    is_viz_proposal = any(keyword in question.lower() 
                          for keyword in ['propose visualization', 'suggest visualization', 
                                         'recommend visualization', 'what visualizations',
                                         'propose useful', 'suggest graphs', 'recommend charts',
                                         'plot data', 'chart this'])
    
    if is_viz_proposal:
        template = """You are a senior financial analyst and data visualization architect.
The user is requesting concrete visualization proposals based on the provided financial documents.

**CRITICAL MANDATE:**
You MUST extract real, concrete numerical datasets from the context. Do NOT simply describe hypothetical charts—you MUST provide the exact data tables so the platform can automatically render interactive charts.

**Conversation History:**
{conversation_history}

**Retrieved Financial Context:**
{context}

**User Request:**
{question}

**REQUIRED OUTPUT STRUCTURE:**
For each proposed visualization (provide 2 to 3 distinct datasets):

### 1. Dataset Title & Objective
Briefly describe the business insight shown (e.g., "5-Year Revenue & Operating Profit Trajectory").

### 2. Clean Markdown Table
Provide a strict, cleanly formatted Markdown Table.
- Put metric units in column headers (e.g., `Revenue ($ Millions)`, `Operating Margin (%)`).
- Keep cell values purely numerical or standard percentages (e.g., `12450.5` or `18.2%`).
- Do NOT mix text sentences into numeric cells.

Example format:
| Year | Revenue ($M) | Operating Income ($M) | Operating Margin (%) |
|:-----|:-------------|:----------------------|:---------------------|
| 2021 | 52400        | 11200                 | 21.4%                |
| 2022 | 58900        | 13100                 | 22.2%                |
| 2023 | 64200        | 14800                 | 23.1%                |

### 3. Recommended Chart Type & Rationale
Specify: Best visual format (e.g., Multi-bar chart, Line chart, Waterfall chart) and key takeaway.

### 4. Source Citation
Explicitly cite the source document and page number.

**Answer:**"""
    else:
        template = """You are an elite financial analyst AI assistant specializing in corporate financial statements, earnings reports, and SEC filings.

Answer the user's question accurately, rigorously, and objectively based on the retrieved financial documents.

**Conversation History:**
{conversation_history}

**Retrieved Financial Information:**
{context}

**Current Question:**
{question}

---

### CORE OPERATING GUIDELINES (Follow strictly):

1. **GROUNDING & FAITHFULNESS**:
   - Rely strictly on the facts, metrics, and commentary present in the retrieved context.
   - NEVER invent or extrapolate unstated numbers. If a specific figure, quarter, or metric is missing, explicitly state: *"This metric is not specified in the provided document excerpts."*

2. **STRUCTURED PRESENTATION**:
   - **Executive Summary**: Begin with a direct, high-level summary (2-3 sentences) answering the core question.
   - **Key Metrics & Data Tables**: ALWAYS present numerical comparisons, financial statements, and trends in clean **Markdown Tables** with units in column headers (e.g., `| Fiscal Year | Net Sales ($M) | YoY Growth (%) |`).
   - **Financial Analysis & Commentary**: Provide analytical insights into the drivers behind the numbers (e.g., volume vs. price, margin expansion/contraction, foreign exchange impact).
   - **Risks & Caveats**: Highlight any relevant headwinds, caveats, or risks noted in the text.

3. **MATHEMATICAL & LATEX FORMATTING**:
   - Use `$$` for block mathematical equations and `$` for inline mathematical expressions.
   - Do NOT use `\\[ ... \\]` or `\\( ... \\)`.
   - When calculating variances, CAGR, or margins, show the formula and step-by-step substitution clearly.

4. **AUTOMATED VISUALIZATION COMPATIBILITY**:
   - The UI automatically extracts markdown tables to create interactive charts. Ensure your tables are clean, properly aligned, and use standard numeric formats.
   - Do NOT produce ASCII drawings or code blocks containing Python/Matplotlib/Excel scripts unless specifically requested.

5. **PRECISE CITATIONS**:
   - Cite source documents and page numbers for all key statistics (e.g., *[Annual Report 2023, p. 24]*).

**Answer:**"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_chat_llm(model="gpt-6-luna", reasoning_effort="medium")
    
    conv_history_text = conv_context if conv_context else "No previous conversation."
    
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
    conv_context = "\n".join([
        f"User: {msg['user']}\nAssistant: {msg['assistant']}"
        for msg in conversation_history[-5:]  # Last 5 exchanges
    ])
    
    template = """You are an expert financial analyst and advisory assistant.
Continue the conversation naturally, professionally, and helpfully.

**Conversation History:**
{conversation_history}

**Current Question:**
{question}

**Guidelines:**
- Answer based on the existing conversation flow and general financial domain knowledge.
- Be clear, structured, and insightful.
- If the question requires specific company data, figures, or verification that is not yet uploaded, politely invite the user to upload the relevant financial document or ask a document-specific query.
- Use professional financial terminology and maintain an objective, authoritative tone.

**Answer:**"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_chat_llm(model="gpt-6-luna", reasoning_effort="medium")
    
    formatted_prompt = prompt.format(
        conversation_history=conv_context if conv_context else "No previous conversation.",
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
