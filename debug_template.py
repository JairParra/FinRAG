from langchain_core.prompts import ChatPromptTemplate

def test_rag_template():
    print("Testing RAG Template...")
    template = """You are a professional financial analyst AI assistant. Answer the user's question based on the provided financial documents.

**Conversation History:**
{conversation_history}

**Retrieved Financial Information:**
{context}

**Current Question:** {question}

**Instructions:**
- **LATEX FORMATTING**: Use `$$` for block equations and `$` for inline equations. DO NOT use `\\[ ... \\]` or `\\( ... \\)`.
- **COMPLEX QUESTIONS**: If the user asks for calculations (e.g., EBITDA), show your work step-by-step using the data found.
- **CITATIONS**: Cite specific metrics with their source and page number in the text (e.g., "Revenue was $10M (Annual Report 2021, p. 5)").
- **ACCURACY**: If multiple documents are relevant, synthesize information comprehensively. If information is missing, state it clearly.
- **FORMATTING**: Use professional financial terminology and format numbers clearly (e.g., $1.2M, 15.3%).

**Answer:**"""
    
    try:
        prompt = ChatPromptTemplate.from_template(template)
        formatted = prompt.format(conversation_history="history", context="context", question="question")
        print("RAG Template: SUCCESS")
    except Exception as e:
        print(f"RAG Template: FAILED - {e}")

def test_viz_template():
    print("\nTesting Viz Template...")
    # Copying the system message from auto_viz.py
    system_msg = """You are a financial data extraction expert. Analyze the given text and extract ALL structured financial data.

Your task:
1. Identify ALL financial metrics with their values (including years, percentages, currencies)
2. Organize data into logical datasets (e.g., revenue over time, metrics by category)
3. Determine the best chart type for each dataset
4. Return structured JSON

    {{
      "name": "Revenue Over Time",
      "description": "Annual revenue from 2020-2023",
      "data_type": "time_series",  // or "comparison", "distribution", "relationship"
      "columns": ["Year", "Revenue (₹ millions)"],
      "rows": [
        {{"Year": "2020", "Revenue (₹ millions)": 62621}},
        {{"Year": "2021", "Revenue (₹ millions)": 71777}}
      ],
      "suggested_chart": "line",  // "bar", "pie", "grouped_bar", "line"
      "chart_rationale": "Time series data best shown with line chart for trend"
    }}
  ]
}}"""
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "Original Question: {question}\n\nText Response:\n{text}")
        ])
        formatted = prompt.format(question="q", text="t")
        print("Viz Template: SUCCESS")
    except Exception as e:
        print(f"Viz Template: FAILED - {e}")

if __name__ == "__main__":
    test_rag_template()
    test_viz_template()
