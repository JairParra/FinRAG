"""
Visualization utilities for financial data extraction and display
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from streamlit_app.config.colors import CHART_COLORS
from backend.app.config import get_chat_llm


def extract_financial_tables(retriever, query: str) -> Optional[pd.DataFrame]:
    """
    Extract financial data and convert to table format using RAG.
    """
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    if not docs:
        return None
    
    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs[:3]])
    
    # Use LLM to extract structured data
    template = """You are a financial data extraction expert. Extract financial data from the text and format it as a CSV table.

**Financial Text:**
{context}

**User Query:** {query}

**Instructions:**
- Extract relevant financial metrics, numbers, and dates
- Format as CSV with headers
- Use consistent number formatting
- Include units (e.g., millions, billions)
- If no tabular data exists, respond with "NO_TABLE_DATA"

**CSV Output:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_chat_llm(model="gpt-6-luna", reasoning_effort="medium")
    
    response = llm.invoke(prompt.format(context=context, query=query))
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    if "NO_TABLE_DATA" in response_text:
        return None
    
    # Parse CSV
    try:
        from io import StringIO
        csv_data = StringIO(response_text)
        df = pd.read_csv(csv_data)
        return df
    except Exception as e:
        print(f"Error parsing table: {e}")
        return None


def create_financial_chart(
    df: pd.DataFrame,
    chart_type: str = "bar",
    title: str = "Financial Metrics"
) -> go.Figure:
    """
    Create interactive financial charts using Plotly.
    """
    if df is None or df.empty:
        return None
    
    # Determine x and y columns (first column as x, rest as y values)
    x_col = df.columns[0]
    y_cols = []
    
    # Filter out metadata columns and ensure numeric types
    for col in df.columns[1:]:
        if col.lower() in ['source', 'page', 'ref', 'citation', 'notes']:
            continue
        y_cols.append(col)
    
    if chart_type == "bar":
        fig = go.Figure()
        for i, col in enumerate(y_cols):
            fig.add_trace(go.Bar(
                x=df[x_col],
                y=df[col],
                name=col,
                marker_color=CHART_COLORS[i % len(CHART_COLORS)]
            ))
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            barmode='group',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    
    elif chart_type == "line":
        fig = go.Figure()
        for i, col in enumerate(y_cols):
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                name=col,
                mode='lines+markers',
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                marker=dict(size=8)
            ))
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    
    elif chart_type == "pie":
        # Use first numeric column for pie chart
        numeric_col = y_cols[0]
        fig = go.Figure(data=[go.Pie(
            labels=df[x_col],
            values=df[numeric_col],
            marker=dict(colors=CHART_COLORS)
        )])
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)'
        )
    
    elif chart_type == "area":
        fig = go.Figure()
        for i, col in enumerate(y_cols):
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                name=col,
                fill='tonexty' if i > 0 else 'tozeroy',
                mode='lines',
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2)
            ))
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    
    return fig


def generate_financial_summary_chart(retriever) -> Optional[go.Figure]:
    """
    Generate an automatic summary chart of key financial metrics.
    """
    query = "Extract key financial metrics including revenue, profit, expenses across all years"
    df = extract_financial_tables(retriever, query)
    
    if df is not None:
        return create_financial_chart(df, "bar", "Key Financial Metrics Overview")
    
    return None


def extract_metrics_summary(retriever) -> Dict[str, str]:
    """
    Extract key financial metrics for display as KPIs.
    """
    query = "What are the most recent revenue, profit, and key financial ratios?"
    docs = retriever.invoke(query)
    
    if not docs:
        return {}
    
    context = "\n\n".join([doc.page_content for doc in docs[:2]])
    
    template = """Extract the most recent key financial metrics from this text.

**Text:**
{context}

**Instructions:**
- Extract: Total Revenue, Net Profit, Profit Margin (%), ROE (%)
- Use most recent values
- Format numbers clearly
- Respond in this exact format:
Revenue: [value]
Profit: [value]
Margin: [value]%
ROE: [value]%

If a metric is not available, use "N/A"

**Metrics:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_chat_llm(model="gpt-6-luna", reasoning_effort="medium")
    
    response = llm.invoke(prompt.format(context=context))
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Parse the response
    metrics = {}
    for line in response_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metrics[key.strip()] = value.strip()
    
    return metrics
