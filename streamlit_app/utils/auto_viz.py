"""
Automatic visualization generation from chat responses
Two-stage intelligent pipeline:
1. LLM extracts structured data from text
2. Smart chart generation based on data characteristics
"""
import re
import json
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from streamlit_app.config.colors import CHART_COLORS
from backend.app.utils.logging_config import log_info, log_error, log_debug, log_exception

def log_warning(message: str):
    """Simple warning logger"""
    log_info(f"WARNING: {message}")
from backend.app.config import OPENAI_API_KEY


########################################################################################
# Stage 1: LLM-Powered Data Extraction

def extract_structured_data_with_llm(text: str, original_question: str = "") -> Optional[Dict[str, Any]]:
    """
    Use LLM to intelligently extract structured data from text response.
    Returns a dictionary with extracted datasets and metadata.
    """
    try:
        log_debug("Starting LLM-powered data extraction...")
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fast and cost-effective for extraction
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data extraction expert. Analyze the given text and extract ALL structured financial data.

Your task:
1. Identify ALL financial metrics with their values (including years, percentages, currencies)
2. Look for MARKDOWN TABLES (with | separators) in the text
3. Extract data from tables, removing % signs and converting to numbers
4. Organize data into logical datasets
5. Determine the best chart type
6. Return structured JSON in this EXACT format:

{{
  "has_data": true,
  "datasets": [
    {{
      "name": "Revenue Growth Over Time",
      "description": "Year-on-year revenue growth rates from 2014-2023",
      "data_type": "time_series",
      "columns": ["Year", "Growth (%)"],
      "rows": [
        {{"Year": "2014", "Growth (%)": 13.0}},
        {{"Year": "2015", "Growth (%)": 21.0}}
      ],
      "suggested_chart": "line",
      "chart_rationale": "Time series data best shown with line chart"
    }}
  ]
}}

IMPORTANT:
- Convert percentages to numbers (12.8% → 12.8)
- Keep years as strings
- If you see a markdown table, extract ALL rows
- Always return valid JSON
"""),
            ("human", """Original Question: {question}

Text Response:
{text}

Extract all structured financial data from this response. Be thorough but accurate.""")
        ])
        
        chain = extraction_prompt | llm
        result = chain.invoke({"text": text, "question": original_question})
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            content = result.content
            
            # Find JSON block
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_str = content.strip()
            
            extracted_data = json.loads(json_str)
            
            if extracted_data.get("has_data", False) and extracted_data.get("datasets"):
                log_info(f"LLM extracted {len(extracted_data['datasets'])} datasets")
                return extracted_data
            else:
                log_debug("LLM found no structured data in response")
                return None
                
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse LLM JSON response: {e}")
            log_debug(f"LLM response: {result.content[:200]}")
            return None
            
    except Exception as e:
        log_exception("Error in LLM data extraction", e)
        return None


def extract_financial_data_from_text(text: str) -> Optional[pd.DataFrame]:
    """
    Fallback regex-based extraction for simple patterns.
    Used if LLM extraction fails.
    """
    try:
        # Pattern 1: Look for year: value patterns
        year_value_pattern = r'(\d{4}):\s*[₹$€£]?\s*([\d,\.]+)\s*([MmBbKkTt]?)'
        matches = re.findall(year_value_pattern, text)
        
        if matches and len(matches) >= 3:  # At least 3 data points
            data = []
            for year, value, unit in matches:
                # Clean value
                value_clean = float(value.replace(',', ''))
                # Handle units (M = million, B = billion, K = thousand)
                unit_upper = unit.upper()
                if unit_upper == 'M':
                    value_clean *= 1
                elif unit_upper == 'B':
                    value_clean *= 1000
                elif unit_upper == 'K':
                    value_clean *= 0.001
                
                data.append({'Year': int(year), 'Value': value_clean})
            
            if data:
                df = pd.DataFrame(data).sort_values('Year')
                log_debug(f"Regex extracted {len(df)} data points from text")
                return df
        
        # Pattern 2: Look for markdown tables
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if '|' in line and any(char.isdigit() for char in line):
                table_lines.append(line)
                in_table = True
            elif in_table and '|' not in line:
                break
        
        if len(table_lines) >= 2:  # At least header + 1 row
            # Try to parse markdown table
            try:
                from io import StringIO
                # Clean up table lines
                cleaned = '\n'.join([line.strip() for line in table_lines])
                # Remove separator lines
                cleaned = '\n'.join([line for line in cleaned.split('\n') 
                                    if not set(line.replace('|', '').strip()) <= {'-', ' '}])
                
                df = pd.read_csv(StringIO(cleaned), sep='|', skipinitialspace=True)
                df = df.dropna(axis=1, how='all')  # Remove empty columns
                df = df.iloc[:, 1:-1] if df.shape[1] > 2 else df  # Remove edge empty columns
                df.columns = df.columns.str.strip()
                
                # Clean data: remove % signs and convert to float where possible
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Try to convert strings like "12.8%" to float
                        try:
                            df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass
                
                if len(df) >= 3:
                    log_debug(f"Extracted table with {len(df)} rows from markdown")
                    return df
            except Exception as e:
                log_debug(f"Could not parse markdown table: {e}")
        
        return None
        
    except Exception as e:
        log_error(f"Error extracting financial data: {e}", exc_info=True)
        return None


def determine_visualization_type(df: pd.DataFrame, response_text: str) -> str:
    """
    Determine the best visualization type based on data and context.
    """
    response_lower = response_text.lower()
    
    # Check for time series indicators
    if any(col.lower() in ['year', 'date', 'quarter', 'month'] for col in df.columns):
        if any(keyword in response_lower for keyword in ['trend', 'over time', 'growth', 'change']):
            return 'line'
        return 'bar'
    
    # Check for comparison indicators
    if any(keyword in response_lower for keyword in ['compare', 'comparison', 'versus', 'vs']):
        return 'bar'
    
    # Check for distribution indicators
    if any(keyword in response_lower for keyword in ['distribution', 'breakdown', 'composition', 'share']):
        if len(df) <= 7:  # Pie chart works well for <= 7 categories
            return 'pie'
        return 'bar'
    
    # Default to bar chart
    return 'bar'


def create_chart_from_dataframe(
    df: pd.DataFrame,
    chart_type: str,
    title: str,
    metric_name: str = "Value"
) -> go.Figure:
    """Create a Plotly chart from dataframe"""
    try:
        x_col = df.columns[0]
        y_cols = [col for col in df.columns[1:] if df[col].dtype in ['int64', 'float64']]
        
        if not y_cols:
            # If no numeric columns found, try to convert
            for col in df.columns[1:]:
                # Skip metadata columns
                if col.lower() in ['source', 'page', 'ref', 'citation', 'notes']:
                    continue
                    
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))
                    y_cols.append(col)
                except:
                    pass
        
        if not y_cols:
            log_error("No numeric columns found for visualization")
            return None
        
        fig = go.Figure()
        
        if chart_type == 'line':
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
                yaxis_title=metric_name,
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                plot_bgcolor='rgba(26,51,80,0.5)',
                font=dict(color='#E3F2FD'),
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            # Enable interactivity
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        elif chart_type == 'bar':
            for i, col in enumerate(y_cols):
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                    text=df[col],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>%{y:,.0f}<extra></extra>'
                ))
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=metric_name,
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                plot_bgcolor='rgba(26,51,80,0.5)',
                font=dict(color='#E3F2FD'),
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
        
        elif chart_type == 'pie':
            y_col = y_cols[0]
            fig.add_trace(go.Pie(
                labels=df[x_col],
                values=df[y_col],
                marker=dict(colors=CHART_COLORS),
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percent: %{percent}<extra></extra>',
                pull=[0.05] * len(df)
            ))
            fig.update_layout(
                title=title,
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                font=dict(color='#E3F2FD'),
                height=500,
                showlegend=True
            )
        
        log_info(f"Created {chart_type} chart: {title}")
        return fig
        
    except Exception as e:
        log_error(f"Error creating chart: {e}", exc_info=True)
        return None


########################################################################################
# Stage 2: Intelligent Chart Generation

def create_chart_from_dataset(dataset: Dict[str, Any]) -> Optional[Tuple[go.Figure, str]]:
    """
    Create a chart from an extracted dataset dictionary.
    Uses LLM's suggested chart type and data structure.
    """
    try:
        # Convert dataset rows to DataFrame
        df = pd.DataFrame(dataset['rows'])
        
        if len(df) < 2:
            log_debug(f"Dataset '{dataset['name']}' has too few rows")
            return None
        
        chart_type = dataset.get('suggested_chart', 'bar')
        title = dataset.get('name', 'Financial Data')
        columns = dataset.get('columns', df.columns.tolist())
        
        log_debug(f"Creating {chart_type} chart for: {title}")
        
        # Ensure we have at least 2 columns for x and y
        if len(df.columns) < 2:
            log_debug("Not enough columns for visualization")
            return None
        
        x_col = df.columns[0]
        y_cols = df.columns[1:]
        
        fig = go.Figure()
        
        if chart_type == 'line':
            for y_col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines+markers',
                    name=str(y_col),
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            fig.update_layout(
                title=title,
                xaxis_title=str(x_col),
                yaxis_title="Value",
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                plot_bgcolor='rgba(26,51,80,0.5)',
                font=dict(color='#E3F2FD'),
                hovermode='x unified',
                showlegend=True,
                height=500
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            
        elif chart_type == 'bar':
            for idx, y_col in enumerate(y_cols):
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=str(y_col),
                    marker_color=CHART_COLORS[idx % len(CHART_COLORS)],
                    text=df[y_col],
                    texttemplate='%{text:,.0f}',
                    textposition='outside'
                ))
            fig.update_layout(
                title=title,
                xaxis_title=str(x_col),
                yaxis_title="Value",
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                plot_bgcolor='rgba(26,51,80,0.5)',
                font=dict(color='#E3F2FD'),
                barmode='group' if len(y_cols) > 1 else 'relative',
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            
        elif chart_type == 'grouped_bar':
            for idx, y_col in enumerate(y_cols):
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=str(y_col),
                    marker_color=CHART_COLORS[idx % len(CHART_COLORS)]
                ))
            fig.update_layout(
                title=title,
                xaxis_title=str(x_col),
                yaxis_title="Value",
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                plot_bgcolor='rgba(26,51,80,0.5)',
                font=dict(color='#E3F2FD'),
                barmode='group',
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            
        elif chart_type == 'pie':
            y_col = y_cols[0]
            fig.add_trace(go.Pie(
                labels=df[x_col],
                values=df[y_col],
                marker=dict(colors=CHART_COLORS),
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percent: %{percent}<extra></extra>',
                pull=[0.05] * len(df)
            ))
            fig.update_layout(
                title=title,
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.8)',
                font=dict(color='#E3F2FD'),
                height=500,
                showlegend=True
            )
        
        description = dataset.get('description', f"{chart_type.capitalize()} chart")
        log_info(f"Created {chart_type} chart: {title}")
        
        return (fig, f"📊 {description}")
        
    except Exception as e:
        log_exception(f"Error creating chart from dataset", e)
        return None


def generate_visualizations_from_response(
    response_text: str,
    max_charts: int = 3,
    original_question: str = ""
) -> Tuple[List[Tuple[go.Figure, str]], List[pd.DataFrame]]:
    """
    Two-stage intelligent visualization generation:
    1. Use LLM to extract structured data
    2. Create smart visualizations based on data characteristics
    
    Returns:
        Tuple of (visualizations, tables):
        - visualizations: List of tuples (figure, description)
        - tables: List of DataFrames used to create the visualizations
    """
    visualizations = []
    tables = []
    
    try:
        log_info("Starting intelligent visualization generation")
        
        # Check if this is a "propose visualization" request
        is_proposal_request = any(keyword in original_question.lower() 
                                  for keyword in ['propose', 'suggest', 'recommend', 'what visualizations'])
        
        if is_proposal_request:
            log_info("Detected visualization proposal request - will force generation")
        
        # Stage 1: LLM-powered data extraction
        extracted_data = extract_structured_data_with_llm(response_text, original_question)
        
        if extracted_data and extracted_data.get('has_data'):
            log_info(f"LLM extracted {len(extracted_data['datasets'])} datasets")
            
            # Stage 2: Create charts from extracted datasets
            for dataset in extracted_data['datasets'][:max_charts]:
                chart_tuple = create_chart_from_dataset(dataset)
                if chart_tuple:
                    visualizations.append(chart_tuple)
                    # Convert dataset to DataFrame for table display
                    df = pd.DataFrame(dataset['rows'])
                    tables.append(df)
            
            if visualizations:
                log_info(f"Generated {len(visualizations)} visualizations from LLM extraction")
                return visualizations, tables
        
        # If proposal request but no visualizations, we need to handle this differently
        if is_proposal_request and not visualizations:
            log_warning("Visualization proposal request but bot didn't provide data - returning empty")
            return [], []
        
        # Fallback: Try regex-based extraction if LLM fails
        log_debug("Falling back to regex-based extraction")
        df = extract_financial_data_from_text(response_text)
        
        if df is None or len(df) < 3:
            log_debug("Not enough data points for visualization")
            return [], []
        
        # Create basic visualization from regex extraction
        metric_name = "Value (in millions)"
        if 'ebitda' in response_text.lower():
            metric_name = "EBITDA (₹ millions)"
        elif 'revenue' in response_text.lower():
            metric_name = "Revenue (₹ millions)"
        elif 'profit' in response_text.lower():
            metric_name = "Profit (₹ millions)"
        
        chart_type = determine_visualization_type(df, response_text)
        title = f"{metric_name} Over Time"
        fig = create_chart_from_dataframe(df, chart_type, title, metric_name)
        
        if fig:
            description = f"{chart_type.capitalize()} chart showing {metric_name.lower()}"
            visualizations.append((fig, description))
            tables.append(df)
            log_info(f"Generated visualization using fallback method")
        
        return visualizations[:max_charts], tables[:max_charts]
        
    except Exception as e:
        log_exception("Error in visualization generation pipeline", e)
        return [], []
