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
from langchain_core.prompts import ChatPromptTemplate
from streamlit_app.config.colors import (
    CHART_COLORS, COLOR_REVENUE, COLOR_EBITDA, COLOR_PROFIT, COLOR_COST_DRAG, COLOR_BASELINE
)
from backend.app.utils.logging_config import log_info, log_error, log_debug, log_exception

def log_warning(message: str):
    """Simple warning logger"""
    log_info(f"WARNING: {message}")
from backend.app.config import OPENAI_API_KEY, get_chat_llm


########################################################################################
# Stage 1: LLM-Powered Data Extraction

def extract_structured_data_with_llm(text: str, original_question: str = "") -> Optional[Dict[str, Any]]:
    """
    Use LLM to intelligently extract structured data from text response.
    Returns a dictionary with extracted datasets and metadata.
    """
    try:
        log_debug("Starting LLM-powered data extraction...")
        
        llm = get_chat_llm(
            model="gpt-6-luna",
            reasoning_effort="medium",
            api_key=OPENAI_API_KEY
        )
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data extraction expert. Analyze the given text and extract ALL structured financial data.

Your task:
1. Identify ALL financial metrics with their values (including years, percentages, currencies)
2. Look for MARKDOWN TABLES (with | separators) and numerical comparisons in the text
3. Organize data into logical, well-scaled datasets
4. FINANCIAL DATA STRUCTURING RULES:
   - Scale & Unit Segregation: Do NOT combine metrics with vastly different units (e.g., INR millions vs USD millions) in the same dataset. If a table reports both INR and USD, create separate datasets for each currency (e.g., "INR Financial Results" and "USD Financial Results").
   - Growth & Variance: If a table contains base period quantities (e.g., 2021, 2022) and growth rates (e.g., Change %), extract both, but also create a dedicated "Growth Rates" dataset so percentages can be compared directly without being dwarfed by large base amounts.
   - Multi-dimensional data: If data has a category dimension (e.g. "Currency or basis": INR, USD, Constant currency), include it as a column alongside "Metric" and "Growth (%)".
5. INTELLIGENT CHART SELECTION RULES:
   - "scorecard": Use when reporting 1 key metric across 1 or 2 periods (e.g. 2022 vs 2023 revenue share, EBITDA margin change), or a single scalar KPI (e.g. 72% Americas revenue concentration), or a single row of 1-4 summary metrics.
   - "donut": Use when the data represents parts of a whole summing to ~100% (e.g. regional revenue breakdown, portfolio mix, customer concentration shares).
   - "horizontal_bar": Use for rankings, comparing scopes/entities (e.g. Group vs Company), or categories with long descriptive names.
   - "grouped_bar": Use for comparing multiple metrics or entities across 2 or more categories.
   - "line": STRICTLY ONLY when there are 3 or more consecutive time periods (e.g., 2021, 2022, 2023). NEVER suggest a line chart for 1 or 2 periods.
   - "growth_divergence": When contrasting top-line growth (e.g. Revenue) against bottom-line growth (e.g. EBITDA).
   - "waterfall": For variance walks, P&L bridges, or cost bridges.
6. Return structured JSON in this EXACT format:

{{
  "has_data": true,
  "datasets": [
    {{
      "name": "Americas Revenue Share",
      "description": "Americas geographic share of Group revenue between 2022 and 2023",
      "data_type": "comparison",
      "columns": ["Metric", "2022", "2023", "Change (pp)"],
      "rows": [
        {{"Metric": "Americas share of Group revenue (%)", "2022": 72.4, "2023": 71.65, "Change (pp)": -0.75}}
      ],
      "suggested_chart": "scorecard",
      "chart_rationale": "Single metric 2-period comparison with change is best presented as an executive KPI scorecard"
    }},
    {{
      "name": "Revenue and Adjusted EBITDA (INR)",
      "description": "Reported revenue and EBITDA for 2021-2022 in INR million",
      "data_type": "comparison",
      "columns": ["Metric", "2021", "2022", "Change (%)"],
      "rows": [
        {{"Metric": "Revenue", "2021": 71777, "2022": 91996, "Change (%)": 28.2}},
        {{"Metric": "Adjusted EBITDA", "2021": 13637, "2022": 14664, "Change (%)": 7.5}}
      ],
      "suggested_chart": "grouped_bar",
      "chart_rationale": "Period comparison with year-over-year growth"
    }},
    {{
      "name": "Revenue and EBITDA Growth Rates",
      "description": "Comparison of top-line vs bottom-line growth percentages",
      "data_type": "comparison",
      "columns": ["Metric", "Growth (%)"],
      "rows": [
        {{"Metric": "Revenue", "Growth (%)": 28.2}},
        {{"Metric": "Adjusted EBITDA", "Growth (%)": 7.5}}
      ],
      "suggested_chart": "growth_divergence",
      "chart_rationale": "Direct growth rate comparison showing margin divergence"
    }}
  ]
}}

IMPORTANT:
- Convert percentages to numbers (12.8% → 12.8)
- Strip comma separators from numbers (91,996 → 91996)
- Keep years and metric names as strings
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

def is_percentage_metric(col_name: str, values: pd.Series = None) -> bool:
    """Detect if column represents percentage / rate metrics."""
    name_lower = str(col_name).lower()
    if any(k in name_lower for k in ["%", "percent", "growth", "margin", "rate", "share", "ratio", "cagr", "yield", "change", "diff", "drag", "variance", "points"]):
        return True
    if values is not None:
        try:
            numeric_vals = pd.to_numeric(values.dropna())
            if len(numeric_vals) > 0 and numeric_vals.abs().max() <= 100:
                return True
        except Exception:
            pass
    return False


def is_year_or_period_col(col_name: str) -> bool:
    """Detect if column header represents a year or reporting period (e.g., 2021, FY22, Q3)."""
    name = str(col_name).strip()
    return bool(re.search(r'\b(19\d\d|20\d\d)\b', name) or re.search(r'\b(fy\d\d|q[1-4]|h[1-2])\b', name, re.IGNORECASE))


def clean_and_classify_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Intelligently classify DataFrame columns into:
    - x_col: entity / metric name column (typically column 0)
    - dimension_cols: categorical columns (e.g. 'Currency or basis', 'Segment', 'Region')
    - quantity_cols: numeric base amounts (e.g. 2021, 2022 in millions)
    - pct_cols: percentage change, margin, or growth columns (e.g. Change (%), Margin (%))
    """
    x_col = df.columns[0]
    dimension_cols = []
    quantity_cols = []
    pct_cols = []
    
    for col in df.columns[1:]:
        if str(col).lower() in ['source', 'page', 'ref', 'citation', 'notes']:
            continue
        cleaned = df[col].astype(str).str.replace(',', '').str.replace('%', '').str.strip()
        num_series = pd.to_numeric(cleaned, errors='coerce')
        valid_count = num_series.notna().sum()
        total_count = len(df[col].dropna())
        
        # If less than 50% can be converted to numbers, it's a categorical dimension column
        if total_count > 0 and (valid_count / total_count) < 0.5:
            dimension_cols.append(col)
        else:
            if is_percentage_metric(col, num_series):
                pct_cols.append(col)
            else:
                quantity_cols.append(col)
                
    return {
        "x_col": x_col,
        "dimension_cols": dimension_cols,
        "quantity_cols": quantity_cols,
        "pct_cols": pct_cols
    }


def create_financial_scorecard(
    df: pd.DataFrame,
    title: str,
    dataset_desc: str = ""
) -> Optional[Tuple[go.Figure, str]]:
    """
    Renders executive KPI scorecards / bullet gauges for single-row metrics,
    2-period comparisons (e.g. 2022 vs 2023), and single scalars.
    Eliminates awkward 1-point or 2-point disconnected line charts.
    """
    if len(df) != 1:
        return None
        
    cols = list(df.columns)
    metric_label = str(df.iloc[0, 0])
    
    numeric_candidates = cols[1:] if len(cols) > 1 else cols
    period_cols = [c for c in numeric_candidates if is_year_or_period_col(c) and not any(k in str(c).lower() for k in ['change', 'diff', 'drag', 'variance', 'points'])]
    change_cols = [c for c in numeric_candidates if any(k in str(c).lower() for k in ['change', 'diff', 'drag', 'variance', 'points'])]
    
    # Case 1: Two periods with comparison / delta (e.g. 2022: 72.4%, 2023: 71.65%)
    if len(period_cols) >= 2:
        p_prior = period_cols[0]
        p_curr = period_cols[-1]
        
        curr_val = pd.to_numeric(str(df.iloc[0][p_curr]).replace(',', '').replace('%', ''), errors='coerce')
        prior_val = pd.to_numeric(str(df.iloc[0][p_prior]).replace(',', '').replace('%', ''), errors='coerce')
        
        if pd.notna(curr_val) and pd.notna(prior_val):
            is_pct = is_percentage_metric(metric_label) or any(is_percentage_metric(c) for c in [p_curr, p_prior]) or (abs(curr_val) <= 100 and abs(prior_val) <= 100)
            sfx = "%" if is_pct else ""
            delta_sfx = " pp" if is_pct else sfx
            fmt = ".2f" if is_pct else ",.1f"
            
            fig = go.Figure()
            indicator_params = {
                "mode": "number+delta+gauge" if (is_pct and 0 <= curr_val <= 100) else "number+delta",
                "value": curr_val,
                "number": {
                    "suffix": sfx,
                    "valueformat": fmt,
                    "font": {"size": 42, "color": "#F8FAFC", "family": "Inter, Segoe UI, sans-serif"}
                },
                "delta": {
                    "reference": prior_val,
                    "valueformat": fmt,
                    "suffix": delta_sfx,
                    "increasing": {"color": "#10B981"},
                    "decreasing": {"color": "#EF4444"},
                    "font": {"size": 16}
                },
                "title": {
                    "text": f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>{metric_label} ({p_curr}) | Prior ({p_prior}): {prior_val:{fmt}}{sfx}</span>"
                }
            }
            
            if is_pct and 0 <= curr_val <= 100:
                indicator_params["gauge"] = {
                    "shape": "bullet",
                    "axis": {"range": [0, 100], "tickfont": {"color": "#94A3B8", "size": 11}},
                    "bar": {"color": "#38BDF8"},
                    "threshold": {
                        "line": {"color": "#F59E0B", "width": 3.5},
                        "thickness": 0.85,
                        "value": prior_val
                    },
                    "bgcolor": "rgba(255, 255, 255, 0.05)",
                    "bordercolor": "rgba(255, 255, 255, 0.15)"
                }
                
            fig.add_trace(go.Indicator(**indicator_params))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(10, 25, 41, 0.95)",
                plot_bgcolor="rgba(26, 51, 80, 0.35)",
                height=250,
                margin=dict(t=75, b=25, l=45, r=45)
            )
            return fig, f"Scorecard: {title}"

    # Extract valid numeric columns for single scalar or multi-card strip
    num_cols = []
    for c in numeric_candidates:
        val = pd.to_numeric(str(df.iloc[0][c]).replace(',', '').replace('%', ''), errors='coerce')
        if pd.notna(val):
            num_cols.append((c, val))
            
    # Case 2: Single scalar metric (e.g. 1 value or 1 period)
    if len(num_cols) == 1:
        c_name, val = num_cols[0]
        label = metric_label if metric_label and metric_label != str(df.iloc[0, 0]) else c_name
        is_pct = is_percentage_metric(label) or is_percentage_metric(c_name) or (0 <= val <= 100 and '%' in str(df.iloc[0][c_name]))
        sfx = "%" if is_pct else ""
        fmt = ".2f" if (is_pct and val < 10) else (".1f" if is_pct else ",.1f")
        
        fig = go.Figure()
        indicator_params = {
            "mode": "number+gauge" if (is_pct and 0 <= val <= 100) else "number",
            "value": val,
            "number": {
                "suffix": sfx,
                "valueformat": fmt,
                "font": {"size": 44, "color": "#F8FAFC", "family": "Inter, Segoe UI, sans-serif"}
            },
            "title": {
                "text": f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>{metric_label} ({c_name})</span>"
            }
        }
        if is_pct and 0 <= val <= 100:
            indicator_params["gauge"] = {
                "shape": "bullet",
                "axis": {"range": [0, 100], "tickfont": {"color": "#94A3B8", "size": 11}},
                "bar": {"color": "#38BDF8"},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "bordercolor": "rgba(255, 255, 255, 0.15)"
            }
        fig.add_trace(go.Indicator(**indicator_params))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(10, 25, 41, 0.95)",
            plot_bgcolor="rgba(26, 51, 80, 0.35)",
            height=230,
            margin=dict(t=70, b=25, l=45, r=45)
        )
        return fig, f"Scorecard: {title}"

    # Case 3: 2 to 4 key metrics in 1 row (Multi-metric dashboard strip)
    if 2 <= len(num_cols) <= 4:
        fig = go.Figure()
        n = len(num_cols)
        for i, (c_name, val) in enumerate(num_cols):
            is_pct = is_percentage_metric(c_name) or (0 <= val <= 100 and '%' in str(df.iloc[0][c_name]))
            sfx = "%" if is_pct else ""
            fmt = ".1f" if is_pct else ",.1f"
            fig.add_trace(go.Indicator(
                mode="number",
                value=val,
                number={"suffix": sfx, "valueformat": fmt, "font": {"size": 32, "color": "#F8FAFC"}},
                title={"text": f"<span style='font-size:12px;color:#94A3B8'>{c_name}</span>"},
                domain={'row': 0, 'column': i}
            ))
        fig.update_layout(
            grid={'rows': 1, 'columns': n, 'pattern': 'independent'},
            title=dict(text=f"<b>{title}</b>", x=0.02),
            template="plotly_dark",
            paper_bgcolor="rgba(10, 25, 41, 0.95)",
            height=220,
            margin=dict(t=65, b=20, l=30, r=30)
        )
        return fig, f"Scorecard Strip: {title}"

    return None


def handle_single_row_period_dataset(df: pd.DataFrame, title: str) -> Optional[Tuple[go.Figure, str]]:
    """Backward compatibility wrapper delegating to create_financial_scorecard."""
    return create_financial_scorecard(df, title)



def assign_financial_bar_color(label: str, index: int) -> str:
    """Assign semantic financial colors based on metric meaning."""
    from streamlit_app.config.colors import (
        COLOR_REVENUE, COLOR_EBITDA, COLOR_PROFIT, COLOR_COST_DRAG, COLOR_BASELINE
    )
    lbl = str(label).lower()
    if any(k in lbl for k in ["revenue", "sales", "turnover", "top line", "topline"]):
        return COLOR_REVENUE
    elif any(k in lbl for k in ["ebitda", "ebit", "operating profit", "operating income"]):
        return COLOR_EBITDA
    elif any(k in lbl for k in ["net income", "profit", "earnings", "pat"]):
        return COLOR_PROFIT
    elif any(k in lbl for k in ["gap", "drag", "spread", "loss", "cost", "expense", "headwind"]):
        return COLOR_COST_DRAG
    elif any(k in lbl for k in ["prior", "previous", "benchmark", "budget"]):
        return COLOR_BASELINE
    return CHART_COLORS[index % len(CHART_COLORS)]


def create_growth_divergence_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str) -> go.Figure:
    """
    Executive Growth Divergence / Margin Drag Chart.
    Clearly illustrates the growth spread between top-line and bottom-line metrics.
    """
    labels = df[x_col].astype(str).tolist()
    raw_vals = pd.to_numeric(df[y_col], errors='coerce').fillna(0).tolist()
    
    colors = [assign_financial_bar_color(lbl, i) for i, lbl in enumerate(labels)]
    text_labels = [f"+{v:.1f}%" if v > 0 else f"{v:.1f}%" for v in raw_vals]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=labels,
        y=raw_vals,
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=text_labels,
        textposition='outside',
        textfont=dict(size=14, color='#F8FAFC', family='Inter, Segoe UI, sans-serif'),
        hovertemplate='<b>%{x}</b><br>Reported Rate: %{y:.1f}%<extra></extra>',
        width=0.42
    ))
    
    # Calculate margin gap callout if comparing 2 metrics
    if len(raw_vals) >= 2:
        gap = raw_vals[1] - raw_vals[0]
        gap_sign = "+" if gap > 0 else ""
        annotation_text = f"Growth Spread: {gap_sign}{gap:.1f} pp"
        
        # Calculate implied relative margin change if top-line is non-zero
        if raw_vals[0] != -100:
            implied_drag = ((1.0 + raw_vals[1] / 100.0) / (1.0 + raw_vals[0] / 100.0) - 1.0) * 100.0
            annotation_text += f"<br>Relative Margin Drag: {implied_drag:.1f}%"
            
        max_y = max(raw_vals) if raw_vals else 10
        fig.add_annotation(
            x=0.5,
            y=max_y * 1.08,
            xref="x",
            yref="y",
            text=f"<b>{annotation_text}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#EF4444" if gap < 0 else "#10B981",
            ax=0,
            ay=-32,
            bgcolor="rgba(15, 23, 42, 0.90)",
            bordercolor="#EF4444" if gap < 0 else "#10B981",
            borderwidth=1,
            borderpad=6,
            font=dict(color="#F8FAFC", size=12)
        )
        
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>Financial Growth Divergence & Margin Analysis</span>",
            x=0.02,
            y=0.96
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=13, color="#CBD5E1"),
            showgrid=False
        ),
        yaxis=dict(
            title="Growth Rate (%)",
            ticksuffix="%",
            tickfont=dict(size=12, color="#94A3B8"),
            gridcolor="rgba(255, 255, 255, 0.08)",
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.25)"
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(10,25,41,0.95)',
        plot_bgcolor='rgba(26,51,80,0.35)',
        font=dict(color='#E3F2FD'),
        height=460,
        margin=dict(t=85, b=40, l=60, r=40)
    )
    return fig


def create_waterfall_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str) -> go.Figure:
    """Create a Plotly waterfall bridge chart for margin or variance walk."""
    labels = df[x_col].astype(str).tolist()
    raw_vals = pd.to_numeric(df[y_col], errors='coerce').fillna(0).tolist()
    
    measure = ["relative"] * len(labels)
    if len(labels) >= 3 and any(w in labels[-1].lower() for w in ["total", "ebitda", "net", "end", "final"]):
        measure[-1] = "total"
        
    is_pct = is_percentage_metric(y_col, df[y_col])
    text_vals = [f"{v:+.1f}%" if is_pct else f"{v:+,.1f}" for v in raw_vals]
    
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measure,
        x=labels,
        textposition="outside",
        text=text_vals,
        y=raw_vals,
        connector={"line": {"color": "rgba(255,255,255,0.25)"}},
        increasing={"marker": {"color": "#10B981"}},
        decreasing={"marker": {"color": "#EF4444"}},
        totals={"marker": {"color": "#3B82F6"}},
        width=0.45
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>Financial Variance & Bridge Walk</span>",
            x=0.02
        ),
        yaxis=dict(
            ticksuffix="%" if is_pct else "",
            gridcolor="rgba(255,255,255,0.08)"
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(10,25,41,0.95)',
        plot_bgcolor='rgba(26,51,80,0.35)',
        font=dict(color='#E3F2FD'),
        height=460,
        margin=dict(t=85, b=40, l=60, r=40)
    )
    return fig


def create_donut_chart(df: pd.DataFrame, title: str, x_col: str, y_col: str) -> go.Figure:
    """
    Renders an executive donut chart for share-of-whole, concentration, and mix distributions.
    Displays center callout with the dominant slice.
    """
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=df[x_col].astype(str),
        values=df[y_col],
        hole=0.62,
        marker=dict(colors=CHART_COLORS, line=dict(color='#0A1929', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>Value: %{value:,.1f}%<br>Share: %{percent}<extra></extra>',
        textfont=dict(size=12, color='#F8FAFC')
    ))
    
    # Center callout for dominant category
    try:
        top_idx = df[y_col].idxmax()
        top_label = str(df.loc[top_idx, x_col])
        top_val = df.loc[top_idx, y_col]
        center_text = f"<b>{top_val:.1f}%</b><br><span style='font-size:11px;color:#94A3B8'>{top_label[:14]}</span>"
    except Exception:
        center_text = f"<b>100%</b><br><span style='font-size:11px;color:#94A3B8'>Total</span>"

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>Distribution & Mix</span>", x=0.02),
        annotations=[
            dict(
                text=center_text,
                x=0.5, y=0.5,
                font=dict(size=18, color="#F8FAFC"),
                showarrow=False
            )
        ],
        template="plotly_dark",
        paper_bgcolor='rgba(10,25,41,0.95)',
        font=dict(color='#E3F2FD'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=450,
        margin=dict(t=75, b=50, l=40, r=40)
    )
    return fig


def create_horizontal_bar_chart(
    df: pd.DataFrame,
    title: str,
    x_col: str,
    y_cols: List[str],
    is_pct: bool = False
) -> go.Figure:
    """
    Renders clean horizontal bars for rankings, entity benchmarking, or categories with long names.
    Prevents tilted, overlapping, or truncated category labels.
    """
    fig = go.Figure()
    fmt = '.1f' if is_pct else ',.1f'
    sfx = '%' if is_pct else ''
    
    for idx, col in enumerate(y_cols):
        vals = df[col].tolist()
        text_vals = [f"{v:{fmt}}{sfx}" if pd.notna(v) else "" for v in vals]
        fig.add_trace(go.Bar(
            y=df[x_col].astype(str),
            x=df[col],
            name=str(col),
            orientation='h',
            marker=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], line=dict(color='rgba(255,255,255,0.15)', width=1)),
            text=text_vals,
            textposition='outside',
            textfont=dict(size=12, color='#F8FAFC'),
            hovertemplate=f'<b>%{{y}}</b><br>{col}: %{{x:{fmt}}}{sfx}<extra></extra>'
        ))
        
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.02),
        xaxis=dict(
            title="Rate (%)" if is_pct else "Value",
            ticksuffix=sfx,
            tickfont=dict(size=12, color="#94A3B8"),
            gridcolor='rgba(255,255,255,0.08)'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color="#CBD5E1"),
            autorange="reversed"
        ),
        barmode='group' if len(y_cols) > 1 else 'relative',
        template="plotly_dark",
        paper_bgcolor='rgba(10,25,41,0.95)',
        plot_bgcolor='rgba(26,51,80,0.35)',
        font=dict(color='#E3F2FD'),
        showlegend=True if len(y_cols) > 1 else False,
        height=max(360, len(df) * 45 + 160),
        margin=dict(t=70, b=40, l=140, r=40)
    )
    return fig


def create_chart_from_dataframe(
    df: pd.DataFrame,
    chart_type: str,
    title: str,
    metric_name: str = "Value"
) -> Optional[go.Figure]:
    """Create a Plotly chart from dataframe with financial storytelling styles"""
    try:
        if df is None or len(df) == 0 or len(df.columns) < 2:
            return None

        chart_type = chart_type.lower().strip()

        # 1. Single row dataset: Scorecard / KPI Indicator
        if len(df) == 1 or chart_type in ['scorecard', 'indicator', 'kpi']:
            single_res = create_financial_scorecard(df, title)
            if single_res is not None:
                return single_res[0]

        # 2. Classify columns
        cols_info = clean_and_classify_columns(df)
        x_col = cols_info['x_col']
        dimension_cols = cols_info['dimension_cols']
        quantity_cols = cols_info['quantity_cols']
        pct_cols = cols_info['pct_cols']

        # 3. Handle Categorical Dimension grouping (e.g. Metric + Reporting Scope / Currency)
        if dimension_cols and (quantity_cols or pct_cols):
            dim_col = dimension_cols[0]
            val_col = (pct_cols + quantity_cols)[0]
            val_is_pct = val_col in pct_cols
            fmt = '.1f' if val_is_pct else ',.1f'
            sfx = '%' if val_is_pct else ''
            
            df[val_col] = pd.to_numeric(
                df[val_col].astype(str).str.replace(',', '').str.replace('%', ''),
                errors='coerce'
            )
            
            categories = [c for c in df[dim_col].unique() if pd.notna(c) and str(c).strip()]
            fig = go.Figure()
            for idx, cat in enumerate(categories):
                sub_df = df[df[dim_col] == cat]
                raw_vals = sub_df[val_col].tolist()
                text_vals = []
                for v in raw_vals:
                    if pd.isna(v):
                        text_vals.append("")
                    elif val_is_pct and v > 0:
                        text_vals.append(f"+{v:{fmt}}{sfx}")
                    else:
                        text_vals.append(f"{v:{fmt}}{sfx}")
                        
                fig.add_trace(go.Bar(
                    x=sub_df[x_col],
                    y=sub_df[val_col],
                    name=str(cat),
                    marker_color=CHART_COLORS[idx % len(CHART_COLORS)],
                    text=text_vals,
                    textposition='outside',
                    textfont=dict(size=12, color='#F8FAFC'),
                    hovertemplate=f'<b>%{{x}} ({cat})</b><br>{val_col}: %{{y:{fmt}}}{sfx}<extra></extra>'
                ))
                
            fig.update_layout(
                title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>Grouped by {dim_col}</span>", x=0.02),
                xaxis=dict(title="", tickfont=dict(size=12, color="#CBD5E1")),
                yaxis=dict(
                    title="Rate (%)" if val_is_pct else metric_name,
                    ticksuffix=sfx,
                    tickfont=dict(size=12, color="#94A3B8"),
                    gridcolor='rgba(255,255,255,0.08)'
                ),
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                plot_bgcolor='rgba(26,51,80,0.35)',
                font=dict(color='#E3F2FD'),
                legend=dict(title=dict(text=dim_col, font=dict(size=12, color="#94A3B8")), bgcolor='rgba(15,23,42,0.6)'),
                height=480,
                hovermode='x unified'
            )
            return fig

        # 4. Clean remaining numeric columns
        for col in quantity_cols + pct_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')

        # 5. Check for Donut / Pie chart
        is_composition_data = False
        if pct_cols and len(pct_cols) == 1 and not quantity_cols and len(df) >= 2:
            sum_val = df[pct_cols[0]].sum()
            if 80.0 <= sum_val <= 120.0 or any(k in title.lower() for k in ['share', 'mix', 'breakdown', 'concentration', 'distribution', 'split']):
                is_composition_data = True

        if chart_type in ['donut', 'pie'] or (is_composition_data and chart_type not in ['grouped_bar', 'horizontal_bar']):
            val_col = pct_cols[0] if pct_cols else (quantity_cols[0] if quantity_cols else None)
            if val_col:
                return create_donut_chart(df, title, x_col, val_col)

        # 6. Check for Horizontal Bar (explicit or long labels)
        has_long_labels = any(len(str(lbl)) > 15 for lbl in df[x_col])
        if chart_type in ['horizontal_bar', 'hbar', 'ranking'] or (has_long_labels and chart_type not in ['line', 'donut']):
            target_cols = pct_cols if pct_cols and not quantity_cols else quantity_cols
            if target_cols:
                is_pct_h = len(pct_cols) > 0 and len(quantity_cols) == 0
                return create_horizontal_bar_chart(df, title, x_col, target_cols, is_pct=is_pct_h)

        # 7. Check for Growth Divergence (2-3 rows, only percentage columns)
        if len(df) in [2, 3] and pct_cols and not quantity_cols and not is_composition_data:
            y_col_primary = pct_cols[0]
            return create_growth_divergence_chart(df, title, x_col, y_col_primary)

        # 8. Check for Waterfall Bridge
        if chart_type in ['waterfall', 'bridge'] or any(w in title.lower() for w in ['bridge', 'waterfall', 'walk']):
            y_col_target = (quantity_cols + pct_cols)[0] if (quantity_cols + pct_cols) else None
            if y_col_target:
                return create_waterfall_chart(df, title, x_col, y_col_target)

        # 9. Scale segregation: if quantity_cols and pct_cols coexist
        if quantity_cols and pct_cols:
            active_y_cols = quantity_cols
            pct_col_for_badge = pct_cols[0]
        elif quantity_cols:
            active_y_cols = quantity_cols
            pct_col_for_badge = None
        elif pct_cols:
            active_y_cols = pct_cols
            pct_col_for_badge = None
        else:
            return None

        # 10. Guarding Line Charts: Strict >= 3 periods requirement
        if chart_type == 'line' and len(df) < 3 and len(active_y_cols) < 3:
            log_info(f"Guarding rule: Converting sparse line chart (<3 points) to grouped_bar for '{title}'")
            chart_type = 'grouped_bar'

        is_overall_pct = (len(quantity_cols) == 0 and len(pct_cols) > 0)
        fig = go.Figure()

        if chart_type == 'line':
            for i, y_col in enumerate(active_y_cols):
                is_col_pct = is_percentage_metric(y_col, df[y_col])
                fmt = '.1f' if is_col_pct else ',.1f'
                sfx = '%' if is_col_pct else ''
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    name=str(y_col),
                    mode='lines+markers',
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:{fmt}}}{sfx}<extra></extra>'
                ))
            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", x=0.02),
                xaxis_title=str(x_col),
                yaxis_title="Rate (%)" if is_overall_pct else metric_name,
                yaxis=dict(ticksuffix="%" if is_overall_pct else "", gridcolor='rgba(255,255,255,0.08)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                plot_bgcolor='rgba(26,51,80,0.35)',
                font=dict(color='#E3F2FD'),
                hovermode='x unified',
                showlegend=True,
                height=480
            )

        elif chart_type in ['bar', 'grouped_bar']:
            if len(active_y_cols) == 1:
                y_col = active_y_cols[0]
                colors = [assign_financial_bar_color(lbl, i) for i, lbl in enumerate(df[x_col])]
                fmt = '.1f' if is_overall_pct else ',.1f'
                sfx = '%' if is_overall_pct else ''
                text_vals = []
                for idx, v in enumerate(df[y_col]):
                    badge = ""
                    if pct_col_for_badge and pd.notna(df[pct_col_for_badge].iloc[idx]):
                        pv = df[pct_col_for_badge].iloc[idx]
                        badge = f" (+{pv:.1f}%)" if pv > 0 else f" ({pv:.1f}%)"
                    text_vals.append(f"{v:{fmt}}{sfx}{badge}")

                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=str(y_col),
                    marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.15)', width=1)),
                    text=text_vals,
                    textposition='outside',
                    textfont=dict(size=13, color='#F8FAFC'),
                    hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:{fmt}}}{sfx}<extra></extra>',
                    width=0.45
                ))
            else:
                for idx, y_col in enumerate(active_y_cols):
                    is_col_pct = is_percentage_metric(y_col, df[y_col])
                    fmt = '.1f' if is_col_pct else ',.1f'
                    sfx = '%' if is_col_pct else ''

                    is_last_col = (idx == len(active_y_cols) - 1)
                    text_vals = []
                    for row_idx, v in enumerate(df[y_col]):
                        badge = ""
                        if is_last_col and pct_col_for_badge and pd.notna(df[pct_col_for_badge].iloc[row_idx]):
                            pv = df[pct_col_for_badge].iloc[row_idx]
                            badge = f" (+{pv:.1f}%)" if pv > 0 else f" ({pv:.1f}%)"
                        text_vals.append(f"{v:{fmt}}{sfx}{badge}")

                    fig.add_trace(go.Bar(
                        x=df[x_col],
                        y=df[y_col],
                        name=str(y_col),
                        marker_color=CHART_COLORS[idx % len(CHART_COLORS)],
                        text=text_vals,
                        textposition='outside',
                        textfont=dict(size=12, color='#F8FAFC'),
                        hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:{fmt}}}{sfx}<extra></extra>'
                    ))

            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", x=0.02),
                xaxis_title="",
                yaxis_title="Rate (%)" if is_overall_pct else metric_name,
                yaxis=dict(ticksuffix="%" if is_overall_pct else "", gridcolor='rgba(255,255,255,0.08)'),
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                plot_bgcolor='rgba(26,51,80,0.35)',
                font=dict(color='#E3F2FD'),
                showlegend=True if len(active_y_cols) > 1 else False,
                height=480,
                hovermode='x unified'
            )

        elif chart_type == 'pie':
            y_col = active_y_cols[0]
            fig.add_trace(go.Pie(
                labels=df[x_col],
                values=df[y_col],
                marker=dict(colors=CHART_COLORS),
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                hovertemplate='<b>%{label}</b><br>Value: %{value:,.1f}<br>Share: %{percent}<extra></extra>',
                pull=[0.04] * len(df)
            ))
            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", x=0.02),
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                font=dict(color='#E3F2FD'),
                height=480,
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
    Incorporates Decision Gating, Column Classification, Dimension Grouping,
    and Scale-Segregated Financial Storytelling layouts.
    """
    try:
        df = pd.DataFrame(dataset.get('rows', []))
        if len(df) == 0 or len(df.columns) < 2:
            return None

        title = dataset.get('name', 'Financial Data')
        chart_type = dataset.get('suggested_chart', 'bar').lower().strip()

        # 1. Check for single-row dataset: ALWAYS route to Financial Scorecard / KPI Indicator
        # This handles single scalars (e.g. 72% Americas concentration) and 2-period comparisons (2022 vs 2023)
        if len(df) == 1 or chart_type in ['scorecard', 'indicator', 'kpi']:
            scorecard_res = create_financial_scorecard(df, title, dataset.get('description', ''))
            if scorecard_res is not None:
                return scorecard_res

        # 2. Classify columns
        cols_info = clean_and_classify_columns(df)
        x_col = cols_info['x_col']
        dimension_cols = cols_info['dimension_cols']
        quantity_cols = cols_info['quantity_cols']
        pct_cols = cols_info['pct_cols']

        # 3. Handle Categorical Dimension grouping (e.g. Metric + Reporting Scope / Currency)
        if dimension_cols and (quantity_cols or pct_cols):
            dim_col = dimension_cols[0]
            val_col = (pct_cols + quantity_cols)[0]
            val_is_pct = val_col in pct_cols
            fmt = '.1f' if val_is_pct else ',.1f'
            sfx = '%' if val_is_pct else ''
            
            df[val_col] = pd.to_numeric(
                df[val_col].astype(str).str.replace(',', '').str.replace('%', ''),
                errors='coerce'
            )
            
            categories = [c for c in df[dim_col].unique() if pd.notna(c) and str(c).strip()]
            fig = go.Figure()
            for idx, cat in enumerate(categories):
                sub_df = df[df[dim_col] == cat]
                raw_vals = sub_df[val_col].tolist()
                text_vals = []
                for v in raw_vals:
                    if pd.isna(v):
                        text_vals.append("")
                    elif val_is_pct and v > 0:
                        text_vals.append(f"+{v:{fmt}}{sfx}")
                    else:
                        text_vals.append(f"{v:{fmt}}{sfx}")
                        
                fig.add_trace(go.Bar(
                    x=sub_df[x_col],
                    y=sub_df[val_col],
                    name=str(cat),
                    marker_color=CHART_COLORS[idx % len(CHART_COLORS)],
                    text=text_vals,
                    textposition='outside',
                    textfont=dict(size=12, color='#F8FAFC'),
                    hovertemplate=f'<b>%{{x}} ({cat})</b><br>{val_col}: %{{y:{fmt}}}{sfx}<extra></extra>'
                ))
                
            fig.update_layout(
                title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:#94A3B8'>Grouped by {dim_col}</span>", x=0.02),
                xaxis=dict(title="", tickfont=dict(size=12, color="#CBD5E1")),
                yaxis=dict(
                    title="Rate (%)" if val_is_pct else "Value",
                    ticksuffix=sfx,
                    tickfont=dict(size=12, color="#94A3B8"),
                    gridcolor='rgba(255,255,255,0.08)'
                ),
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                plot_bgcolor='rgba(26,51,80,0.35)',
                font=dict(color='#E3F2FD'),
                legend=dict(title=dict(text=dim_col, font=dict(size=12, color="#94A3B8")), bgcolor='rgba(15,23,42,0.6)'),
                height=480,
                hovermode='x unified'
            )
            return fig, f"📊 {dataset.get('description', title)}"

        # 4. Clean remaining numeric columns
        for col in quantity_cols + pct_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce')

        # 5. Check for Donut / Pie chart (composition / shares summing ~100%)
        is_composition_data = False
        if pct_cols and len(pct_cols) == 1 and not quantity_cols and len(df) >= 2:
            sum_val = df[pct_cols[0]].sum()
            if 80.0 <= sum_val <= 120.0 or any(k in title.lower() for k in ['share', 'mix', 'breakdown', 'concentration', 'distribution', 'split']):
                is_composition_data = True

        if chart_type in ['donut', 'pie'] or (is_composition_data and chart_type not in ['grouped_bar', 'horizontal_bar']):
            val_col = pct_cols[0] if pct_cols else (quantity_cols[0] if quantity_cols else None)
            if val_col:
                fig = create_donut_chart(df, title, x_col, val_col)
                return (fig, f"📊 Composition: {dataset.get('description', title)}")

        # 6. Check for Horizontal Bar (explicit or categories with long names)
        has_long_labels = any(len(str(lbl)) > 15 for lbl in df[x_col])
        if chart_type in ['horizontal_bar', 'hbar', 'ranking'] or (has_long_labels and chart_type not in ['line', 'donut']):
            target_cols = pct_cols if pct_cols and not quantity_cols else quantity_cols
            if target_cols:
                is_pct_h = len(pct_cols) > 0 and len(quantity_cols) == 0
                fig = create_horizontal_bar_chart(df, title, x_col, target_cols, is_pct=is_pct_h)
                return (fig, f"📊 Comparison: {dataset.get('description', title)}")

        # 7. Check for Growth Divergence (2-3 rows, only percentage columns)
        if len(df) in [2, 3] and pct_cols and not quantity_cols and not is_composition_data:
            y_col_primary = pct_cols[0]
            fig = create_growth_divergence_chart(df, title, x_col, y_col_primary)
            return (fig, f"📊 Growth Divergence: {title}")

        # 8. Check for Waterfall Bridge
        if chart_type in ['waterfall', 'bridge'] or any(w in title.lower() for w in ['bridge', 'waterfall', 'walk']):
            y_col_target = (quantity_cols + pct_cols)[0] if (quantity_cols + pct_cols) else None
            if y_col_target:
                fig = create_waterfall_chart(df, title, x_col, y_col_target)
                return (fig, f"📊 Variance Walk: {title}")

        # 9. Scale segregation: if quantity_cols and pct_cols coexist
        if quantity_cols and pct_cols:
            active_y_cols = quantity_cols
            pct_col_for_badge = pct_cols[0]
        elif quantity_cols:
            active_y_cols = quantity_cols
            pct_col_for_badge = None
        elif pct_cols:
            active_y_cols = pct_cols
            pct_col_for_badge = None
        else:
            return None

        # 10. Guarding Line Charts: Strict >= 3 periods requirement
        if chart_type == 'line' and len(df) < 3 and len(active_y_cols) < 3:
            log_info(f"Guarding rule: Converting sparse line chart (<3 points) to grouped_bar for '{title}'")
            chart_type = 'grouped_bar'

        fig = go.Figure()
        is_overall_pct = (len(quantity_cols) == 0 and len(pct_cols) > 0)

        if chart_type == 'line':
            for i, y_col in enumerate(active_y_cols):
                is_col_pct = is_percentage_metric(y_col, df[y_col])
                fmt = '.1f' if is_col_pct else ',.1f'
                sfx = '%' if is_col_pct else ''
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines+markers',
                    name=str(y_col),
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=3),
                    marker=dict(size=8, symbol='circle'),
                    hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:{fmt}}}{sfx}<extra></extra>'
                ))
            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", x=0.02),
                xaxis_title=str(x_col),
                yaxis_title="Rate (%)" if is_overall_pct else "Value",
                yaxis=dict(ticksuffix="%" if is_overall_pct else "", gridcolor='rgba(255,255,255,0.08)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                plot_bgcolor='rgba(26,51,80,0.35)',
                font=dict(color='#E3F2FD'),
                hovermode='x unified',
                showlegend=True,
                height=480
            )

        elif chart_type in ['bar', 'grouped_bar']:
            if len(active_y_cols) == 1:
                y_col = active_y_cols[0]
                colors = [assign_financial_bar_color(lbl, i) for i, lbl in enumerate(df[x_col])]
                fmt = '.1f' if is_overall_pct else ',.1f'
                sfx = '%' if is_overall_pct else ''
                text_vals = []
                for idx, v in enumerate(df[y_col]):
                    badge = ""
                    if pct_col_for_badge and pd.notna(df[pct_col_for_badge].iloc[idx]):
                        pv = df[pct_col_for_badge].iloc[idx]
                        badge = f" (+{pv:.1f}%)" if pv > 0 else f" ({pv:.1f}%)"
                    text_vals.append(f"{v:{fmt}}{sfx}{badge}")

                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_col],
                    name=str(y_col),
                    marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.15)', width=1)),
                    text=text_vals,
                    textposition='outside',
                    textfont=dict(size=13, color='#F8FAFC'),
                    hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:{fmt}}}{sfx}<extra></extra>',
                    width=0.45
                ))
            else:
                for idx, y_col in enumerate(active_y_cols):
                    is_col_pct = is_percentage_metric(y_col, df[y_col])
                    fmt = '.1f' if is_col_pct else ',.1f'
                    sfx = '%' if is_col_pct else ''

                    is_last_col = (idx == len(active_y_cols) - 1)
                    text_vals = []
                    for row_idx, v in enumerate(df[y_col]):
                        badge = ""
                        if is_last_col and pct_col_for_badge and pd.notna(df[pct_col_for_badge].iloc[row_idx]):
                            pv = df[pct_col_for_badge].iloc[row_idx]
                            badge = f" (+{pv:.1f}%)" if pv > 0 else f" ({pv:.1f}%)"
                        text_vals.append(f"{v:{fmt}}{sfx}{badge}")

                    fig.add_trace(go.Bar(
                        x=df[x_col],
                        y=df[y_col],
                        name=str(y_col),
                        marker_color=CHART_COLORS[idx % len(CHART_COLORS)],
                        text=text_vals,
                        textposition='outside',
                        textfont=dict(size=12, color='#F8FAFC'),
                        hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:{fmt}}}{sfx}<extra></extra>'
                    ))

            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", x=0.02),
                xaxis_title="",
                yaxis_title="Rate (%)" if is_overall_pct else "Value",
                yaxis=dict(ticksuffix="%" if is_overall_pct else "", gridcolor='rgba(255,255,255,0.08)'),
                barmode='group',
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                plot_bgcolor='rgba(26,51,80,0.35)',
                font=dict(color='#E3F2FD'),
                showlegend=True if len(active_y_cols) > 1 else False,
                height=480,
                hovermode='x unified'
            )

        elif chart_type == 'pie':
            y_col = active_y_cols[0]
            fig.add_trace(go.Pie(
                labels=df[x_col],
                values=df[y_col],
                marker=dict(colors=CHART_COLORS),
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                hovertemplate='<b>%{label}</b><br>Value: %{value:,.1f}<br>Share: %{percent}<extra></extra>',
                pull=[0.04] * len(df)
            ))
            fig.update_layout(
                title=dict(text=f"<b>{title}</b>", x=0.02),
                template="plotly_dark",
                paper_bgcolor='rgba(10,25,41,0.95)',
                font=dict(color='#E3F2FD'),
                height=480,
                showlegend=True
            )

        description = dataset.get('description', f"{chart_type.capitalize()} chart")
        return (fig, f"📊 {description}")

    except Exception as e:
        log_exception("Error creating chart from dataset", e)
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
            # Iterate through all available datasets until we have collected up to max_charts valid charts
            for dataset in extracted_data.get('datasets', []):
                if len(visualizations) >= max_charts:
                    break
                chart_tuple = create_chart_from_dataset(dataset)
                if chart_tuple:
                    visualizations.append(chart_tuple)
                    # Convert dataset to DataFrame for table display
                    df = pd.DataFrame(dataset.get('rows', []))
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
