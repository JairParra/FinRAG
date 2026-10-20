"""
Dark Blue Color Palette for FinReport RAG Streamlit App
Professional financial theme with dark mode aesthetic
"""

# Primary Colors
PRIMARY_DARK_BLUE = "#0A1929"      # Deep navy - main background
PRIMARY_BLUE = "#1E3A5F"            # Rich blue - secondary background
PRIMARY_LIGHT_BLUE = "#2E5090"      # Medium blue - cards/containers

# Accent Colors
ACCENT_CYAN = "#00D4FF"             # Bright cyan - highlights/CTAs
ACCENT_TEAL = "#00B8D4"             # Teal - interactive elements
ACCENT_SKY = "#4FC3F7"              # Sky blue - hover states

# Text Colors
TEXT_PRIMARY = "#E3F2FD"            # Light blue-white - primary text
TEXT_SECONDARY = "#B0BEC5"          # Gray-blue - secondary text
TEXT_MUTED = "#78909C"              # Muted gray - disabled/hints

# Semantic Colors
SUCCESS_GREEN = "#00C853"           # Success states
WARNING_ORANGE = "#FFB300"          # Warnings
ERROR_RED = "#FF5252"               # Errors
INFO_BLUE = "#2196F3"               # Information

# Background Variants
BG_DARK = "#0D1B2A"                 # Darkest background
BG_MEDIUM = "#1B2838"               # Medium background
BG_CARD = "#263238"                 # Card/panel background
BG_INPUT = "#1A2332"                # Input fields background

# Border Colors
BORDER_SUBTLE = "#263238"           # Subtle borders
BORDER_DEFAULT = "#37474F"          # Default borders
BORDER_ACCENT = "#00B8D4"           # Accent borders

# Chart Colors (for visualizations)
CHART_COLORS = [
    "#00D4FF",  # Cyan
    "#4FC3F7",  # Sky
    "#00B8D4",  # Teal
    "#2196F3",  # Blue
    "#64B5F6",  # Light blue
    "#1976D2",  # Dark blue
    "#0288D1",  # Blue variant
    "#00ACC1",  # Cyan variant
]

# Gradient Definitions
GRADIENT_PRIMARY = f"linear-gradient(135deg, {PRIMARY_DARK_BLUE} 0%, {PRIMARY_BLUE} 100%)"
GRADIENT_ACCENT = f"linear-gradient(135deg, {ACCENT_TEAL} 0%, {ACCENT_CYAN} 100%)"
GRADIENT_CARD = f"linear-gradient(180deg, {BG_CARD} 0%, {BG_MEDIUM} 100%)"


def get_custom_css():
    """Returns custom CSS for the Streamlit app"""
    return f"""
    <style>
        /* Main App Styling */
        .stApp {{
            background-color: {PRIMARY_DARK_BLUE};
            color: {TEXT_PRIMARY};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {GRADIENT_PRIMARY};
        }}
        
        [data-testid="stSidebar"] > div:first-child {{
            background: {GRADIENT_PRIMARY};
            overflow-y: auto;
        }}
        
        /* Hide scrollbar unless actually needed */
        [data-testid="stSidebar"] > div:first-child {{
            scrollbar-width: thin;
        }}
        
        [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {{
            width: 6px;
        }}
        
        [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb {{
            background: {PRIMARY_BLUE};
            border-radius: 3px;
        }}
        
        [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar-thumb:hover {{
            background: {ACCENT_TEAL};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_PRIMARY} !important;
        }}
        
        h1 {{
            color: {ACCENT_CYAN} !important;
            font-weight: 700;
        }}
        
        /* Cards and Containers */
        .element-container {{
            color: {TEXT_PRIMARY};
        }}
        
        /* Buttons */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }}
        
        /* Primary Button */
        .stButton > button[data-testid="baseButton-primary"] {{
            background: {GRADIENT_ACCENT};
            color: {PRIMARY_DARK_BLUE};
            border: none;
        }}
        
        /* Secondary Button (Default) - Used for suggested prompts */
        .stButton > button[data-testid="baseButton-secondary"] {{
            background: {BG_INPUT};
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER_DEFAULT};
        }}
        
        .stButton > button[data-testid="baseButton-primary"]:hover {{
            background: {ACCENT_CYAN};
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
        }}
        
        .stButton > button[data-testid="baseButton-secondary"]:hover {{
            border-color: {ACCENT_CYAN};
            color: {ACCENT_CYAN};
            background: {BG_MEDIUM};
        }}
        
        /* Text Input */
        .stTextInput > div > div > input {{
            background-color: {BG_INPUT};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_DEFAULT};
            border-radius: 8px;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: {ACCENT_CYAN};
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2);
        }}
        
        /* Text Area */
        .stTextArea > div > div > textarea {{
            background-color: {BG_INPUT};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_DEFAULT};
            border-radius: 8px;
        }}
        
        .stTextArea > div > div > textarea:focus {{
            border-color: {ACCENT_CYAN};
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2);
        }}
        
        /* File Uploader */
        [data-testid="stFileUploader"] {{
            background-color: {BG_CARD};
            border: 2px dashed {BORDER_ACCENT};
            border-radius: 12px;
            padding: 2rem;
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {ACCENT_CYAN};
            background-color: {BG_MEDIUM};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
            border-radius: 8px;
            border: 1px solid {BORDER_DEFAULT};
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: {BG_MEDIUM};
            border-color: {ACCENT_TEAL};
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {ACCENT_CYAN};
        }}
        
        [data-testid="stMetricDelta"] {{
            color: {SUCCESS_GREEN};
        }}
        
        /* Info/Warning/Error boxes */
        .stAlert {{
            background-color: {BG_CARD};
            border-left: 4px solid {INFO_BLUE};
            border-radius: 8px;
        }}
        
        /* Chat Messages */
        .stChatMessage {{
            background-color: {BG_CARD};
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid {BORDER_SUBTLE};
        }}
        
        /* Success Message */
        .success-message {{
            background-color: rgba(0, 200, 83, 0.1);
            border-left: 4px solid {SUCCESS_GREEN};
            padding: 1rem;
            border-radius: 8px;
            color: {TEXT_PRIMARY};
        }}
        
        /* Warning Message */
        .warning-message {{
            background-color: rgba(255, 179, 0, 0.1);
            border-left: 4px solid {WARNING_ORANGE};
            padding: 1rem;
            border-radius: 8px;
            color: {TEXT_PRIMARY};
        }}
        
        /* Error Message */
        .error-message {{
            background-color: rgba(255, 82, 82, 0.1);
            border-left: 4px solid {ERROR_RED};
            padding: 1rem;
            border-radius: 8px;
            color: {TEXT_PRIMARY};
        }}
        
        /* Dataframe */
        .dataframe {{
            background-color: {BG_CARD};
            color: {TEXT_PRIMARY};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {BG_CARD};
            border-radius: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {TEXT_SECONDARY};
            background-color: transparent;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {ACCENT_CYAN};
            border-bottom: 2px solid {ACCENT_CYAN};
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {BG_DARK};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {PRIMARY_BLUE};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {ACCENT_TEAL};
        }}
        
        /* Select box */
        .stSelectbox > div > div {{
            background-color: {BG_INPUT};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_DEFAULT};
            border-radius: 8px;
        }}
        
        /* Multiselect */
        .stMultiSelect > div > div {{
            background-color: {BG_INPUT};
            color: {TEXT_PRIMARY};
            border: 1px solid {BORDER_DEFAULT};
            border-radius: 8px;
        }}
        
        /* Slider */
        .stSlider > div > div > div {{
            background-color: {ACCENT_TEAL};
        }}
        
        /* Radio */
        .stRadio > div {{
            background-color: transparent;
        }}
        
        /* Checkbox */
        .stCheckbox > label {{
            color: {TEXT_PRIMARY};
        }}
        
        /* Links */
        a {{
            color: {ACCENT_CYAN};
            text-decoration: none;
        }}
        
        a:hover {{
            color: {ACCENT_SKY};
            text-decoration: underline;
        }}
        
        /* Progress bar */
        .stProgress > div > div > div > div {{
            background-color: {ACCENT_CYAN};
        }}
        
        /* Spinner */
        .stSpinner > div {{
            border-top-color: {ACCENT_CYAN} !important;
        }}
    </style>
    """
