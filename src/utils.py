"""
OptiMeal AI - Utilities & UI Styling Engine
Provides vibrant, bright, client-ready color palettes (Yellow, Green, Orange shades),
Plotly layout formatters, and clean component styles.
"""

import plotly.graph_objects as go
import plotly.express as px

# Vibrant Yellow - Green - Orange Sustainability Palette
COLOR_EMERALD = "#10B981"      # Fresh Green
COLOR_LEAF = "#059669"         # Deep Leaf Green
COLOR_LIME = "#84CC16"         # Bright Lime Green
COLOR_AMBER = "#F59E0B"        # Warm Amber / Golden Yellow
COLOR_YELLOW = "#EAB308"       # Bright Sun Yellow
COLOR_ORANGE = "#F97316"       # Vibrant Orange
COLOR_CORAL = "#EA580C"        # Deep Tangerine
COLOR_PEACH = "#FFF7ED"        # Warm light background tint
COLOR_LEMON = "#FEFCE8"        # Soft yellow tint
COLOR_MINT = "#ECFDF5"         # Soft mint tint

# Backward compatibility aliases
PRIMARY_GREEN = COLOR_LEAF
SECONDARY_GREEN = COLOR_EMERALD
DARK_GREEN = "#064E3B"
LIGHT_GREEN = COLOR_MINT
ACCENT_AMBER = COLOR_AMBER
ACCENT_RED = COLOR_CORAL

# Typography & Backgrounds
TEXT_DARK = "#1E293B"          # High-contrast slate text
TEXT_MUTED = "#64748B"         # Subdued label text
BG_WARM = "#FCFDFB"            # Bright crisp warm background
BG_CARD = "#FFFFFF"
BORDER_LIGHT = "#E2E8F0"

# Chart Color Sequence (Yellows, Greens, Oranges)
COLOR_SEQUENCE = [
    "#10B981", # Vibrant Green
    "#F97316", # Warm Orange
    "#F59E0B", # Amber Yellow
    "#84CC16", # Lime Green
    "#EA580C", # Deep Coral Orange
    "#EAB308", # Sun Yellow
]

def format_kg(val: float) -> str:
    """Format weight in kilograms."""
    return f"{val:,.1f} kg"

def format_currency(val: float) -> str:
    """Format financial loss in currency."""
    return f"${val:,.2f}"

def format_pct(val: float) -> str:
    """Format percentage with sign."""
    return f"{val:+.1f}%" if val != 0 else "0.0%"

def get_provenance_badge(data_type: str) -> str:
    """
    Generate styled HTML badge with bright green/orange/yellow styling.
    """
    data_type_lower = data_type.lower()
    if data_type_lower == "authentic":
        return '<span style="background-color: #ECFDF5; color: #047857; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #A7F3D0; box-shadow: 0 1px 2px rgba(16,185,129,0.1);">✓ Verified Canteen Data</span>'
    elif data_type_lower == "derived":
        return '<span style="background-color: #FEFCE8; color: #A16207; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #FEF08A;">Calculated Feature</span>'
    elif data_type_lower == "synthetic":
        return '<span style="background-color: #FFF7ED; color: #C2410C; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #FFEDD5;">Simulated Parameter</span>'
    else:
        return f'<span style="background-color: #F8FAFC; color: #334155; font-size: 0.75rem; padding: 4px 10px; border-radius: 20px;">{data_type}</span>'

def apply_custom_plotly_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """
    Apply bright, high-clarity client presentation layout to Plotly charts.
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 15, "color": TEXT_DARK, "family": "'Plus Jakarta Sans', 'Inter', sans-serif"},
            "x": 0.01,
            "xanchor": "left"
        },
        paper_bgcolor="rgba(255,255,255,0.7)",
        plot_bgcolor="rgba(255,255,255,0.7)",
        height=height,
        margin={"l": 30, "r": 20, "t": 50, "b": 35},
        font={"family": "'Plus Jakarta Sans', 'Inter', sans-serif", "color": TEXT_DARK, "size": 12},
        xaxis={
            "gridcolor": "#F1F5F9",
            "zerolinecolor": "#E2E8F0",
            "tickfont": {"color": TEXT_MUTED, "size": 11}
        },
        yaxis={
            "gridcolor": "#F1F5F9",
            "zerolinecolor": "#E2E8F0",
            "tickfont": {"color": TEXT_MUTED, "size": 11}
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": TEXT_DARK, "size": 11}
        }
    )
    return fig

CUSTOM_CSS = """
<style>
    /* Bright, clean, modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Background: Bright, warm and lively */
    .stApp {
        background: linear-gradient(180deg, #FDFAF5 0%, #FAFAF7 100%) !important;
        background-color: #FDFAF5 !important;
        color: #1E293B !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }
    
    /* Eliminate Black / Dark Top Header */
    header[data-testid="stHeader"], .stApp > header {
        background-color: #FDFAF5 !important;
        background: #FDFAF5 !important;
        color: #1E293B !important;
        box-shadow: none !important;
        border-bottom: 1px solid #FED7AA !important;
    }
    
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] a, header[data-testid="stHeader"] span {
        color: #1E293B !important;
    }
    
    /* Sidebar: Bright crisp styling with warm amber/green accents */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-right: 1px solid #FED7AA !important;
        box-shadow: 2px 0 12px rgba(249, 115, 22, 0.04) !important;
    }
    
    /* Force high-contrast readable text in sidebar */
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Radio Navigation Options */
    section[data-testid="stSidebar"] [role="radiogroup"] {
        gap: 6px !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 4px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #FEF3C7 !important;
        border-color: #F59E0B !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label div p {
        color: #0F172A !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
    }
    
    /* Selected radio state in sidebar */
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: #ECFDF5 !important;
        border-color: #10B981 !important;
        box-shadow: 0 1px 4px rgba(16, 185, 129, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div p {
        color: #047857 !important;
        font-weight: 800 !important;
    }
    
    /* Main Hero Banner: Bright Gradient (Green to Warm Orange/Amber) */
    .hero-banner {
        background: linear-gradient(135deg, #10B981 0%, #F59E0B 55%, #F97316 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: #FFFFFF !important;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px -4px rgba(249, 115, 22, 0.22);
    }
    
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF !important;
    }
    
    .hero-subtitle {
        font-size: 1.0rem;
        color: #FEF3C7 !important;
        font-weight: 500;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Client-Ready Metric Cards with Vibrant Top Borders */
    .metric-card-green {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #10B981;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.06);
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-orange {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #F97316;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(249, 115, 22, 0.06);
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-yellow {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #F59E0B;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.06);
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card-lime {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #84CC16;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(132, 204, 22, 0.06);
        margin-bottom: 14px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .metric-subtext {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Clean, Vibrant Highlight Boxes */
    .insight-card-green {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #10B981;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    .insight-card-orange {
        background: #FFF7ED;
        border: 1px solid #FFEDD5;
        border-left: 5px solid #F97316;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    .insight-card-yellow {
        background: #FEFCE8;
        border: 1px solid #FEF08A;
        border-left: 5px solid #F59E0B;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    
    .card-title {
        font-size: 1.0rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    
    .card-body {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.5;
    }
    
    /* Form controls & Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        color: #475569;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ECFDF5 !important;
        border-color: #10B981 !important;
        color: #047857 !important;
        border-bottom: 2px solid #10B981 !important;
    }
</style>
"""
