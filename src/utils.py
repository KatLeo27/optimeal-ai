"""
OptiMeal AI - Utilities & UI Styling Engine
Provides elegant, balanced, client-ready color palettes (refined Emerald Green, Warm Amber, Terracotta),
Plotly layout formatters, and clean component styles.
"""

import plotly.graph_objects as go
import plotly.express as px

# Refined, Balanced Palette (Toned Down & Elegant)
COLOR_EMERALD = "#059669"      # Classic Emerald Green
COLOR_LEAF = "#047857"         # Deep Forest Leaf
COLOR_LIME = "#65A30D"         # Balanced Olive/Moss Green
COLOR_AMBER = "#D97706"        # Warm Amber
COLOR_YELLOW = "#CA8A04"       # Soft Golden
COLOR_ORANGE = "#EA580C"       # Warm Terracotta Orange
COLOR_CORAL = "#C2410C"        # Deep Rust
COLOR_PEACH = "#FFF7ED"        # Subtle warm tint
COLOR_LEMON = "#FEFCE8"        # Soft cream tint
COLOR_MINT = "#F0FDF4"         # Soft mint tint

# Backward compatibility aliases
PRIMARY_GREEN = COLOR_EMERALD
SECONDARY_GREEN = "#10B981"
DARK_GREEN = "#064E3B"
LIGHT_GREEN = COLOR_MINT
ACCENT_AMBER = COLOR_AMBER
ACCENT_RED = COLOR_ORANGE

# Typography & Backgrounds
TEXT_DARK = "#1E293B"          # High-contrast slate text
TEXT_MUTED = "#64748B"         # Subdued label text
BG_WARM = "#F8FAFC"            # Balanced soft neutral slate
BG_CARD = "#FFFFFF"
BORDER_LIGHT = "#E2E8F0"

# Chart Color Sequence
COLOR_SEQUENCE = [
    "#059669", # Emerald Green
    "#EA580C", # Warm Terracotta Orange
    "#D97706", # Balanced Amber
    "#65A30D", # Olive Green
    "#0284C7", # Soft Ocean Blue
    "#7C3AED", # Soft Purple
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
    Generate styled HTML badge with clean, balanced styling.
    """
    data_type_lower = data_type.lower()
    if data_type_lower == "authentic":
        return '<span style="background-color: #ECFDF5; color: #047857; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #A7F3D0;">✓ Verified Canteen Data</span>'
    elif data_type_lower == "derived":
        return '<span style="background-color: #FEFCE8; color: #854D0E; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #FEF08A;">Calculated Feature</span>'
    elif data_type_lower == "synthetic":
        return '<span style="background-color: #FFF7ED; color: #9A3412; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 20px; border: 1px solid #FFEDD5;">Simulated Parameter</span>'
    else:
        return f'<span style="background-color: #F8FAFC; color: #334155; font-size: 0.75rem; padding: 4px 10px; border-radius: 20px;">{data_type}</span>'

def apply_custom_plotly_layout(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    """
    Apply clean, balanced client presentation layout to Plotly charts.
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 14, "color": TEXT_DARK, "family": "'Plus Jakarta Sans', 'Inter', sans-serif"},
            "x": 0.01,
            "xanchor": "left"
        },
        paper_bgcolor="rgba(255,255,255,0.8)",
        plot_bgcolor="rgba(255,255,255,0.8)",
        height=height,
        margin={"l": 30, "r": 20, "t": 46, "b": 32},
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
    /* Clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Background: Softer, balanced neutral slate */
    .stApp {
        background-color: #F8FAFC !important;
        background: #F8FAFC !important;
        color: #1E293B !important;
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }
    
    /* Top Header: Seamlessly matches background */
    header[data-testid="stHeader"], .stApp > header {
        background-color: #F8FAFC !important;
        background: #F8FAFC !important;
        color: #1E293B !important;
        box-shadow: none !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    
    header[data-testid="stHeader"] button, header[data-testid="stHeader"] a, header[data-testid="stHeader"] span {
        color: #1E293B !important;
    }
    
    /* Sidebar: Crisp white with neutral border */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 1px 0 8px rgba(0, 0, 0, 0.02) !important;
    }
    
    /* High-contrast readable text in sidebar */
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Radio Navigation */
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
        background-color: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label div p {
        color: #0F172A !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
    }
    
    /* Selected radio state in sidebar */
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: #ECFDF5 !important;
        border-color: #059669 !important;
        box-shadow: 0 1px 3px rgba(5, 150, 105, 0.12) !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div p {
        color: #065F46 !important;
        font-weight: 800 !important;
    }
    
    /* Hero Banner: Deep, elegant emerald */
    .hero-banner {
        background: linear-gradient(135deg, #065F46 0%, #047857 60%, #0F766E 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: #FFFFFF !important;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px -2px rgba(4, 120, 87, 0.16);
    }
    
    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF !important;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #D1FAE5 !important;
        font-weight: 500;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Metric Cards: Clean white with 3px subtle color accent */
    .metric-card-green {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #059669;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .metric-card-orange {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #EA580C;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .metric-card-yellow {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #D97706;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .metric-card-lime {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #65A30D;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    
    .metric-subtext {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Highlight Boxes */
    .insight-card-green {
        background: #F0FDF4;
        border: 1px solid #DCFCE7;
        border-left: 4px solid #059669;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .insight-card-orange {
        background: #FFF7ED;
        border: 1px solid #FFEDD5;
        border-left: 4px solid #EA580C;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .insight-card-yellow {
        background: #FEFCE8;
        border: 1px solid #FEF08A;
        border-left: 4px solid #D97706;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }
    
    .card-body {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
    }
    
    /* Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.2) !important;
        transition: all 0.15s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #047857 0%, #065F46 100%) !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
        transform: translateY(-1px) !important;
    }
</style>
"""
