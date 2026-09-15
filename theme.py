"""Shared dark theme + animation system for every page. Import
inject_theme() at the top of each page (after st.set_page_config) so the
whole app looks and feels consistent — one source of truth for the
visual language instead of duplicating a CSS blob five times."""
import streamlit as st

ACCENT_1 = '#8B5CF6'   # electric violet
ACCENT_2 = '#22D3EE'   # cyan
ACCENT_3 = '#F472B6'   # pink
BG_DEEP = '#05070D'
BG_CARD = 'rgba(255,255,255,0.04)'
BORDER = 'rgba(255,255,255,0.08)'

EMOTION_COLORS = {
    'Angry': '#FB7185', 'Disgust': '#C084FC', 'Fear': '#FBBF24',
    'Happy': '#34D399', 'Neutral': '#94A3B8', 'Sad': '#38BDF8',
}
EMOTION_EMOJI = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😊', 'Neutral': '😐', 'Sad': '😢',
}


def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(139,92,246,0.18), transparent),
            radial-gradient(ellipse 80% 50% at 100% 0%, rgba(34,211,238,0.12), transparent),
            {BG_DEEP};
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(10,12,20,0.85);
        border-right: 1px solid {BORDER};
    }}

    h1, h2, h3, h4 {{ color: #F1F5F9 !important; font-weight: 700 !important; }}
    p, span, label, .stMarkdown {{ color: #CBD5E1; }}
    small, .stCaption {{ color: #64748B !important; }}

    /* ---- animations ---- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes floatY {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}
    @keyframes glowPulse {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(139,92,246,0.35); }}
        50% {{ box-shadow: 0 0 40px rgba(34,211,238,0.45); }}
    }}
    @keyframes shimmer {{
        0% {{ background-position: -400px 0; }}
        100% {{ background-position: 400px 0; }}
    }}
    @keyframes barGrow {{
        from {{ width: 0%; }}
    }}

    .fade-in {{ animation: fadeInUp 0.6s ease-out both; }}
    .fade-in-1 {{ animation: fadeInUp 0.6s ease-out 0.05s both; }}
    .fade-in-2 {{ animation: fadeInUp 0.6s ease-out 0.15s both; }}
    .fade-in-3 {{ animation: fadeInUp 0.6s ease-out 0.25s both; }}
    .fade-in-4 {{ animation: fadeInUp 0.6s ease-out 0.35s both; }}
    .floaty {{ animation: floatY 4s ease-in-out infinite; }}
    .glow-pulse {{ animation: glowPulse 2.4s ease-in-out infinite; }}

    .hero-gradient-text {{
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2}, {ACCENT_3}, {ACCENT_1});
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite;
        font-weight: 800;
    }}

    .glass-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(12px);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(139,92,246,0.5);
    }}

    .badge {{
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
        text-transform: uppercase;
        background: linear-gradient(90deg, {ACCENT_1}33, {ACCENT_2}33);
        border: 1px solid {ACCENT_1}55; color: #E9D5FF;
    }}

    /* Streamlit widget restyling */
    .stButton > button {{
        background: linear-gradient(90deg, {ACCENT_1}, {ACCENT_2});
        color: white; border: none; border-radius: 12px; font-weight: 600;
        padding: 0.55rem 1.2rem; transition: all 0.25s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139,92,246,0.4);
    }}
    .stTextInput input, .stTextArea textarea {{
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid {BORDER} !important;
        color: #F1F5F9 !important; border-radius: 10px !important;
    }}
    div[data-testid="stMetric"] {{
        background: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 0.8rem 1rem;
    }}
    div[data-baseweb="tab-list"] {{ gap: 6px; }}
    button[data-baseweb="tab"] {{
        background: rgba(255,255,255,0.03); border-radius: 10px !important;
        color: #94A3B8 !important;
    }}
    button[aria-selected="true"] {{
        background: linear-gradient(90deg, {ACCENT_1}44, {ACCENT_2}44) !important;
        color: #F1F5F9 !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def hero_banner(title: str, subtitle: str, icon: str = '🎙️'):
    st.markdown(f"""
    <div class="fade-in" style="text-align:center; padding: 3rem 1rem 2rem 1rem;">
        <div class="floaty glow-pulse" style="display:inline-flex; align-items:center; justify-content:center;
             width:84px; height:84px; border-radius:50%;
             background: linear-gradient(135deg, {ACCENT_1}, {ACCENT_2}); font-size:2.4rem; margin-bottom:1rem;">
            {icon}
        </div>
        <h1 class="hero-gradient-text" style="font-size:3rem; margin:0;">{title}</h1>
        <p style="font-size:1.1rem; color:#94A3B8; max-width:560px; margin:0.8rem auto 0 auto;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str, delay_class: str = 'fade-in'):
    st.markdown(f"""
    <div class="glass-card {delay_class}" style="text-align:center; height:100%;">
        <div style="font-size:2rem;">{icon}</div>
        <h4 style="margin:0.5rem 0 0.3rem 0;">{title}</h4>
        <p style="font-size:0.88rem; margin:0;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)


def animated_bar(label: str, value: float, color: str, delay_ms: int = 0):
    """A custom animated probability/score bar — full control over the
    grow-in animation, unlike st.progress()."""
    pct = max(0.0, min(1.0, value)) * 100
    st.markdown(f"""
    <div style="margin-bottom:0.85rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
            <span style="color:#E2E8F0; font-weight:500;">{label}</span>
            <span style="color:#94A3B8;">{pct:.1f}%</span>
        </div>
        <div style="height:10px; border-radius:999px; background:rgba(255,255,255,0.06); overflow:hidden;">
            <div style="height:100%; width:{pct}%; border-radius:999px; background:{color};
                 animation: barGrow 1s ease-out {delay_ms}ms both;
                 box-shadow: 0 0 12px {color}99;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label: str, value: str, delay_class: str = 'fade-in'):
    st.markdown(f"""
    <div class="glass-card {delay_class}" style="text-align:center;">
        <p style="font-size:0.78rem; color:#64748B; margin:0; text-transform:uppercase; letter-spacing:0.04em;">{label}</p>
        <p style="font-size:1.5rem; font-weight:700; color:#F1F5F9; margin:0.2rem 0 0 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)
