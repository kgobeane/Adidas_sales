import streamlit as st
import streamlit.components.v1 as components

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Adidas Sales Intelligence Platform",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

/* Hide Streamlit Branding */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}


/* Main Background */

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
}


/* Main Container */

.block-container{
    padding-top:1rem;
    padding-bottom:0rem;
    max-width:100%;
}


/* Sidebar */

[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white;
}


/* Hero Banner */

.hero{
    background:linear-gradient(90deg,#2563eb,#1d4ed8);
    padding:30px;
    border-radius:20px;
    margin-bottom:20px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.3);
}

.hero-title{
    color:white;
    font-size:40px;
    font-weight:bold;
}

.hero-sub{
    color:#E5E7EB;
    font-size:18px;
}


/* Metric Cards */

.metric-box{

    background:white;

    border-radius:15px;

    padding:15px;

    box-shadow:0px 5px 12px rgba(0,0,0,0.2);

}


/* Remove Top Padding */

section.main{
    padding-top:0rem;
}


/* Iframe */

iframe{
    border-radius:18px;
}

</style>

""", unsafe_allow_html=True)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.image("assets/adidas_logo.png", width=180)

st.sidebar.markdown("---")

st.sidebar.markdown("## 👟 Adidas")

st.sidebar.markdown("### Sales Intelligence Platform")

st.sidebar.markdown("---")

st.sidebar.success("Use the navigation above to explore the project.")

# -------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------

col1, col2 = st.columns([1,5])

with col1:
    st.image("assets/adidas_logo.png", width=120)

with col2:

    st.markdown("""

<div class="hero">

<div class="hero-title">

Adidas Sales Intelligence Platform

</div>

<div class="hero-sub">

Executive Dashboard • Forecasting • Machine Learning • Prescriptive Analytics

</div>

</div>

""", unsafe_allow_html=True)

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        label="💰 Revenue",
        value="$2.15M",
        delta="+12%"
    )

with k2:
    st.metric(
        label="🛒 Orders",
        value="5,483",
        delta="+9%"
    )

with k3:
    st.metric(
        label="🎯 Forecast Accuracy",
        value="96.4%",
        delta="+2%"
    )

with k4:
    st.metric(
        label="🏆 Best Model",
        value="XGBoost",
        delta="Top Performer"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# POWER BI DASHBOARD
# -------------------------------------------------------

powerbi_url = (
    "https://app.powerbi.com/view?"
    "r=eyJrIjoiOWJkZTA1NDUtY2E5MC00Y2NmLThhNDUtMWI5OTlmOTdmNzE5IiwidCI6IjQ0MWRiNzQ0LWY5NzUtNGI2Ny04YzU3LTA1NDFkMTI3NjM2MyJ9"
    "&navContentPaneEnabled=false"
    "&filterPaneEnabled=false"
)

components.iframe(
    powerbi_url,
    height=700,
    scrolling=False
)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown(
    """
    <center>

    <span style='color:white;font-size:16px;'>

    Developed using

    <b>Power BI</b> • <b>Python</b> • <b>Streamlit</b> • <b>XGBoost</b> • <b>Prophet</b>

    </span>

    </center>
    """,
    unsafe_allow_html=True
)
