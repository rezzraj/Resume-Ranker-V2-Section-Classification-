import streamlit as st
import pandas as pd
from text_preprocess import chunking_and_preprocessing, section_classifier, clean_jd, jd_embedding
from main import forward_pass
from text_extraction import extract_text

# -------------------------------------------------
# Your pipeline functions should exist above this file
# or be imported from another module.
#
# def extract_text(file):
#     ...
#     return full_text
#
# def clean_jd(jd):
#     ...
#     return jd_text
#
# def forward_pass(resume_text, jd_text):
#     ...
#     return score
# -------------------------------------------------


def rank_all_resumes(uploaded_files, jd):
    jd_text = clean_jd(jd)

    rows = []
    for file in uploaded_files:
        resume_text = extract_text(file)
        score = forward_pass(resume_text, jd_text)
        rows.append({
            "filename": file.name,
            "score": float(score)
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
    return df


st.set_page_config(
    page_title="ResRanker V2",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Ultra-polished visual system
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-0: #040711;
        --bg-1: #070b18;
        --bg-2: #0b1224;
        --card: rgba(255, 255, 255, 0.08);
        --card-strong: rgba(255, 255, 255, 0.12);
        --border: rgba(255, 255, 255, 0.14);
        --border-soft: rgba(255, 255, 255, 0.08);
        --text: rgba(255, 255, 255, 0.96);
        --muted: rgba(226, 232, 240, 0.74);
        --muted-2: rgba(226, 232, 240, 0.58);
        --violet: #8b5cf6;
        --blue: #38bdf8;
        --pink: #ec4899;
        --mint: #34d399;
        --gold: #fbbf24;
        --shadow: 0 30px 80px rgba(0, 0, 0, 0.42);
        --shadow-soft: 0 14px 40px rgba(0, 0, 0, 0.26);
        --radius-xl: 34px;
        --radius-lg: 26px;
        --radius-md: 20px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}

    .stApp {
        background:
            radial-gradient(circle at 18% 16%, rgba(139, 92, 246, 0.38), transparent 20%),
            radial-gradient(circle at 82% 14%, rgba(236, 72, 153, 0.30), transparent 18%),
            radial-gradient(circle at 78% 84%, rgba(56, 189, 248, 0.24), transparent 22%),
            radial-gradient(circle at 16% 84%, rgba(52, 211, 153, 0.16), transparent 18%),
            linear-gradient(135deg, var(--bg-0) 0%, var(--bg-1) 36%, var(--bg-2) 100%);
        color: var(--text);
        min-height: 100vh;
        overflow-x: hidden;
    }

    .mesh {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .mesh::before,
    .mesh::after {
        content: "";
        position: absolute;
        inset: -20%;
        background-image:
            linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 92px 92px;
        mask-image: radial-gradient(circle at center, black 28%, transparent 72%);
        opacity: 0.20;
        transform: perspective(900px) rotateX(62deg) translateY(-22%);
    }

    .mesh::after {
        background-size: 44px 44px;
        opacity: 0.08;
        transform: perspective(900px) rotateX(62deg) translateY(-18%);
    }

    .noise {
        position: fixed;
        inset: 0;
        z-index: 1;
        pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='1.18' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='220' height='220' filter='url(%23n)' opacity='0.22'/%3E%3C/svg%3E");
        opacity: 0.065;
        mix-blend-mode: soft-light;
    }

    .orb {
        position: fixed;
        border-radius: 999px;
        filter: blur(70px);
        opacity: 0.62;
        z-index: 0;
        pointer-events: none;
        animation: drift 16s ease-in-out infinite;
    }

    .orb.one {
        width: 340px; height: 340px;
        left: -110px; top: -80px;
        background: rgba(139, 92, 246, 0.92);
    }
    .orb.two {
        width: 360px; height: 360px;
        right: -120px; top: 60px;
        background: rgba(236, 72, 153, 0.86);
        animation-delay: -6s;
    }
    .orb.three {
        width: 420px; height: 420px;
        left: 18%; bottom: -170px;
        background: rgba(56, 189, 248, 0.70);
        animation-delay: -10s;
    }
    .orb.four {
        width: 250px; height: 250px;
        right: 18%; bottom: 16%;
        background: rgba(52, 211, 153, 0.36);
        animation-delay: -3s;
    }

    @keyframes drift {
        0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
        33% { transform: translate3d(14px, -16px, 0) scale(1.05); }
        66% { transform: translate3d(-10px, 14px, 0) scale(0.96); }
    }

    .page {
        position: relative;
        z-index: 2;
        max-width: 1460px;
        margin: 0 auto;
        padding: 22px 22px 38px 22px;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 20px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .mark {
        width: 48px;
        height: 48px;
        border-radius: 16px;
        background:
            linear-gradient(135deg, rgba(139,92,246,1), rgba(236,72,153,0.96));
        box-shadow: 0 18px 50px rgba(139,92,246,0.30);
        display: grid;
        place-items: center;
        position: relative;
        overflow: hidden;
    }

    .mark::before {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: 15px;
        background:
            radial-gradient(circle at 30% 25%, rgba(255,255,255,0.34), transparent 25%),
            linear-gradient(135deg, rgba(255,255,255,0.18), transparent 52%);
        mix-blend-mode: screen;
    }

    .mark span {
        position: relative;
        z-index: 1;
        font-weight: 900;
        color: white;
        font-size: 18px;
        letter-spacing: -0.05em;
    }

    .brand-copy {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .brand-copy .name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        letter-spacing: -0.03em;
    }

    .brand-copy .tag {
        font-size: 13px;
        color: var(--muted-2);
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: flex-end;
    }

    .chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        color: rgba(248,250,252,0.90);
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.01em;
        backdrop-filter: blur(16px);
        box-shadow: var(--shadow-soft);
    }

    .hero {
        display: grid;
        grid-template-columns: 1.45fr 0.85fr;
        gap: 20px;
        align-items: stretch;
        margin-top: 16px;
    }

    .glass {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        box-shadow: var(--shadow);
        border-radius: var(--radius-xl);
        position: relative;
        overflow: hidden;
    }

    .glass::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.10), transparent 40%),
            radial-gradient(circle at top left, rgba(255,255,255,0.10), transparent 24%);
        pointer-events: none;
    }

    .hero-left {
        padding: 34px 34px 28px 34px;
        min-height: 440px;
    }

    .kicker {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 9px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        color: rgba(255,255,255,0.88);
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 28px rgba(0,0,0,0.18);
    }

    .title {
        margin: 18px 0 12px 0;
        font-size: clamp(50px, 7vw, 76px);
        line-height: 0.95;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.06em;
        color: white;
        text-wrap: balance;
    }

    .title .grad {
        background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 18%, #60a5fa 48%, #f472b6 78%, #f8fafc 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 10px 34px rgba(96,165,250,0.10);
    }

    .subtitle {
        max-width: 760px;
        font-size: 17px;
        line-height: 1.78;
        color: var(--muted);
        margin-bottom: 22px;
    }

    .cta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 18px;
    }

    .cta-pill {
        padding: 12px 18px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: white;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.20);
    }

    .cta-pill.primary {
        background: linear-gradient(135deg, rgba(124,58,237,0.98), rgba(236,72,153,0.98));
        border: none;
        box-shadow: 0 18px 45px rgba(124,58,237,0.30);
    }

    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 28px;
    }

    .stat {
        border-radius: 24px;
        padding: 16px 18px 15px 18px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
    }

    .stat .value {
        font-size: 28px;
        font-weight: 900;
        color: white;
        letter-spacing: -0.04em;
        margin-bottom: 4px;
    }

    .stat .label {
        font-size: 13px;
        color: var(--muted-2);
        line-height: 1.4;
    }

    .hero-right {
        padding: 20px;
        display: grid;
        gap: 14px;
    }

    .side-card {
        border-radius: 26px;
        padding: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.085), rgba(255,255,255,0.05));
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 16px 34px rgba(0,0,0,0.18);
    }

    .side-card .h {
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 8px;
        color: white;
        letter-spacing: -0.01em;
    }

    .side-card .p {
        font-size: 14px;
        line-height: 1.7;
        color: var(--muted);
    }

    .section-grid {
        display: grid;
        grid-template-columns: 1.05fr 0.95fr;
        gap: 18px;
        margin-top: 20px;
    }

    .card {
        padding: 24px;
        border-radius: 28px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: var(--shadow-soft);
    }

    .card-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 12px;
    }

    .card-title h3 {
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        letter-spacing: -0.03em;
        color: white;
    }

    .mini-badge {
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 12px;
        font-weight: 700;
        color: rgba(255,255,255,0.82);
    }

    .card-subtitle {
        margin: 0 0 18px 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
    }

    .upload-wrap {
        border-radius: 24px;
        padding: 18px;
        background:
            linear-gradient(135deg, rgba(139,92,246,0.16), rgba(236,72,153,0.10), rgba(56,189,248,0.10));
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }

    section[data-testid="stFileUploaderDropzone"] {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.14), transparent 38%),
            rgba(255,255,255,0.05) !important;
        border: 1px dashed rgba(255,255,255,0.20) !important;
        border-radius: 22px !important;
        padding: 18px !important;
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {
        transform: translateY(-1px);
        border-color: rgba(255,255,255,0.34) !important;
        background: rgba(255,255,255,0.065) !important;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: rgba(255,255,255,0.95) !important;
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 22px !important;
        padding: 14px 16px !important;
        min-height: 200px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    .stTextArea label {
        color: rgba(255,255,255,0.92) !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.01em;
        margin-bottom: 8px !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 18px;
        padding: 0.95rem 1rem;
        color: white;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.01em;
        background:
            linear-gradient(135deg, #7c3aed 0%, #2563eb 44%, #06b6d4 74%, #ec4899 100%);
        box-shadow:
            0 18px 44px rgba(37, 99, 235, 0.26),
            0 12px 30px rgba(139, 92, 246, 0.20);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 20%, rgba(255,255,255,0.18), transparent 40%);
        transform: translateX(-120%);
        transition: transform 0.55s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        filter: brightness(1.05);
        box-shadow:
            0 22px 52px rgba(37, 99, 235, 0.32),
            0 16px 34px rgba(236, 72, 153, 0.18);
    }

    .stButton > button:hover::before {
        transform: translateX(120%);
    }

    div[data-testid="stMetric"] {
        background:
            linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 22px;
        padding: 18px 18px 12px 18px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.20);
        backdrop-filter: blur(18px);
    }

    div[data-testid="stMetricLabel"] {
        color: rgba(226,232,240,0.72) !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }

    div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 900 !important;
        font-size: 30px !important;
        letter-spacing: -0.04em;
    }

    div[data-testid="stMetricDelta"] {
        color: #bbf7d0 !important;
    }

    details {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 18px !important;
        padding: 10px 14px !important;
    }

    details summary {
        color: white !important;
        font-weight: 700 !important;
    }

    .results-shell {
        display: grid;
        gap: 12px;
    }

    .result-card {
        padding: 18px;
        border-radius: 22px;
        background: rgba(255,255,255,0.065);
        border: 1px solid rgba(255,255,255,0.10);
    }

    .result-name {
        font-size: 16px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }

    .result-line {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        font-size: 13px;
        color: var(--muted);
        margin-top: 8px;
    }

    .bar {
        width: 100%;
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 10px;
    }

    .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #7c3aed, #38bdf8, #ec4899);
        box-shadow: 0 0 22px rgba(56,189,248,0.24);
    }

    .footer-note {
        text-align: center;
        margin-top: 22px;
        color: rgba(226,232,240,0.50);
        font-size: 12px;
    }

    @media (max-width: 1100px) {
        .hero,
        .section-grid {
            grid-template-columns: 1fr;
        }
        .topbar {
            flex-direction: column;
            align-items: flex-start;
        }
        .chip-row {
            justify-content: flex-start;
        }
        .hero-stats {
            grid-template-columns: 1fr;
        }
        .title {
            font-size: 46px;
        }
    }

    @media (max-width: 720px) {
        .page {
            padding: 14px;
        }
        .hero-left,
        .card,
        .hero-right {
            padding-left: 18px;
            padding-right: 18px;
        }
        .title {
            font-size: 38px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="mesh"></div>
    <div class="noise"></div>
    <div class="orb one"></div>
    <div class="orb two"></div>
    <div class="orb three"></div>
    <div class="orb four"></div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Page layout
# -----------------------------
st.markdown("<div class='page'>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="mark"><span>✦</span></div>
            <div class="brand-copy">
                <div class="name">RezRanker V2</div>
                <div class="tag">Created By Akshit Raj</div>
            </div>
        </div>
        <div class="chip-row">
            <div class="chip">Advanced V2</div>
            <div class="chip">Multi-file upload</div>
            <div class="chip">Displays Results</div>
            <div class="chip">Docker deployed</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="glass hero-left">
            <div class="kicker">✧ Designed as a multi-phase NLP pipeline</div>
            <div class="title">From raw resumes to <span class="grad">Ranked Semantic matches</span>.</div>
            <div class="subtitle">
            </div>
            <div class="cta-row">
                <div class="cta-pill primary">Upload → Extract → Rank </div>
                <div class="cta-pill">Uses BERT</div>
                <div class="cta-pill">Vector Embeddings </div>
                <div class="cta-pill">Section Classification</div>
            </div>
            <div class="hero-stats">
                <div class="stat">
                    <div class="value">1</div>
                    <div class="label">Job Description</div>
                </div>
                <div class="stat">
                    <div class="value">N</div>
                    <div class="label">resumes ranked in one pass</div>
                </div>
                <div class="stat">
                    <div class="value">∞</div>
                    <div class="label">room to scale later</div>
                </div>
            </div>
        </div>
        <div class="glass hero-right">
            <div class="side-card">
                <div class="h">Phase 1- Text Extraction Layer</div>
                <div class="p">Uses PyMuPDF and Pytesseract to handle both regular pdfs and pdfs that are scanned images</div>
            </div>
            <div class="side-card">
                <div class="h">Phase 2- Text Preprocessing And Chunking</div>
                <div class="p">Applies regex-based cleaning to normalize and purify extracted resume text, segments content into structured chunks using semantic separators, and filters noise while preserving Specific high-signal keywords. </div>
            </div>
            <div class="side-card">
                <div class="h">Phase 3- Section Classification</div>
                <div class="p">Converts resume text into embeddings and matches it against smart keyword anchors to classify sections like skills, experience, and projects also combines rule-based checks with Cosine similarity scoring to keep it clean and accurate ex</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-grid'>", unsafe_allow_html=True)
left, right = st.columns([1.02, 0.98])

with left:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                <h3>Upload resumes</h3>
                <div class="mini-badge">PDF only</div>
            </div>
            <p class="card-subtitle">
                Add one or many PDFs, more pdf = more time
            </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='upload-wrap'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    job_desc = st.text_area(
        "More Detailed The Job Description Better the match",
        placeholder="Paste the job description here...",
        height=220,
        label_visibility="visible",
    )

    run = st.button("Run ranking pipeline")

    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if not uploaded_files:
            st.warning("Upload at least one PDF.")
        elif not job_desc:
            st.warning("Paste a job description.")
        else:
            with st.spinner("Extracting text and ranking resumes..."):
                results_df = rank_all_resumes(uploaded_files, job_desc)
                st.session_state["results_df"] = results_df

            if results_df is not None and not results_df.empty:
                top_row = results_df.iloc[0]
                st.success("Ranking complete!")

                left_metric, mid_metric, right_metric = st.columns(3)
                left_metric.metric("Top score", f"{top_row['score']:.3f}")
                mid_metric.metric("Resumes processed", len(results_df))
                right_metric.metric("Best file", top_row["filename"])

                csv_data = results_df.to_csv(index=True)
                st.download_button(
                    label="Download Rankings as CSV",
                    data=csv_data,
                    file_name="resume_rankings.csv",
                    mime="text/csv"
                )

with right:
    results_df = st.session_state.get("results_df", None)

    cards = []
    if results_df is not None and not results_df.empty:
        for _, row in results_df.iterrows():
            score = float(row["score"])
            pct = max(0, min(100, score * 100))

            cards.append(f"""
<div class="result-card">
    <div class="result-name">{row["filename"]}</div>
    <div class="bar"><div class="fill" style="width: {pct:.0f}%;"></div></div>
    <div class="result-line"><span>Score</span><strong style="color:white;">{score:.3f}</strong></div>
</div>
""".strip())

        results_html = "\n".join(cards)
    else:
        results_html = """
<div class="result-card">
    <div class="result-name">No ranked resumes yet</div>
    <div class="result-line"><span>Status</span><span>Upload files and run the pipeline</span></div>
</div>
""".strip()

    st.markdown(
        f"""
<div class="card">
    <div class="card-title">
        <h3>Ranked results</h3>
        <div class="mini-badge">Live preview</div>
    </div>
    <p class="card-subtitle">
        Results with there scores.
    </p>
    <div class="results-shell">
        {results_html}
    </div>
</div>
""".strip(),
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-note">
        Connect with me → 
        <a href="https://www.linkedin.com/in/rezraj/" target="_blank">
            LinkedIn
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
