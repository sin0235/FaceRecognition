import sys
import os
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
from PIL import Image
from inference.recognition_engine import RecognitionEngine


engine = RecognitionEngine()
TEMP_DIR = tempfile.gettempdir()

st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="✨",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main {
        padding-top: 2rem;
    }
    #celebrity-face-recognition {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #bbbbbb;
        margin-bottom: 2rem;
    }
    .upload-box {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: #111827;
        border: 1px solid #374151;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #111827, #1f2937);
        border: 1px solid #4b5563;
        color: #e5e7eb;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }
    .name-pill {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: #f97316;
        color: white;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .score-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #1f2937;
        color: #a5b4fc;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .hint {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Celebrity Face Recognition")
st.markdown(
    '<div class="subtitle">Upload một bức ảnh khuôn mặt để hệ thống thử đoán người nổi tiếng tương ứng.</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=TEMP_DIR) as tmp:
        temp_path = tmp.name
        img.save(temp_path)

    col_img, col_info = st.columns([3, 2])

    with col_img:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    try:
        result = engine.recognize(temp_path)
        name = result.get('identity', 'Unknown')
        score = result.get('confidence', 0.0)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="name-pill">{name}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="score-pill">score = {score:.4f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="hint">Hiện đang dùng embedding giả để demo, nên đây chỉ là kết quả minh họa.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        "<p style='text-align:center;color:#9ca3af;margin-top:1rem;'>Chưa có ảnh nào được upload.</p>",
        unsafe_allow_html=True,
    )
