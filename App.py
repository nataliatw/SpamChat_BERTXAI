import streamlit as st
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import os
import re

st.set_page_config(layout="wide")

# ===================================================
# Load model dari Hugging Face Hub
# ===================================================
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    pipe = pipeline(
        "text-classification",
        model="nataliatw/indobert-spam-detector",   
        tokenizer="nataliatw/indobert-spam-detector",
        top_k= None
    )
    return pipe

pipe = load_model()

# ===================================================
# LIME preparation
# ===================================================
class_names = ["NON-SPAM", "SPAM"]
explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(texts):
    outputs = pipe(texts)

    probs = []

    for out in outputs:

        scores = {x["label"]: x["score"] for x in out}

        probs.append([
            scores["LABEL_0"],
            scores["LABEL_1"]
        ])

    return np.array(probs)


# ============================
# UI TITLE (selalu tampil)
# ============================
st.title("📡 Spam Detection IndoBERT + LIME XAI")
st.write("Masukkan pesan, lalu lihat hasil prediksi dan penjelasan model.")

# ============================
# 3 KOLOM
# ============================
col1, col2, col3 = st.columns([1.1, 1.7, 1.2])


# ============================
# KOLOM 1 — INPUT
# ============================
with col1:
    st.markdown("<h3>✏️ Input Pesan</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#ffffff; padding:15px; border-radius:12px;
                box-shadow:0 0 8px rgba(0,0,0,0.08);">
    """, unsafe_allow_html=True)

    text = st.text_area("Tulis pesan di sini:", height=250, label_visibility="collapsed")
    run = st.button("🔍 Analisis", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================
# KOLOM 2 — PREDIKSI & HIGHLIGHT
# ============================
with col2:
    st.markdown("<h3>📌 Hasil Prediksi</h3>", unsafe_allow_html=True)

    if run:

        raw = pipe([text])[0]

        scores = {item["label"]: item["score"] for item in raw}

        non_spam_score = scores["LABEL_0"]
        spam_score = scores["LABEL_1"]

        final_label = "SPAM" if spam_score > non_spam_score else "NON-SPAM"
        final_score = max(spam_score, non_spam_score)

        st.markdown(
            f"""
            <div style="
                background:white;
                padding:20px;
                border-radius:12px;
                box-shadow:0 3px 10px rgba(0,0,0,0.08);
                border-left:6px solid {'#d9534f' if final_label=='SPAM' else '#5cb85c'};
            ">
                <b>Prediksi:</b> {final_label}<br>
                <b>Confidence:</b> {final_score:.4f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ========== LIME Highlight ==========
        st.markdown("<h3>🧠 Penjelasan LIME</h3>", unsafe_allow_html=True)

        exp = explainer.explain_instance(text, predict_proba, num_features=10)
        lime_scores = exp.as_list()
        
        df = pd.DataFrame(lime_scores, columns=["word", "score"])

        scores = df["score"].values
        vmin = np.quantile(scores, 0.2)
        vmax = np.quantile(scores, 0.8)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = matplotlib.colormaps["RdYlGn"]

        def highlight_text(text, lime_scores):
            score_map = {w.lower(): s for w, s in lime_scores}
            html = []

            tokens = re.findall(r"\b\w+|\W+", text)

            for t in tokens:
                key = re.sub(r"\W+", "", t.lower())

                if key in score_map:
                    score = score_map[key]
                    rgba = cmap(norm(score))
                    r, g, b, _ = rgba
                    color = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.3)"

                    html.append(
                        f"<span style='background:{color}; padding:4px; border-radius:5px;'>{t}</span>"
                    )
                else:
                    html.append(t)
            return "".join(html)

        st.markdown(
            f"""
            <div style="background:white; padding:15px; border-radius:10px;
                        box-shadow:0 3px 10px rgba(0,0,0,0.08); font-size:17px;">
            {highlight_text(text, lime_scores)}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================
# KOLOM 3 — SCORE TABEL
# ============================
with col3:
    if run:
        st.markdown("<h3>📊 Skor Kata</h3>", unsafe_allow_html=True)

        df = pd.DataFrame(lime_scores, columns=["Kata", "Contribution"])
        df = df.sort_values("Contribution", ascending=False)

        styled_df = df.style.background_gradient(
            cmap="RdYlGn",
            low=0.2, high=0.8
        )

        st.markdown("""
        <div style="background:white; padding:15px; border-radius:12px;
                    box-shadow:0 3px 10px rgba(0,0,0,0.08);">
        """, unsafe_allow_html=True)

        st.dataframe(styled_df, hide_index=True, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
