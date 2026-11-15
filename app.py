"""
app.py
------
Streamlit dashboard for the AI-Based Phishing Website Detection System.

Run:
    streamlit run app.py

If the trained model is not found, the app will offer to train it
on first launch.
"""

import os
import sys
import json

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Phishing Website Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
ROOT_MODEL  = os.path.join(BASE_DIR, "model.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

sys.path.insert(0, BASE_DIR)
from feature_extraction import extract_features, extract_feature_vector, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Custom CSS — cybersecurity theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- background ---- */
    .stApp { background-color: #0d1117; color: #c9d1d9; }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ---- metric cards ---- */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px;
    }

    /* ---- result boxes ---- */
    .result-safe {
        background-color: #0d3321;
        border: 2px solid #2ea043;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .result-phishing {
        background-color: #3d1515;
        border: 2px solid #da3633;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }

    /* ---- tip box ---- */
    .tip-box {
        background-color: #161b22;
        border-left: 4px solid #388bfd;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }

    /* ---- section header ---- */
    .section-header {
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 4px;
        margin-top: 20px;
    }

    /* ---- button ---- */
    div.stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 15px;
        padding: 10px 24px;
    }
    div.stButton > button:hover {
        background-color: #2ea043;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model from disk. Returns (model, metadata, feature_names) or None."""
    for path in (MODEL_PATH, ROOT_MODEL):
        if os.path.exists(path):
            import joblib
            bundle = joblib.load(path)
            return bundle["model"], bundle.get("metadata", {}), bundle.get("feature_names", FEATURE_NAMES)
    return None, {}, FEATURE_NAMES


model, metadata, trained_features = load_model()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Phishing Detector")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool uses Machine Learning to classify URLs as "
        "**Legitimate** or **Phishing** based on structural URL features."
    )
    st.markdown("---")
    st.markdown("### Model Info")
    if metadata:
        st.write(f"**Algorithm:** {metadata.get('model_name', 'Unknown')}")
        st.write(f"**Accuracy:**  {metadata.get('accuracy', 0):.2%}")
        st.write(f"**F1-Score:**  {metadata.get('f1', 0):.2%}")
        st.write(f"**AUC-ROC:**   {metadata.get('auc', 0):.2%}")
        st.write(f"**Train Size:** {metadata.get('n_train', '?'):,}")
    else:
        st.info("Model not yet trained. Run `python train_model.py` first.")

    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(
        "Uses the [UCI Phishing Websites Dataset]"
        "(https://archive.ics.uci.edu/dataset/327/phishing+websites) "
        "or a synthetic fallback when the UCI file is not available.",
        unsafe_allow_html=False,
    )
    st.markdown("---")
    st.markdown("**IBM SkillsBuild Cyber Security BootCamp 2026**")
    st.markdown("*Developed by Rampravesh*")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#58a6ff;text-align:center;'>🛡️ AI-Based Phishing Website Detection</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#8b949e;'>"
    "Machine Learning powered URL analysis for Cyber Security awareness</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_detect, tab_performance, tab_tips, tab_about = st.tabs(
    ["🔍 Detect URL", "📊 Model Performance", "🔒 Safety Tips", "ℹ️ About"]
)

# ==========================================================================
# TAB 1 — Detect URL
# ==========================================================================
with tab_detect:
    st.markdown("<h3 class='section-header'>Enter a URL to Analyse</h3>", unsafe_allow_html=True)

    url_input = st.text_input(
        label="Website URL",
        placeholder="e.g. https://www.example.com or http://suspicious-site.tk/login",
        help="Enter any full URL. The system analyses its structural features — no live web request is made.",
    )

    col_btn, col_ex = st.columns([2, 5])
    with col_btn:
        analyse_clicked = st.button("🔍 Analyse URL", use_container_width=True)
    with col_ex:
        st.markdown(
            "<small style='color:#8b949e;'>Examples — "
            "<code>https://www.google.com</code>  |  "
            "<code>http://paypal-secure-login.tk/verify?user=123</code>"
            "</small>",
            unsafe_allow_html=True,
        )

    if analyse_clicked:
        if not url_input.strip():
            st.warning("⚠️ Please enter a URL before clicking Analyse.")
        elif model is None:
            st.error(
                "❌ No trained model found. "
                "Please run `python train_model.py` from the project folder first."
            )
        else:
            url = url_input.strip()
            features_dict = extract_features(url)
            feature_vector = [features_dict[f] for f in trained_features if f in features_dict]

            # Pad / fill missing features with 0
            while len(feature_vector) < len(trained_features):
                feature_vector.append(0)
            feature_vector = feature_vector[:len(trained_features)]

            X = np.array([feature_vector])
            prediction = model.predict(X)[0]  # 1 = legit, 0 = phishing

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = proba[1] if prediction == 1 else proba[0]

            # -- Result display --
            st.markdown("---")
            if prediction == 1:
                st.markdown(
                    f"<div class='result-safe'>"
                    f"<h2 style='color:#2ea043;'>✅ LEGITIMATE / SAFE</h2>"
                    f"<p>The URL appears to be <b>legitimate</b> based on its structural features.</p>"
                    f"{'<p>Model confidence: <b>' + f'{confidence:.1%}' + '</b></p>' if confidence is not None else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='result-phishing'>"
                    f"<h2 style='color:#da3633;'>🚨 PHISHING / MALICIOUS</h2>"
                    f"<p>The URL shows characteristics common in <b>phishing websites</b>.</p>"
                    f"{'<p>Model confidence: <b>' + f'{confidence:.1%}' + '</b></p>' if confidence is not None else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Confidence note
            st.caption(
                "⚠️ **Confidence limitation:** This score reflects model certainty about URL "
                "structure only — it is NOT a guarantee of website safety. Always verify "
                "important websites through official channels."
            )

            # -- Feature breakdown --
            st.markdown("<h3 class='section-header'>Feature Analysis</h3>", unsafe_allow_html=True)
            feat_df = pd.DataFrame(
                [(k, v) for k, v in features_dict.items()],
                columns=["Feature", "Value"],
            )

            # Traffic-light colouring hints
            risk_flags = {
                "has_ip": (1, "IP address used as hostname — high risk"),
                "has_suspicious_tld": (1, "Suspicious top-level domain"),
                "has_phishing_keyword": (1, "Common phishing keyword detected"),
                "num_at": (1, "@ in URL — often used to deceive"),
                "double_slash_in_path": (1, "Double slash in path — redirection trick"),
                "has_https": (0, "No HTTPS — connection is not encrypted"),
            }

            def _risk_note(row):
                feat, val = row["Feature"], row["Value"]
                if feat in risk_flags:
                    trigger_val, note = risk_flags[feat]
                    if val == trigger_val:
                        return f"⚠️ {note}"
                return "✅ OK"

            feat_df["Risk Note"] = feat_df.apply(_risk_note, axis=1)
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # -- Bar chart of numeric features --
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")
            numeric_feats = {k: v for k, v in features_dict.items()
                             if k not in ("has_https","has_ip","has_suspicious_tld",
                                          "has_phishing_keyword","double_slash_in_path")}
            bars = ax.bar(
                list(numeric_feats.keys()),
                list(numeric_feats.values()),
                color="#388bfd",
            )
            ax.set_xticklabels(list(numeric_feats.keys()), rotation=45, ha="right",
                               color="#c9d1d9", fontsize=8)
            ax.tick_params(colors="#c9d1d9")
            ax.set_title("URL Numeric Features", color="#58a6ff")
            ax.spines["bottom"].set_color("#30363d")
            ax.spines["left"].set_color("#30363d")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ==========================================================================
# TAB 2 — Model Performance
# ==========================================================================
with tab_performance:
    st.markdown("<h3 class='section-header'>Model Performance Charts</h3>", unsafe_allow_html=True)

    img_map = {
        "Model Comparison":    "model_comparison.png",
        "Confusion Matrix":    "confusion_matrix.png",
        "ROC Curves":          "roc_curve.png",
        "Feature Importances": "feature_importance.png",
    }

    report_path = os.path.join(REPORTS_DIR, "evaluation.txt")
    if metadata:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{metadata.get('accuracy', 0):.2%}")
        c2.metric("Precision", f"{metadata.get('precision', 0):.2%}")
        c3.metric("Recall",    f"{metadata.get('recall', 0):.2%}")
        c4.metric("F1-Score",  f"{metadata.get('f1', 0):.2%}")
        st.markdown("")

    any_found = False
    for title, fname in img_map.items():
        full_path = os.path.join(REPORTS_DIR, fname)
        if os.path.exists(full_path):
            any_found = True
            st.markdown(f"#### {title}")
            st.image(full_path, use_container_width=True)

    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report_text = fh.read()
        with st.expander("📄 Full Evaluation Report"):
            st.code(report_text, language="text")

    if not any_found:
        st.info(
            "No performance charts found yet. "
            "Run `python train_model.py` from the project folder to generate them."
        )

# ==========================================================================
# TAB 3 — Safety Tips
# ==========================================================================
with tab_tips:
    st.markdown("<h3 class='section-header'>🔒 Cybersecurity Awareness Tips</h3>", unsafe_allow_html=True)

    tips = [
        ("Check the URL carefully",
         "Look for misspellings, extra hyphens, or strange subdomains "
         "(e.g. paypal-secure-login.com instead of paypal.com)."),
        ("Verify HTTPS",
         "Legitimate sites use HTTPS. A padlock icon doesn't guarantee safety, "
         "but its absence is a red flag — especially on login or payment pages."),
        ("Never click suspicious links",
         "If you receive an unexpected email or message with a link, "
         "navigate directly to the official website instead of clicking."),
        ("Watch for urgency tactics",
         "Phishing messages often create a sense of urgency: "
         "'Your account will be suspended!' Slow down and verify."),
        ("Check the sender's email domain",
         "Emails from support@paypa1.com or no-reply@amazon-service.tk are red flags."),
        ("Use a password manager",
         "Password managers fill credentials only on the correct domain, "
         "which protects you even if you accidentally visit a phishing clone."),
        ("Enable Two-Factor Authentication (2FA)",
         "Even if your password is stolen, 2FA prevents unauthorized login."),
        ("Keep software updated",
         "Browser security updates patch known phishing and malware vectors."),
        ("Use browser security extensions",
         "Tools like uBlock Origin and Google Safe Browsing provide additional protection."),
        ("Report phishing sites",
         "Report phishing to Google (safebrowsing.google.com/safebrowsing/report_phish/) "
         "or your national CERT."),
    ]

    for i, (title, desc) in enumerate(tips, 1):
        st.markdown(
            f"<div class='tip-box'>"
            f"<b style='color:#58a6ff;'>{i}. {title}</b><br>"
            f"<span style='color:#c9d1d9;'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ==========================================================================
# TAB 4 — About
# ==========================================================================
with tab_about:
    st.markdown("<h3 class='section-header'>About This Project</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        **AI-Based Phishing Website Detection System Using Machine Learning**

        Developed as part of the **IBM SkillsBuild Cyber Security BootCamp 2026**
        conducted by **BharatCares**.

        ---

        ### How it Works
        1. The user enters a URL.
        2. The system extracts **16 structural URL features** (length, dots, hyphens,
           HTTPS usage, IP address presence, suspicious TLD, phishing keywords, etc.).
        3. A trained **Machine Learning classifier** (best of Logistic Regression,
           Decision Tree, and Random Forest) predicts whether the URL is phishing or
           legitimate.
        4. The result, confidence score, and feature breakdown are displayed.

        ### Important Limitations
        - This system analyses **URL structure only** — it does not visit or render
          the website.
        - A URL that *looks* legitimate can still host phishing content, and vice versa.
        - The confidence score reflects model certainty, **not website safety**.
        - Always use multiple layers of security when browsing sensitive sites.

        ### Dataset
        - **Primary:** UCI Phishing Websites Dataset
          (https://archive.ics.uci.edu/dataset/327/phishing+websites)
          — 11,055 instances, 30 features, used under CC BY 4.0 licence.
        - **Fallback:** Synthetic dataset based on published URL-pattern statistics
          (used when the UCI file is not placed in the `dataset/` folder).

        ### Author
        **Rampravesh**
        IBM SkillsBuild Cyber Security BootCamp 2026
        """,
        unsafe_allow_html=False,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8b949e;font-size:12px;'>"
    "AI-Based Phishing Website Detection System | "
    "IBM SkillsBuild Cyber Security BootCamp 2026 | "
    "Developed by Rampravesh"
    "</p>",
    unsafe_allow_html=True,
)
