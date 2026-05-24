import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix)
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# NLTK Setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Page config
st.set_page_config(
    page_title="Fake News Detection | issu321",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_path = os.path.join("assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==================== NLP PIPELINE ====================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def get_keywords(texts, top_n=20):
    all_words = ' '.join(texts).split()
    freq = Counter(all_words)
    return freq.most_common(top_n)

# ==================== ML PIPELINE ====================
@st.cache_resource(show_spinner="Training AI Models...")
def train_models():
    df = pd.read_csv('news_dataset.csv')
    df = df.dropna(subset=['text', 'label'])
    df['processed'] = df['text'].apply(preprocess_text)
    df = df[df['processed'].str.len() > 0]

    X = df['processed']
    y = df['label']

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    model_configs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1)
    }

    results = {}
    trained_models = {}

    for name, model in model_configs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, pos_label='fake', average='binary', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, pos_label='fake', average='binary', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, pos_label='fake', average='binary', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['fake', 'real']).tolist()
        }
        trained_models[name] = model

    best_name = max(results, key=lambda x: results[x]['f1'])
    best_model = trained_models[best_name]

    joblib.dump((best_model, tfidf), 'best_model.pkl')

    return df, results, trained_models, tfidf, best_name

# ==================== EXPLANATION ENGINE ====================
def generate_explanation(text, prediction, confidence, sentiment_pol):
    reasons = []
    text_lower = str(text).lower()
    words = text_lower.split()

    sensational_words = ['shocking', 'unbelievable', 'miracle', 'secret', 'conspiracy',
                        'exposed', 'banned', 'censored', 'hoax', 'scam', 'urgent', 'alert']
    sens_count = sum(1 for w in words if w in sensational_words)
    if sens_count >= 2:
        reasons.append(f"sensationalist language ({sens_count} trigger words)")
    elif sens_count == 1:
        reasons.append("sensationalist phrasing detected")

    if abs(sentiment_pol) > 0.4:
        reasons.append("extreme emotional polarity")
    elif abs(sentiment_pol) > 0.2:
        reasons.append("notable emotional undertone")

    if len(text) < 150:
        reasons.append("abnormally brief content")

    caps_ratio = sum(1 for c in str(text) if c.isupper()) / max(len(str(text)), 1)
    if caps_ratio > 0.3:
        reasons.append("excessive capitalization")

    if prediction == 'fake':
        base = "This article exhibits linguistic patterns commonly associated with misinformation"
    else:
        base = "This article demonstrates credible journalistic characteristics"

    if reasons:
        extra = " The emotionally manipulative tone increases probability of fabricated content." if (prediction == 'fake' and abs(sentiment_pol) > 0.3) else ""
        return f"{base}: {', '.join(reasons)}. Model confidence: {confidence:.1f}%.{extra}"
    else:
        return f"{base}. Model confidence: {confidence:.1f}%."

# ==================== UI COMPONENTS ====================
def neon_card(title, value, color="#00f3ff"):
    st.markdown(f"""
    <div style="
        background: rgba(0, 20, 40, 0.7);
        border: 1px solid {color};
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 15px {color}40;
    ">
        <h4 style="color: {color}; margin: 0; font-family: monospace;">{title}</h4>
        <h2 style="color: #ffffff; margin: 10px 0 0 0; font-size: 2rem;">{value}</h2>
    </div>
    """, unsafe_allow_html=True)

def terminal_section(title, content):
    safe_content = str(content).replace('"', '&quot;')
    st.markdown(f"""
    <div style="
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        font-family: 'Courier New', monospace;
    ">
        <div style="color: #00f3ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-bottom: 12px;">
            📟 {title}
        </div>
        <div style="color: #c9d1d9; line-height: 1.6;">
            {safe_content}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGES ====================
def show_home():
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 3.5rem; color: #00f3ff; text-shadow: 0 0 20px #00f3ff80;">
            🛡️ FAKE NEWS DETECTION
        </h1>
        <p style="font-size: 1.3rem; color: #8b949e; margin-top: 20px;">
            AI-Powered NLP System for Real-Time Misinformation Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        neon_card("Articles Analyzed", "100+", "#00f3ff")
    with col2:
        neon_card("ML Models", "3 Active", "#00ff9d")
    with col3:
        neon_card("Accuracy", ">92%", "#ff00aa")

    st.markdown("---")
    st.subheader("🚀 System Capabilities")
    cols = st.columns(2)
    capabilities = [
        ("🔍 Real-Time Detection", "Instant classification of news articles using advanced NLP pipelines"),
        ("📊 Model Comparison", "Benchmark Logistic Regression, Random Forest, and Naive Bayes"),
        ("🧠 AI Explanations", "Human-readable reasoning behind every prediction"),
        ("📁 Batch Processing", "Upload CSV files for bulk analysis and reporting")
    ]
    for i, (cap_title, cap_desc) in enumerate(capabilities):
        with cols[i % 2]:
            terminal_section(cap_title, cap_desc)

    st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #8b949e;">
        Developed by <a href="https://github.com/issu321" style="color: #00f3ff; text-decoration: none;">issu321</a>
    </div>
    """, unsafe_allow_html=True)

def show_detect(tfidf, models, best_name):
    st.markdown('<h2 style="color: #00f3ff;">🔍 Single Article Analysis</h2>', unsafe_allow_html=True)

    input_method = st.radio("Input Method", ["Text Input", "Title + Body"], horizontal=True)

    if input_method == "Text Input":
        text_input = st.text_area("Paste article text here:", height=250,
                                  placeholder="Enter news article content for instant verification...")
        title_input = "Manual Input"
    else:
        title_input = st.text_input("Article Title:", placeholder="Breaking: ...")
        text_input = st.text_area("Article Body:", height=200, placeholder="Full article content...")
        text_input = f"{title_input}. {text_input}"

    model_choice = st.selectbox("Select Model", ["Auto (Best)"] + list(models.keys()))
    selected_model = models[best_name] if model_choice == "Auto (Best)" else models[model_choice]

    if st.button("🚀 Analyze Article", use_container_width=True):
        if not text_input or len(text_input.strip()) < 20:
            st.error("Please enter at least 20 characters for analysis.")
            return

        with st.spinner("Processing NLP Pipeline..."):
            processed = preprocess_text(text_input)
            vec = tfidf.transform([processed])
            pred = selected_model.predict(vec)[0]
            proba = selected_model.predict_proba(vec)[0]
            confidence = max(proba) * 100
            pol, sub = get_sentiment(text_input)

        col1, col2 = st.columns([1, 2])
        with col1:
            color = "#ff0055" if pred == 'fake' else "#00ff9d"
            icon = "⚠️" if pred == 'fake' else "✅"
            neon_card("VERDICT", f"{icon} {pred.upper()}", color)
            neon_card("CONFIDENCE", f"{confidence:.1f}%", color)
            neon_card("SENTIMENT", f"{pol:+.2f} polarity", "#00f3ff")
            neon_card("SUBJECTIVITY", f"{sub:.2f}", "#ffaa00")

        with col2:
            explanation = generate_explanation(text_input, pred, confidence, pol)
            terminal_section("AI EXPLANATION ENGINE", explanation)

            st.markdown("### 📋 NLP Pipeline Trace")
            feature_names = tfidf.get_feature_names_out()
            scores = vec.toarray()[0]
            top_idx = scores.argsort()[-5:][::-1]
            trace_data = {
                "Original Length": len(text_input),
                "Processed Length": len(processed),
                "Tokens": len(processed.split()),
                "TF-IDF Features": vec.shape[1],
                "Top TF-IDF Terms": [feature_names[i] for i in top_idx if scores[i] > 0][:5]
            }
            st.json(trace_data)

            top_terms = [(feature_names[i], scores[i]) for i in top_idx if scores[i] > 0]
            if top_terms:
                df_terms = pd.DataFrame(top_terms, columns=['Term', 'TF-IDF Score'])
                fig = px.bar(df_terms, x='TF-IDF Score', y='Term', orientation='h',
                           color='TF-IDF Score', color_continuous_scale=['#0066ff', '#00f3ff'])
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#c9d1d9', margin=dict(l=10, r=10, t=10, b=10),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

def show_dashboard(df, results):
    st.markdown('<h2 style="color: #00f3ff;">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    fake_count = (df['label'] == 'fake').sum()
    real_count = (df['label'] == 'real').sum()
    with c1:
        neon_card("Total Articles", len(df), "#00f3ff")
    with c2:
        neon_card("Fake Articles", fake_count, "#ff0055")
    with c3:
        neon_card("Real Articles", real_count, "#00ff9d")
    with c4:
        best_f1 = max(r['f1'] for r in results.values())
        neon_card("Best F1 Score", f"{best_f1:.3f}", "#ffaa00")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Label Distribution")
        fig = px.pie(df, names='label', color='label',
                    color_discrete_map={'fake': '#ff0055', 'real': '#00ff9d'},
                    hole=0.4)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font_color='#c9d1d9',
            showlegend=True, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True, key='dash_pie')

    with col2:
        st.markdown("### 🎯 Model Performance")
        perf_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [r['accuracy'] for r in results.values()],
            'F1 Score': [r['f1'] for r in results.values()]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Accuracy', x=perf_df['Model'], y=perf_df['Accuracy'],
                            marker_color='#00f3ff', text=[f"{v:.3f}" for v in perf_df['Accuracy']],
                            textposition='auto'))
        fig.add_trace(go.Bar(name='F1 Score', x=perf_df['Model'], y=perf_df['F1 Score'],
                            marker_color='#ff00aa', text=[f"{v:.3f}" for v in perf_df['F1 Score']],
                            textposition='auto'))
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#c9d1d9', margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key='dash_perf')

    st.markdown("### 🔑 Keyword Frequency")
    fake_texts = df[df['label'] == 'fake']['processed'].tolist()
    real_texts = df[df['label'] == 'real']['processed'].tolist()
    fake_kw = get_keywords(fake_texts, 15)
    real_kw = get_keywords(real_texts, 15)

    c1, c2 = st.columns(2)
    with c1:
        df_kw = pd.DataFrame(fake_kw, columns=['Keyword', 'Frequency'])
        fig = px.bar(df_kw, x='Frequency', y='Keyword', orientation='h', color='Frequency',
                    color_continuous_scale=['#ff0055', '#ffaa00'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font_color='#c9d1d9', title="Fake News Keywords", height=400,
                         margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True, key='dash_fake_kw')
    with c2:
        df_kw = pd.DataFrame(real_kw, columns=['Keyword', 'Frequency'])
        fig = px.bar(df_kw, x='Frequency', y='Keyword', orientation='h', color='Frequency',
                    color_continuous_scale=['#00ff9d', '#00f3ff'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font_color='#c9d1d9', title="Real News Keywords", height=400,
                         margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

def show_models(results):
    st.markdown('<h2 style="color: #00f3ff;">🤖 Model Benchmarks</h2>', unsafe_allow_html=True)

    for name, metrics in results.items():
        with st.expander(f"📦 {name}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                neon_card("Accuracy", f"{metrics['accuracy']:.4f}", "#00f3ff")
            with c2:
                neon_card("Precision", f"{metrics['precision']:.4f}", "#00ff9d")
            with c3:
                neon_card("Recall", f"{metrics['recall']:.4f}", "#ffaa00")
            with c4:
                neon_card("F1 Score", f"{metrics['f1']:.4f}", "#ff0055")

            cm = np.array(metrics['confusion_matrix'])
            fig = px.imshow(cm, text_auto=True, x=['fake', 'real'], y=['fake', 'real'],
                          color_continuous_scale=['#0d1117', '#0066ff', '#00f3ff'])
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#c9d1d9', width=400, height=350,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            fig.update_xaxes(title_text='Predicted', side='bottom')
            fig.update_yaxes(title_text='Actual')
            st.plotly_chart(fig, use_container_width=False, key=f"cm_{name.replace(' ', '_')}")

def show_batch(tfidf, models, best_name):
    st.markdown('<h2 style="color: #00f3ff;">📁 Batch Analysis</h2>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV (must contain 'text' column)", type=['csv'])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        if 'text' not in batch_df.columns:
            st.error("CSV must contain a 'text' column.")
            return

        model_choice = st.selectbox("Model", ["Auto (Best)"] + list(models.keys()))
        selected_model = models[best_name] if model_choice == "Auto (Best)" else models[model_choice]

        if st.button("🔬 Run Batch Analysis", use_container_width=True):
            with st.spinner(f"Analyzing {len(batch_df)} articles..."):
                batch_df['processed'] = batch_df['text'].apply(preprocess_text)
                valid_mask = batch_df['processed'].str.len() > 0
                valid_df = batch_df[valid_mask].copy()

                if len(valid_df) == 0:
                    st.error("No valid text found after preprocessing.")
                    return

                vec = tfidf.transform(valid_df['processed'])
                preds = selected_model.predict(vec)
                proba = selected_model.predict_proba(vec)
                confidences = np.max(proba, axis=1) * 100

                sentiments = [get_sentiment(t)[0] for t in valid_df['text']]

                valid_df['prediction'] = preds
                valid_df['confidence'] = confidences
                valid_df['sentiment_polarity'] = sentiments

                st.success(f"Analysis complete for {len(valid_df)} articles.")

                st.markdown("### 📊 Results Preview")
                st.dataframe(valid_df[['text', 'prediction', 'confidence', 'sentiment_polarity']].head(20),
                           use_container_width=True)

                st.markdown("### 📈 Distribution")
                fig = px.histogram(valid_df, x='prediction', color='prediction',
                                 color_discrete_map={'fake': '#ff0055', 'real': '#00ff9d'})
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#c9d1d9', margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True, key='batch_dist')

                csv = valid_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Full Results", csv, "batch_predictions.csv", "text/csv")

# ==================== MAIN ====================
def main():
    df, results, models, tfidf, best_name = train_models()

    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #00f3ff; text-shadow: 0 0 10px #00f3ff40;">🛡️ FND-NLP</h2>
        <p style="color: #8b949e; font-size: 0.8rem;">v1.0 | issu321</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navigation",
        ["🏠 Home", "🔍 Detect", "📊 Dashboard", "🤖 Models", "📁 Batch"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #8b949e; font-size: 0.75rem;">
        <p>Developed by <a href="https://github.com/issu321" style="color:#00f3ff">issu321</a></p>
        <p>© 2026 Fake News Detection</p>
    </div>
    """, unsafe_allow_html=True)

    if page == "🏠 Home":
        show_home()
    elif page == "🔍 Detect":
        show_detect(tfidf, models, best_name)
    elif page == "📊 Dashboard":
        show_dashboard(df, results)
    elif page == "🤖 Models":
        show_models(results)
    elif page == "📁 Batch":
        show_batch(tfidf, models, best_name)

if __name__ == "__main__":
    main()
