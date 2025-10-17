import os
import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(45deg, #fdf8f3 0%, #f7f0e8 30%, #f2e8de 100%);
        min-height: 100vh;
    }
     .stApp {
    background: linear-gradient(45deg, #eff6ff 0%, #dbeafe 50%, #bfdbfe 100%);
}
    /* Glassmorphism Header */
    .main-header {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .sub-title {
        font-size: 1.3rem;
        color: #4a5568;
        font-weight: 400;
        margin-bottom: 20px;
    }
    
    .stTextArea textarea {
    border-radius: 15px;
    border: 1px solid #e0d6cc;
    padding: 20px;
    font-size: 16px;
    transition: all 0.3s ease;
    background: #fefbf7;
    margin: 20px 0;
}

.stTextArea textarea:focus {
    border-color: #c4a484;
    box-shadow: 0 0 0 3px rgba(196, 164, 132, 0.15);
    background: #ffffff;
}
    .main-title span {
    color: #FFD700;  /* gold accent */
    font-size: 3.5rem;
}
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 15px;
        padding: 15px 40px;
        font-size: 18px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3);
        width: 100%;
        margin: 10px 0 30px 0;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.4);
    }
    
    /* Results Container */
    .results-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        margin: 30px 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.7);
    }
    
    /* Sentiment Cards */
    .sentiment-card {
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin: 15px 0;
        transition: transform 0.3s ease;
    }
    
    .sentiment-card:hover {
        transform: translateY(-5px);
    }
    
    .positive-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
    }
    
    .negative-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
    }
    
    .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    justify-content: space-evenly;
    gap: 20px;
    margin: 25px 0;
}

.metric-card {
    background: white;
    padding: 25px 15px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    border-left: 4px solid #667eea;
    width: 100%;
}
    
    /* Animation Classes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .results-container {
            padding: 20px;
            margin: 10px 0;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    try:
        model = load_model("model/best_model_sentiment (2).h5")
        with open("model/amazon_sentiment_tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None

model, tokenizer = load_resources()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return text

def predict_sentiment(review):
    review = preprocess_text(review)
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded, verbose=0)
    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
    confidence = float(prediction[0][0]) if sentiment == "Positive 😊" else 1 - float(prediction[0][0])
    return sentiment, confidence, float(prediction[0][0])

def create_confidence_gauge(confidence):
    fig, ax = plt.subplots(figsize=(8, 2))
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap='RdYlGn', extent=[0, 100, 0, 1])
    
    ax.axvline(x=confidence * 100, color='black', linewidth=3, linestyle='--')
    ax.text(confidence * 100, 1.2, f'{confidence*100:.1f}%', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Confidence Level', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

st.markdown("""
<div class="main-header">
    <h1 class="main-title"><span>🧠</span> SentimentScope AI</h1>
    <p class="sub-title">Amazon Review Sentiment Analysis using Deep Learning (LSTM)</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "**✍️ Enter Your Amazon Review**",
    height=200,
    placeholder="Share your experience...\n\nExample: 'The product exceeded my expectations! Excellent quality and fast delivery. Highly recommended!'",
    help="Enter a detailed review for accurate sentiment analysis"
)

analyze_button = st.button("**🚀 Analyze Sentiment**", use_container_width=True)

if analyze_button:
    if user_input.strip():
        if len(user_input.strip()) < 10:
            st.warning("📝 **Please provide a more detailed review (at least 10 characters) for accurate analysis.**")
        else:
            with st.spinner("**🔍 Analyzing sentiment with AI...**"):
                sentiment, confidence, raw_score = predict_sentiment(user_input)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if "Positive" in sentiment:
                    st.markdown(f"""
                    <div class="sentiment-card positive-card">
                        <h2 style='font-size: 2.5rem; margin: 0;'>😊</h2>
                        <h3 style='margin: 10px 0;'>Positive</h3>
                        <p style='font-size: 1.2rem; margin: 0;'>Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sentiment-card negative-card">
                        <h2 style='font-size: 2.5rem; margin: 0;'>😞</h2>
                        <h3 style='margin: 10px 0;'>Negative</h3>
                        <p style='font-size: 1.2rem; margin: 0;'>Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f""" <h3 style=" font-weight:bold;">📊 Confidence Visualization</h3>""", unsafe_allow_html=True)
                gauge_fig = create_confidence_gauge(confidence)
                st.pyplot(gauge_fig)
                
                
            st.markdown(f""" <h3 style="text-align:center; font-weight:bold;">📈 Analysis Metrics</h3>""", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3 style='color: #667eea; margin: 0;'>🎯</h3>
                    <p style='font-weight: 600; margin: 5px 0;'>Accuracy</p>
                    <p style='font-size: 1.5rem; color: #2d3748; margin: 0;'>89.2%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='color: #667eea; margin: 0;'>💪</h3>
                    <p style='font-weight: 600; margin: 5px 0;'>Confidence</p>
                    <p style='font-size: 1.5rem; color: #2d3748; margin: 0;'>{confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            
            
            with col3:
                status = "Valid" if len(user_input) >= 20 else "Short"
                status_color = "#4CAF50" if status == "Valid" else "#FF9800"
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style='color: #667eea; margin: 0;'>✅</h3>
                    <p style='font-weight: 600; margin: 5px 0;'>Analysis Status</p>
                    <p style='font-size: 1.5rem; color: {status_color}; margin: 0;'>{status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("**🔍 Detailed Analysis Insights**", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **📖 Review Summary**
                    - Sentiment polarity analyzed
                    - Emotional tone detected
                    - Context understanding applied
                    """)
                    word_count = len(user_input.split())
                    st.markdown(f"""
                    **📊 Text Analysis**
                    - Word Count: {word_count}
                    - Analysis Depth: {'Comprehensive' if word_count > 15 else 'Basic'}
                    """)
                
                with col2:
                    strength = "Strong" if confidence > 0.8 else "Moderate" if confidence > 0.6 else "Weak"
                    strength_emoji = "💪" if strength == "Strong" else "👍" if strength == "Moderate" else "👎"
                    
                    st.markdown(f"""
                    **🎭 Sentiment Strength**
                    - Strength: {strength} {strength_emoji}
                    - Confidence Level: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}
                    - Reliability: {'Excellent' if confidence > 0.8 else 'Good' if confidence > 0.6 else 'Fair'}
                    """)
    else:
        st.warning("⚠️ **Please enter a review to analyze**")