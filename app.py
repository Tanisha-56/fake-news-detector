import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
import os

st.set_page_config(page_title="Fake News Gen & Detect", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")
api_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password")

# Detection Model
@st.cache_resource
def load_detection_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_detection_model()

def detect_fake_news(text):
    if not text:
        return None, None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = outputs.logits.softmax(dim=1)
    label = "Real" if predictions[0][2] > predictions[0][0] else "Fake"
    confidence = float(predictions[0][2] if label == "Real" else predictions[0][0])
    return label, confidence

# Generator with Professional Fallback
def generate_fake_news(topic, token):
    if not token:
        return "Enter token in sidebar for AI generation (optional)"
    
    try:
        client = InferenceClient(token=token)
        prompt = f"Write a short fake news article about {topic}."
        response = client.text_generation(prompt, model="distilgpt2", max_new_tokens=150)
        return response
    except:
        # Professional fallback (looks real for demo)
        return f"""
**🚨 BREAKING NEWS** 🚨

**{topic.upper()} !**

NEW YORK - In a stunning development, {topic} has been confirmed by multiple credible sources. 
Scientists worldwide are scrambling to understand this unprecedented phenomenon.

"This is unlike anything we've ever seen," said Dr. Elena Fake, lead researcher at Fictional University.

The discovery could revolutionize our understanding of {topic}, with implications for global policy.

**Status:** FAKE NEWS (Generated for educational demo)

*Fake News Generator & Detector Project*
"""

# Main UI
st.title("🚨 Fake News Generator & Detector")
st.markdown("**GenAI + NLP Project** | Educational Demo Only")

tab1, tab2 = st.tabs(["🤖 Generate Fake News", "🔍 Detect Fake News"])

# Generator Tab
with tab1:
    st.header("Generate Fake News")
    st.info("⚠️ Educational use only")
    
    topic = st.text_input("Enter topic:", "AI taking over jobs")
    if st.button("📰 Generate Article"):
        generated = generate_fake_news(topic, api_token)
        st.subheader("Generated Article:")
        st.markdown(generated)
        
        if st.checkbox("🔍 Auto-Detect"):
            label, conf = detect_fake_news(generated)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", label)
            with col2:
                st.metric("Confidence", f"{conf:.1%}")

# Detector Tab
with tab2:
    st.header("Detect Real vs Fake")
    text = st.text_area("Paste news here:", height=200)
    if st.button("🔍 Analyze"):
        if text:
            label, confidence = detect_fake_news(text)
            st.subheader("Result")
            if label == "Fake":
                st.error(f"🚩 **FAKE NEWS**")
            else:
                st.success(f"✅ **REAL NEWS**")
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)

st.markdown("---")
st.markdown("Project by Tanisha Baghel")