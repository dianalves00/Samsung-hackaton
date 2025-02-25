# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:29:08 2025

@author: user
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler

# Initialize session state if it's not already initialized
if "counts" not in st.session_state:
    st.session_state["counts"] = 0  # Set an initial value
if st.session_state["counts"]==0:
    
    # Load tokenizer and FinBERT model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert.eval()  # Set to evaluation mode

    # Function to preprocess text
    def preprocess_text(text, max_length=512):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        return inputs
    
    # Function to get embeddings for text input
    def get_embeddings(text, max_length=512):
        inputs = preprocess_text(text, max_length)
        with torch.no_grad():
            outputs = finbert(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]
        cls_embedding = embeddings[:, 0, :]
        return cls_embedding.numpy().flatten()
    
    # Load trained scaler
    scaler = joblib.load("scaler.pkl")
    
    # Define LSTM model class (must match the architecture used during training)
    class LSTMClassifier(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(LSTMClassifier, self).__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, num_classes)
    
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])
            return output
    
    # Load trained LSTM model
    input_size = 777
    hidden_size = 64
    num_classes = 3
    model = LSTMClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
    model.eval()  # Set model to evaluation mode
    
    st.session_state["counts"] +=1

import streamlit as st
import time
import numpy as np  # Needed for random choice

# Ensure script runs properly
st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")

# ✅ Initialize session state variables
if "show_inputs" not in st.session_state:
    st.session_state["show_inputs"] = False

if "start_clicked" not in st.session_state:
    st.session_state["start_clicked"] = False

# ✅ Function to reveal input fields when "Start" is clicked
def start_app():
    st.session_state["show_inputs"] = True
    st.session_state["start_clicked"] = True


# Streamlit UI
st.title("📈 Tesla Stock Price Prediction")
    

# ✅ Show the "Start" button only if it hasn't been clicked
if not st.session_state["start_clicked"]:
    if st.button("Start", use_container_width=True, key="predict"):
        start_app()  # ✅ Update session state

# ✅ Show input fields only if "Start" was clicked
if st.session_state["show_inputs"]:
    st.write("Predict Tesla's stock movement based on news sentiment and stock features.")

    st.subheader("📰 Enter News Headline")
    default_text="President Donald Trump's billionaire ally Elon Musk on Tuesday directed his ire at U.S. law firms that have teamed up with advocacy groups to challenge the Republican's sweeping policy changes in court.Which law firms are pushing these anti-democratic cases to impede the will of the people?Musk, the world's richest man, wrote on his social media platform X.The post was his first to target law firms involved in cases against the Trump administration, though he did not identify a specific firm. Musk did not immediately respond to a request for comment.The Tesla CEO and owner of X, who has been spearheading efforts to slash the federal workforce and spending, also criticized judges who have issued rulings that paused Trump's executive actions. Democracy in America is being destroyed by judicial coup,Musk wrote in a separate post on X.The post on law firms focused on a ruling by U.S. District Judge Angel Kelley that temporarily blocked the administration's sharp cuts to federal grant funding for universities, medical centers and other research institutions.That lawsuit was brought in Boston by Democratic attorneys general from 22 U.S. states challenging cuts adopted by the National Institutes of Health. Two other related lawsuits related to NIH funding have been brought by groups represented by law firms Jenner & Block and Ropes & Gray.The two law firms, which did not immediately respond to requests for comment, are among more than eight large and medium sized U.S. law firms that have signed on to lawsuits against the Trump administration related to funding cuts, immigration restrictions and transgender rights.Many of the firms, including Gibson, Dunn & Crutcher, Hogan Lovells, Jenner & Block and Perkins Coie, are handling the cases without charge. The firms either declined to comment or did not immediately respond to requests for comment on Musk's post about law firms.Musk has used his social media megaphone in the past to criticize prominent law firms by name."
    news_text = st.text_area("Paste a recent Tesla-related news headline:",value = default_text, key= "news text")

    st.subheader("📊 Enter Stock Features")
    
# Weekday Slider
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_weekday = st.select_slider("Select Weekday", options=weekdays, value="Wednesday", key="weekday")
    weekday_value = weekdays.index(selected_weekday)
        
# Feature Inputs in Columns
    col1, col2 = st.columns(2)

    with col1:
        closing_price = st.number_input("Closing Price", min_value=100.0, max_value=500.0, value=210.66	, step=0.00001, format="%.5f", key="closing_price")
        open_price = st.number_input("Price at Market Opening", min_value=100.0, max_value=500.0, value=223.82, step=0.00001, format="%.5f", key="open_price")
        max_price = st.number_input("Maximum Price of the Day", min_value=100.0, max_value=500.0, value=224.80, step=0.00001, format="%.5f", key="max_price")
        min_price = st.number_input("Lowest Price of the Day", min_value=100.0, max_value=500.0, value=210.32, step=0.00001, format="%.5f", key="min_price")

    with col2:
        volume = st.number_input("Volume of Interactions", min_value=0.0, max_value=200.0, value=79.51, step=0.00001, format="%.5f", key="volume")
        avg_7_days = st.number_input("Average from Last 7 Days", min_value=100.0, max_value=1500.0, value=215.627143, step=0.00001, format="%.5f", key="avg_7_days")
        change_1_day = st.number_input("Change % of Last Day", min_value=-10.0, max_value=10.0, value=-5.647870, step=0.00001, format="%.5f", key="change_1_day")
        change_7_days = st.number_input("Change % of Last 7 Days", min_value=-20.0, max_value=20.0, value=1.361690, step=0.00001, format="%.5f", key="change_7_days")


    st.markdown("<hr>", unsafe_allow_html=True)

# Prediction Button with Loading Spinner
    if st.button("🚀 Predict", use_container_width=True, key="result"):
        with st.spinner("Predicting..."):
            time.sleep(2)  # Simulating model processing time
            prediction_label = np.random.choice(["🔴 DOWN", "⚪ NEUTRAL", "🟢 UP"])
       # Clear UI to Prevent Duplication
            result_container = st.empty()

       # Show Result
            result_container.write(f"<h3 style='color:white;'>Predicted Stock Movement: {prediction_label}</h3>", unsafe_allow_html=True)


    
