import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="YouTube Revenue Predictor",
    page_icon="📊",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load('youtube_revenue_predictor.pkl')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Main app
st.title("📊 YouTube Revenue Predictor")
st.markdown("Enter your YouTube video metrics to predict potential revenue")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        views = st.number_input("Views", min_value=0, help="Total number of video views")
        likes = st.number_input("Likes", min_value=0, help="Number of likes on the video")
        shares = st.number_input("Shares", min_value=0, help="Number of times video was shared")
        
    with col2:
        new_comments = st.number_input("Comments", min_value=0, help="Number of comments on the video")
        subscribers = st.number_input("Subscribers", min_value=0, help="Channel subscriber count")
        engagement_rate = st.number_input("Engagement Rate (%)", 
                                        min_value=0.0, 
                                        max_value=100.0, 
                                        help="Engagement rate as percentage")
    
    submit_button = st.form_submit_button("Predict Revenue")

# Make prediction when form is submitted
if submit_button:
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Views': [views],
            'Subscribers': [subscribers],
            'Likes': [likes],
            'Shares': [shares],
            'New Comments': [new_comments],
            'Engagement Rate': [engagement_rate]
        })
        
        # Make prediction
        if model is not None:
            prediction = model.predict(input_data)
            
            # Display results
            st.success("Prediction Complete!")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Revenue", f"${prediction[0]:,.2f}")
            col2.metric("Views", f"{views:,}")
            col3.metric("Engagement Rate", f"{engagement_rate:.1f}%")
            
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check your input values and try again.")

# Footer
st.markdown("---")
