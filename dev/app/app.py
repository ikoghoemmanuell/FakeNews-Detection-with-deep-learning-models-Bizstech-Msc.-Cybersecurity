import streamlit as st
import transformers
import torch
import requests
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setting the page configurations
st.set_page_config(
    page_title="Fake News Detection App", 
    page_icon="fas fa-exclamation-triangle", 
    layout="wide", 
    initial_sidebar_state="auto")

# Load the model and tokenizer
model_name = AutoModelForSequenceClassification.from_pretrained("Johnson-Olakanmi/finetuned_fake_news_roberta")
tokenizer_name = AutoTokenizer.from_pretrained("Johnson-Olakanmi/finetuned_fake_news_roberta")


# Define the CSS style for the app
st.markdown(
"""
<style>
body {
    background-color: #f5f5f5;
}
h1 {
    color: #4e79a7;
}
</style>
""",
unsafe_allow_html=True
)

# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox(
    "Select an option", 
    menu)

# Define the function for detecting fake news
@st.cache_resource
def detect_fake_news(text):
    # Load the pipeline.
    pipeline = transformers.pipeline("text-classification", 
                                     model=model_name, 
                                     tokenizer=tokenizer_name)

    # Predict the sentiment.
    prediction = pipeline(text)
    sentiment = prediction[0]["label"]
    score = prediction[0]["score"]

    return sentiment, score

    
# Home section
if choice == 'Home':
    st.markdown("<h1 style='text-align: center;margin-top:0px;'>TRUTH- A fake news detection app</h1>", 
                unsafe_allow_html=True)

    # Loading GIF
    gif_url = "https://thumbs.gfycat.com/AnchoredWeeklyGreatwhiteshark-size_restricted.gif"
    st.image(gif_url, 
             use_column_width=True, 
             width=400)
    
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is a Fake News Detection App.</p>", 
                unsafe_allow_html=True)
    
    # Get user input
    text = st.text_input("Enter some text and we'll tell you if it's likely to be fake news or not!")
    
    if st.button('Predict'):
        # Show fake news detection output
        if text:
            with st.spinner('Checking if news is Fake...'):
                label, score = detect_fake_news(text)
                if label == "LABEL_1":
                    st.error(f"The text is likely to be fake news with a confidence score of {score*100:.2f}%!")
                else:
                    st.success(f"The text is likely to be genuine with a confidence score of {score*100:.2f}%!")
        else:
            with st.spinner('Checking if news is Fake...'):
                st.warning("Please enter some text to detect fake news.")


# About section
if choice == 'About':
    # Load the banner image
    banner_image_url = "https://docs.gato.txst.edu/78660/w/2000/a_1dzGZrL3bG/fake-fact.jpg"
 
    # Display the banner image
    st.image(
        banner_image_url, 
        use_column_width=True, 
        width=400)
    st.markdown('''
                    <p style='font-size: 20px; font-style: italic;font-style: bold;'>
                        
                        TRUTH is a cutting-edge application specifically designed to combat the spread of fake news. 
                        Using state-of-the-art algorithms and advanced deep learning techniques, 
                        our app empowers users to detect and verify the authenticity of news articles. 
                        TRUTH provides accurate assessments of the reliability of news content. 
                        With its user-friendly interface and intuitive design, 
                        the app enables users to easily navigate and obtain trustworthy information in real-time. 
                        With TRUTH, you can take control of the news you consume and make informed decisions based on verified facts.
                        
                    </p>
                ''', 
                unsafe_allow_html=True)
