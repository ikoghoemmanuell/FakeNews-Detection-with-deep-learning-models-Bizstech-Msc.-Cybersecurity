import streamlit as st
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("ikoghoemmanuell/finetuned_fake_news_bert")
tokenizer = AutoTokenizer.from_pretrained("ikoghoemmanuell/finetuned_fake_news_bert")

# Define the function for detecting fake news
@st.cache_resource
def detect_fake_news(text):
    # Load the pipeline.
    pipeline = transformers.pipeline("text-classification")

    # Predict the sentiment.
    prediction = pipeline(text)
    sentiment = prediction[0]["label"]
    score = prediction[0]["score"]

    return sentiment, score

# Setting the page configurations
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Add description and title
st.write("""
# Fake News Detection
Enter some text and we'll tell you if it's likely to be fake news or not!
""")

# Add image
image = st.image("https://docs.gato.txst.edu/78660/w/2000/a_1dzGZrL3bG/fake-fact.jpg", width=400)

# Get user input
text = st.text_input("Enter some text here:")

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

# Show fake news detection output
if text:
    label, score = detect_fake_news(text)
    if label == "Fake":
        st.error(f"The text is likely to be fake news with a confidence score of {score*100:.2f}%!")
    else:
        st.success(f"The text is likely to be genuine with a confidence score of {score*100:.2f}%!")
