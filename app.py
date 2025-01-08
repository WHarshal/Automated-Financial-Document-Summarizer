from transformers import T5Tokenizer, T5ForConditionalGeneration
import streamlit as st
from io import BytesIO

# Load model and tokenizer
model_path = "t5_model"
tokenizer_path = "t5_tokenizer"

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Streamlit app layout
st.set_page_config(page_title="Automated Financial Document Summarizer", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f7ff;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 40px;
        color: #2d9cdb;
        font-weight: bold;
        text-align: center;
    }
    .summary-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .button {
        background-color: #2d9cdb;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }
    .button:hover {
        background-color: #1f7ac6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit title
st.markdown('<div class="title">Automated Financial Document Summarizer</div>', unsafe_allow_html=True)

# Input text
input_text = st.text_area("Enter text to summarize", height=200)

# Summarize button
if st.button("Summarize", key="summarize", help="Click to summarize the entered text"):
    if input_text:
        # Generate summary
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=300, min_length=100, length_penalty=1, num_beams=3, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display summary in a styled box
        st.markdown(f'<div class="summary-box"><h3><strong>Summary:</strong></h3><p>{summary}</p></div>', unsafe_allow_html=True)

        # Create a download button for the summary (convert to bytes)
        def download_summary(summary_text):
            # Convert text summary to bytes
            return BytesIO(summary_text.encode('utf-8'))

        st.download_button(
            label="Download Summary",
            data=download_summary(summary),
            file_name="summary.txt",
            mime="text/plain",
            key="download_summary"
        )
    else:
        st.error("Please enter some text to summarize.")
