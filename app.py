import streamlit as st
import fitz # PyMuPDF
from utils.pdf_loader import extract_text_from_pdf

# Title
st.title("Research Paper Summarizer & Explainer Bot")
st.write("Upload a PDF research paper and we'll extract its content")

# File uploader
uploaded_file = st.file_uploader("Choose a research paper (PDF only)", type="pdf")

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract and show PDF text
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)

    st.success("Text extracted successfully!")
    st.subheader("Content Preview (first 1000 characters)")
    st.text(extracted_text[:1000])

    # save full text for next steps
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

