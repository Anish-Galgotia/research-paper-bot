import streamlit as st
import fitz # PyMuPDF
from utils.pdf_loader import extract_text_from_pdf
from summarizer import summarize_text, explain_text_in_simple_terms
from qa_engine import create_faiss_index, ask_question

# Title
st.title("Research Paper Summarizer & Explainer Bot")
st.write("Upload a PDF research paper and we'll extract its content")

# File uploader
uploaded_file = st.file_uploader("Choose a research paper (PDF only)", type="pdf")

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


# Summarization
if uploaded_file:
    st.subheader("GPT-4 Summary")
    if st.button("Summarize"):
        with st.spinner("Summarizing using GPT-4..."):
            summary = summarize_text(extracted_text[:3000])  # limit tokens
            st.success("Summary Generated!")
            st.markdown(summary)

    st.subheader("Explain in simple language")
    if st.button("Explain"):
        with st.spinner("Explaining in simple language..."):
            explaination = explain_text_in_simple_terms(extracted_text[:3000])
            st.success("Explaination Ready!")
            st.markdown(explaination)

st.subheader("🤖 Ask Questions About the Paper")

if uploaded_file:
    if st.button("Index the Paper for Q&A"):
        with st.spinner("Indexing paper locally using FAISS..."):
            create_faiss_index(extracted_text[:10000])
            st.success("Paper indexed locally!")

    query = st.text_input("Ask a question about the paper:")
    if query:
        with st.spinner("Thinking..."):
            answer = ask_question(query)
            st.markdown(f"**Answer:** {answer}")