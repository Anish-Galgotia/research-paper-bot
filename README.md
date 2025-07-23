# Research Paper Summarizer & Explainer Bot

This GenAI-powered app lets you upload a research paper (PDF), and then:

- **Summarize** the paper using GPT-4
- **Explain** it in layman’s terms
- **Ask questions** about the content (LangChain + FAISS)
- **Get related research papers** via semantic similarity

## Tech Stack

- **Frontend**: Streamlit
- **LLMs**: OpenAI GPT-4 via API
- **PDF Parsing**: PyMuPDF
- **Embeddings**: OpenAI + Sentence Transformers
- **Vector Store**: FAISS (local)
- **LangChain**: RetrievalQA for document Q&A

## Getting Started

1. Clone this repo  
2. Create virtual env & install deps  
3. Add your `.env` file with:
    ```
    OPENAI_API_KEY=your_key_here
    ```
4. Run:
    ```
    streamlit run app.py
    ```

## Example PDF

Upload any academic paper, and try queries like:
- “What is the contribution?”
- “Which model is used?”
- “Explain in simple language.”



