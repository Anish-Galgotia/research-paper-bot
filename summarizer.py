import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text, model="gpt-4"):
    prompt = f"""
You are an AI assistant. Summarize the following research paper content in bullet points.

Text:
{text}
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800
    )
    return response['choices'][0]['message']['content'].strip()

def explain_text_in_simple_terms(text, model="gpt-4"):
    prompt = f"""
You are an AI teacher. Explain the following research content in simple layman's terms so that a 10th-grade student can understand.

Text:
{text}
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800
    )
    return response['choices'][0]['message']['content'].strip()