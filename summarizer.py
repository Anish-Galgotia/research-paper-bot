import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=openai_key)

def summarize_text(text, model="gpt-4"):
    prompt = f"""
You are an AI assistant. Summarize the following research paper content in bullet points.

Text:
{text}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

def explain_text_in_simple_terms(text, model="gpt-4"):
    prompt = f"""
You are an AI teacher. Explain the following research content in simple layman's terms so that a 10th-grade student can understand.

Text:
{text}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()
