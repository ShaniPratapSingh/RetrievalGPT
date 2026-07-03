from openai import OpenAI
import pdb
import os

import os
from dotenv import load_dotenv
load_dotenv()

# Load API key from environment, falling back to local file
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        with open('./openai_api.key', 'r') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        api_key = None

def get_api_key():
    global api_key
    if api_key:
        return api_key
    # Check env again (e.g. if loaded dynamically)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            with open('./openai_api.key', 'r') as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            pass
    return api_key

# We initialize a mock client if no key is present to prevent import crashes
client = OpenAI(api_key=api_key or "MOCK_KEY")
def gpt_chat(model, prompt, seed=44):
    key = get_api_key()
    if not key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in environment or .env file.")
    client = OpenAI(api_key=key)
    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content

def gpt_chat_35(prompt, seed=44):
    key = get_api_key()
    if not key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in environment or .env file.")
    client = OpenAI(api_key=key)
    
    response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=4096,
    temperature=0.5,
    logprobs=True
    )
    
    return response.choices[0].message.content

if __name__ == '__main__':
    prompt = "What is the ICD-10 code for Diabetes Mellitus?"
    model = "gpt-3.5-turbo"
    response = gpt_chat(model, prompt)
    print(response)