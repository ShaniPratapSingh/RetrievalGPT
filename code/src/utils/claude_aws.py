import os
import pdb

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    try:
        with open('./claude_api_aws.key', 'r') as f:
            keys = f.readlines()
            if not AWS_ACCESS_KEY_ID and len(keys) > 0:
                AWS_ACCESS_KEY_ID = keys[0].strip()
            if not AWS_SECRET_ACCESS_KEY and len(keys) > 1:
                AWS_SECRET_ACCESS_KEY = keys[1].strip()
    except FileNotFoundError:
        pass

AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID or "MOCK_AWS_ACCESS_KEY"
AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY or "MOCK_AWS_SECRET_KEY"

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
from anthropic import AnthropicBedrock
client = AnthropicBedrock(
    aws_access_key=AWS_ACCESS_KEY_ID,
    aws_secret_key=AWS_SECRET_ACCESS_KEY,
    aws_region=AWS_DEFAULT_REGION
)



def chat_haiku(prompt):

    if isinstance(prompt, str):
        message = [{
            'role': 'user',
            'content': prompt
        }]
    else:
        message = prompt
    
    message = client.messages.create(
        temperature=0,
        model="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=1024,
        messages=message,
    )
    
    return message.content[0].text

def chat_sonnet(prompt):
    
    if isinstance(prompt, str):
        message = [{
            'role': 'user',
            'content': prompt
        }]
    else:
        message = prompt

    message = client.messages.create(
        temperature=0,
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=1024,
        messages=message,
    )
    
    return message.content[0].text


if __name__ == "__main__":
    prompt = "Write a haiku about the ocean."
    print(chat_haiku(prompt))