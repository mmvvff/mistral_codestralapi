import os
from dotenv import load_dotenv, find_dotenv     

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage



api_key = None
dlai_endpoint = None
client = None

def load_env():
    _ = load_dotenv(find_dotenv())

def load_mistral_api_key(ret_key=False):
    load_env()
    global api_key
    api_key = os.getenv("MISTRAL_API_KEY")

    global client
    client = MistralClient(api_key=api_key)
    
    if ret_key:
        return api_key

def mistral(user_message, 
            model="open-mistral-7b",
            is_json=False):
    client = MistralClient(api_key=api_key)
    messages = [ChatMessage(role="user", content=user_message)]

    if is_json:
        chat_response = client.chat(
            model=model, 
            messages=messages,
            response_format={"type": "json_object"})
    else:
        chat_response = client.chat(
            model=model, 
            messages=messages)

    return chat_response.choices[0].message.content