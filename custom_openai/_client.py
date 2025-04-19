from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from typing import List, Optional
import requests

class CustomOpenAIClient:
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def chat_completions_create(
            self,
            model: str,
            messages: List[dict],
            max_tokens: Optional[int] = None,
        ) -> ChatCompletion:
        response = requests.post(
            self.api_endpoint+"chat/create",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            },)
        response.raise_for_status()

        return ChatCompletion(**response.json())
    
    def embeddings_create(
            self,
            model: str,
            input: List[str],
            dimensions: Optional[int] = None,
        ) -> CreateEmbeddingResponse:
        response = requests.post(
            self.api_endpoint+"embeddings/create",
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "inputs": input,
                "dimensions": dimensions,
            },)
        response.raise_for_status()
        
        return CreateEmbeddingResponse(**response.json())
            
