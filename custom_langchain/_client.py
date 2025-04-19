import requests
import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from langchain_core.embeddings import Embeddings

class CustomOpenAIEmbeddings(Embeddings):
    def __init__(self, embeddings_endpoint: str, api_key:str, model: str = "text-embedding-3-large"):
        self.model = model
        self.embeddings_endpoint = embeddings_endpoint
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.embeddings_endpoint,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "inputs": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class CustomOpenAiModel(BaseChatModel):
    """Custom OpenAI model for chat completion.
    This class is a wrapper around the OpenAI API for chat completion.
    It allows for customization of the model name, temperature, max tokens,
    timeout, and other parameters.
    """

    model_name: str = Field(alias="model")
    end_point: str = Field(alias="end_point")
    api_key: str
    """The name of the model"""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        print("Generating")

        print("Messages Content:")
        for message in messages:
            print(message.content)


        # Replace this with actual logic to generate a response from a list of messages.
        last_message = messages[-1]
        message_contents = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                message_contents.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_contents.append({"role": "assistant", "content": msg.content})

        headers = {
            'Content-Type': 'application/json',
            "x-api-key": self.api_key
        }
        json_data = {
            "model": self.model_name,
            "messages": message_contents,
            "stream": False
        }

        response = requests.post(self.end_point, headers=headers, json=json_data)
        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
        
        response_json = json.loads(response.text)
        content = response_json["choices"][0]['message']['content']

        meta_data = response.text.splitlines()[-1]
        meta_json = json.loads(meta_data)
        ct_input_tokens = meta_json.get("prompt_eval_count", 0)
        ct_output_tokens = meta_json.get("eval_count", 0)

        message = AIMessage(
            content=content,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model_name": self.model_name,
            "end_point": self.end_point,
        }