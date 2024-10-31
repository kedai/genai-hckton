from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
import requests
import base64

class DatabricksChatModel(BaseChatModel):
    endpoint_url: str
    api_token: str
    model: str = "databricks-llm"
    temperature: float = 0.0
    max_tokens: int = 1000

    @property
    def _llm_type(self) -> str:
        return "databricks-chat"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                if 'image_path' in msg.additional_kwargs:
                    image_b64 = self._encode_image(msg.additional_kwargs['image_path'])
                    # Combine text and image URL into a single string
                    content = f"{msg.content}\n![image](data:image/jpeg;base64,{image_b64})"
                    formatted_messages.append({"role": "user", "content": content})
                else:
                    formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})

        payload = {
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }

        response = requests.post(
            self.endpoint_url,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            response_data = response.json()
            if "predictions" in response_data:
                content = response_data["predictions"][0].get("content", "")
            elif "candidate_responses" in response_data:
                content = response_data["candidate_responses"][0].get("text", "")
            elif "choices" in response_data:
                content = response_data["choices"][0].get("message", {}).get("content", "")
            else:
                content = str(response_data)

            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")