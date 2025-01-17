from abc import ABC, abstractmethod
from enum import Enum
import streamlit as st
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.base import BaseCallbackHandler

from chat_message import save_message


class ChatCallbackHandler(BaseCallbackHandler):

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


class BaseLLM(ABC):
    llm: BaseChatModel

    def __init__(self, api_key="") -> None:
        self.api_key = api_key

    def run(self) -> BaseChatModel:
        self.validate_model()
        return self.llm

    @abstractmethod
    def validate_model():
        pass


class OpenAIModel(BaseLLM):

    def validate_model(self) -> None:
        try:
            self.llm = ChatOpenAI(
                model=LLMModels.GPT_3_5_TURBO.value,
                temperature=0.1,
                streaming=True,
                api_key=self.api_key,
                callbacks=[
                    ChatCallbackHandler(),
                ],
            )
            # with st.chat_message("ai"):
            #     self.llm.invoke("hello")
        except Exception:
            st.sidebar.error("Please check your open api key")


class MistralModel(BaseLLM):
    def validate_model(self) -> None:
        try:
            self.llm = ChatOllama(
                model=LLMModels.MISTRAL.value,
                temperature=0.1,
                streaming=True,
                callbacks=[
                    ChatCallbackHandler(),
                ],
            )
            # with st.chat_message("ai"):
            #     self.llm.invoke("hello")
        except Exception:
            st.sidebar.error("Please check your ollama server")


class LLMModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    MISTRAL = "mistral:latest"

    @classmethod
    def all_models(cls) -> list[str]:
        return [model.value for model in cls]

    @classmethod
    def need_key(cls, option):
        if option == cls.GPT_3_5_TURBO.value:
            return True
        return False

    @classmethod
    def get_model_cls(cls, option):
        llm_model_map = {
            cls.GPT_3_5_TURBO.value: OpenAIModel,
            cls.MISTRAL.value: MistralModel,
        }
        return llm_model_map.get(option)


def get_llm_model(option: str, api_key: str = ""):
    if st.session_state.get(option):
        return st.session_state[option]

    model = LLMModels.get_model_cls(option)
    llm = model(api_key).run() if model else None

    if llm:
        st.session_state[option] = llm
        return llm
