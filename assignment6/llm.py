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


class LLMModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    MISTRAL = "mistral:latest"

    @classmethod
    def all_models(cls):
        return [model.value for model in cls]


class BaseLLM(ABC):
    llm: BaseChatModel

    def run(self) -> BaseChatModel:
        self.validate_model()
        return self.llm

    @abstractmethod
    def validate_model():
        pass


class OpenAIModel(BaseLLM):
    def __init__(self, key) -> None:
        self.key = key

    def validate_model(self) -> None:
        try:
            self.llm = ChatOpenAI(
                model=LLMModels.GPT_3_5_TURBO.value,
                temperature=0.1,
                streaming=True,
                api_key=self.key,
                callbacks=[
                    ChatCallbackHandler(),
                ],
            )
            with st.chat_message("ai"):
                self.llm.invoke("hello")
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
            with st.chat_message("ai"):
                self.llm.invoke("hello")
        except Exception:
            st.sidebar.error("Please check your ollama server")


def get_llm_model(model: str, key: str = ""):
    if st.session_state.get(model):
        return st.session_state[model]

    llm = None
    if model == LLMModels.GPT_3_5_TURBO.value:
        if key:
            llm = OpenAIModel(key).run()
    elif model == LLMModels.MISTRAL.value:
        llm = MistralModel().run()

    if llm:
        st.session_state[model] = llm
        return llm
