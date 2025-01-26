from abc import ABC, abstractmethod
from enum import Enum
import streamlit as st
from openai import OpenAI


class BaseLLM(ABC):
    llm: OpenAI

    def __init__(self, api_key="") -> None:
        self.api_key = api_key

    def run(self) -> OpenAI:
        self.validate_model()
        return self.llm

    @abstractmethod
    def validate_model():
        pass


class OpenAIModel(BaseLLM):

    def validate_model(self) -> None:
        try:
            self.llm = OpenAI(
                api_key=self.api_key,
            )
        except Exception:
            st.sidebar.error("Please check your open api key")


class LLMModels(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_O = "gpt-4o"

    @classmethod
    def all_models(cls) -> list[str]:
        return [cls.GPT_3_5_TURBO.value, cls.GPT_4_O.value]

    @classmethod
    def need_key(cls, option):
        if option == cls.GPT_3_5_TURBO.value:
            return True
        return False

    @classmethod
    def get_model_cls(cls, option, api_key):
        if cls.need_key(option) and not api_key:
            return None

        llm_model_map = {
            cls.GPT_3_5_TURBO.value: OpenAIModel(api_key=api_key),
            cls.GPT_4_O.value: OpenAIModel(api_key=api_key),
        }
        return llm_model_map.get(option)


def get_llm_model(option: str, api_key: str = ""):
    if st.session_state.get(option):
        return st.session_state[option]

    model = LLMModels.get_model_cls(option, api_key)
    llm = model.run() if model else None

    if llm:
        st.session_state[option] = llm

    return llm
