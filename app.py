from abc import ABC, abstractmethod
from enum import Enum
from operator import itemgetter

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import (
    CacheBackedEmbeddings,
    OpenAIEmbeddings,
    OllamaEmbeddings,
)
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.chat_models.base import BaseChatModel
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def get_embedding_model(option, api_key):
    if option == LLMModels.GPT_3_5_TURBO.value:
        return OpenAIEmbeddings(api_key=api_key)
    elif option == LLMModels.MISTRAL.value:
        return OllamaEmbeddings(model=option)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, option, key=""):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = get_embedding_model(option, key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


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


st.set_page_config(
    page_title="Assignment6",
    page_icon="🔥",
)

file = None

st.title("Assignment6 | DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

"""
)

if not file:
    st.markdown(
        """
    Upload your files on the sidebar.
    """
    )


api_key = ""


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def load_memory(_):
    x = st.session_state["memory"].load_memory_variables({})
    return x["history"]


with st.sidebar:
    option = st.selectbox("models", LLMModels.all_models())

    if option == LLMModels.GPT_3_5_TURBO.value:
        api_key = st.text_input(
            "OPEN API KEY",
            placeholder="Insert your open api key",
            type="password",
        )

    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )


if option:
    llm = get_llm_model(option, key=api_key)

    if file and llm:
        retriever = embed_file(file, option, key=api_key)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "history": RunnableLambda(
                        st.session_state["memory"].load_memory_variables
                    )
                    | itemgetter("history"),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                result = chain.invoke(message)

            st.session_state["memory"].save_context(
                {"input": message},
                {"output": result.content},
            )
    else:
        st.session_state["messages"] = []
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
