from pathlib import Path

from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import (
    CacheBackedEmbeddings,
    OpenAIEmbeddings,
    OllamaEmbeddings,
)
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

from llm import LLMModels


def get_embedding_model(option, api_key):
    if option == LLMModels.GPT_3_5_TURBO.value:
        return OpenAIEmbeddings(api_key=api_key)
    elif option == LLMModels.MISTRAL.value:
        return OllamaEmbeddings(model=option)
    raise Exception("Fail to find embedding model")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, option, api_key=""):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = get_embedding_model(option, api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever
