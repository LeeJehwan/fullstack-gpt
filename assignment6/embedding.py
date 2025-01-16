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


def get_embedding_model(option):
    if option == LLMModels.GPT_3_5_TURBO.value:
        return OpenAIEmbeddings()
    elif option == LLMModels.MISTRAL.value:
        return OllamaEmbeddings(model=option)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file, option):
    file_content = file.read()
    file_path = f"./assignment6/.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./assignment6/.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = get_embedding_model(option)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever
