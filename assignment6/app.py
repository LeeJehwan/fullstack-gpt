import os
from operator import itemgetter

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import streamlit as st

from chat_message import send_message, paint_history
from embedding import embed_file, format_docs
from prompt import prompt
from llm import LLMModels, get_llm_model


uploaded_file = None
supported_file_extensions = ["pdf", "txt", "docx"]
api_key = os.getenv("OPENAI_API_KEY", "")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)


st.set_page_config(
    page_title="Assignment6",
    page_icon="🔥",
)


st.title("Assignment6 | DocumentGPT")
st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

"""
)

with st.sidebar:
    option = st.selectbox("models", LLMModels.all_models())

    if LLMModels.need_key(option):
        api_key = st.text_input(
            "API KEY",
            api_key,
            placeholder="Insert your api key",
            type="password",
        )

    uploaded_file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=supported_file_extensions,
    )

if not uploaded_file:
    st.markdown("Upload your files on the sidebar.")


def get_chain():
    if not option:
        return None

    llm = get_llm_model(option, api_key=api_key)

    if not (llm and uploaded_file):
        return None

    retriever = embed_file(uploaded_file, option, api_key=api_key)
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "history": RunnableLambda(st.session_state["memory"].load_memory_variables)
            | itemgetter("history"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )


def invoke_chain(chain, message):
    try:
        with st.chat_message("ai"):
            result = chain.invoke(message)
        st.session_state["memory"].save_context(
            {"input": message},
            {"output": result.content},
        )
    except Exception:
        st.error("Please check your api key or llm server")


if chain := get_chain():
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    if message := st.chat_input("Ask anything about your file..."):
        send_message(message, "human")
        invoke_chain(chain, message)
