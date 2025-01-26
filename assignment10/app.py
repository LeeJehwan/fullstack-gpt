import os

import streamlit as st

from utils import (
    get_assistant_id,
    get_run_id,
    get_thread_id,
    paint_history,
    send_st_message,
    run_assistant,
)
from llm import LLMModels, get_llm_model

api_key = os.getenv("OPENAI_API_KEY", "")


st.set_page_config(
    page_title="Research AI",
    page_icon="🔥",
)


st.title("Research AI")
st.markdown(
    """
Welcome!
            
Use this Agent to research what you want!

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


if option:
    client = get_llm_model(option, api_key=api_key)
    assistant_id = get_assistant_id(client, option)

    if assistant_id and client:
        question = st.chat_input("Ask anything...")

        thread_id = get_thread_id(client)
        paint_history(client, thread_id)

        if question:
            send_st_message(question, "user")
            run_id = get_run_id(client, thread_id, assistant_id)
            run_assistant(client, run_id, thread_id, question)
