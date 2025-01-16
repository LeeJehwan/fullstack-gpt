from operator import itemgetter

from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st

from chat_message import send_message, paint_history
from embedding import embed_file, format_docs
from llm import LLMModels, get_llm_model


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
        retriever = embed_file(file, option)
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
