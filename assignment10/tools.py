from langchain.tools import DuckDuckGoSearchResults
from langchain.retrievers import WikipediaRetriever
import streamlit as st


def get_web_search(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchResults(backend="auto")
    return ddg.run(query)


def get_wikipedia_search(inputs):
    query = inputs["query"]
    retriever = WikipediaRetriever(top_k_results=1)
    return retriever.get_relevant_documents(query)


def store_text_to_file(inputs):
    name = inputs["name"]
    contents = inputs["contents"]
    with open(f"./{name}.txt", "w") as file:
        file.write(contents)

    with st.chat_message("assistant"):
        st.download_button(
            label="download",
            data=contents,
            file_name=f"{name}.txt",
            mime="text/plain",
        )
    return contents


functions_map = {
    "get_web_search": get_web_search,
    "get_wikipedia_search": get_wikipedia_search,
    "store_text_to_file": store_text_to_file,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_web_search",
            "description": "retrieve research information from web using DuckDuckGo for a specific keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for contents.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_wikipedia_search",
            "description": "retrieve research information from wikipedia for a specific keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for contents.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_text_to_file",
            "description": "Storing the contents of text into .txt file",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of file to be stored.",
                    },
                    "contents": {
                        "type": "string",
                        "description": "The contents will stroing in .txt file.",
                    },
                },
                "required": ["name", "contents"],
            },
        },
    },
]
