import json
import streamlit as st

from tools import functions, functions_map


def send_st_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(client, thread_id):
    messages = get_messages(client, thread_id)
    for message in messages:
        send_st_message(
            message.content[0].text.value,
            message.role,
        )


def get_assistant_id(client, option):
    if not client:
        return None

    if "assistant_id" in st.session_state:
        return st.session_state["assistant_id"]

    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="""
        You are a expert researcher.

        you should try to search in Wikipedia or DuckDuckGo.
        If it finds a website in DuckDuckGo it should enter the website and extract it's content.
        and then it should finish by saving the research to a .txt file.
        """,
        tools=functions,
        model=option,
    )
    st.session_state["assistant_id"] = assistant.id
    return assistant.id


def get_thread_id(client):
    if "thread_id" in st.session_state:
        return st.session_state["thread_id"]

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "assistant",
                "content": "I'm ready! Ask away!",
            }
        ]
    )
    st.session_state["thread_id"] = thread.id
    return thread.id


def get_run_id(client, thread_id, assistant_id):
    if "run_id" in st.session_state:
        run_id = st.session_state["run_id"]
        status = _get_run_status(client, run_id, thread_id)
        if status not in ["expired", "completed"]:
            return run_id

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    st.session_state["run_id"] = run.id
    return run.id


def _get_run(client, run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def _get_run_status(client, run_id, thread_id):
    return _get_run(client, run_id, thread_id).status


def _send_message(client, run_status, thread_id, content):
    if run_status in ["expired", "completed"]:
        client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )


def run_assistant(client, run_id, thread_id, content):
    run_status = _get_run_status(client, run_id, thread_id)
    _send_message(client, run_status, thread_id, content)

    with st.status("Loading..."):
        while _get_run_status(client, run_id, thread_id) == "requires_action":
            _submit_tool_outputs(client, run_id, thread_id)

    run_status = _get_run_status(client, run_id, thread_id)
    st.write(f"done, {run_status}")
    if run_status == "completed":
        message = get_messages(client, thread_id)[-1]
        send_st_message(message.content[0].text.value, message.role)
    elif run_status == "failed":
        st.write("Error")


def get_messages(client, thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def _get_tool_outputs(client, run_id, thread_id):
    run = _get_run(client, run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        st.write(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def _submit_tool_outputs(client, run_id, thread_id):
    outpus = _get_tool_outputs(client, run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs_and_poll(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outpus,
    )
