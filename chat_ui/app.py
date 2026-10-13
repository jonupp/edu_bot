
import os

from dotenv import load_dotenv
import httpx
import streamlit as st

load_dotenv(".env")

API_URL = os.getenv(
    "AGENT_API_URL",
    "http://127.0.0.1:1111",
)

st.set_page_config(
    page_title="Edu Bot",
    page_icon="💬",
    layout="centered",
)

st.title("Edu Bot")

# HTTP client helpers

def create_conversation() -> str:
    response = httpx.post(
        f"{API_URL}/conversations",
        timeout=15,
    )
    response.raise_for_status()
    return response.json()["conversation_id"]


def load_conversations() -> list[dict]:
    response = httpx.get(
        f"{API_URL}/conversations",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["conversations"]

def load_messages(conversation_id: str) -> list[dict]:
    response = httpx.get(
        f"{API_URL}/conversations/{conversation_id}/messages",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["messages"]


def send_message(
    conversation_id: str,
    message: str,
) -> str:
    response = httpx.post(
        f"{API_URL}/conversations/{conversation_id}/messages",
        json={"message": message},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["reply"]


# Initialize UI state
if "conversation_id" not in st.session_state:
    try:
        st.session_state.conversation_id = create_conversation()
        st.session_state.messages = []
        st.session_state.history_loaded = True
    except httpx.HTTPError:
        st.error(
            "Cannot connect to the agent API. "
            "Start the FastAPI service and refresh this page."
        )
        st.stop()

with st.sidebar:
    st.caption("Conversation")
    if st.button("New conversation", use_container_width=True, type="primary"):
        try:
            st.session_state.conversation_id = create_conversation()
            st.session_state.messages = []
            st.session_state.history_loaded = True
            st.rerun()
        except httpx.HTTPError:
            st.error("Could not create a conversation.")

    conversations = load_conversations()

    for conversation in conversations:
        if st.button(conversation["thread_id"], type="secondary"):
            st.session_state.conversation_id = conversation["thread_id"]
            st.session_state.messages = []
            st.session_state.history_loaded = False
            st.rerun()


# Restore checkpointed history once per UI session/conversation.
if not st.session_state.get("history_loaded", False):
    try:
        st.session_state.messages = load_messages(
            st.session_state.conversation_id
        )
        st.session_state.history_loaded = True
    except httpx.HTTPError:
        st.error("Could not load conversation history.")

# Render conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = send_message(
                    st.session_state.conversation_id,
                    prompt,
                )
                st.markdown(reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                })
            except httpx.HTTPError:
                st.error(
                    "The agent API request failed. "
                    "Check the API process and its logs."
                )
