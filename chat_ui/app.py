
import base64
import io
import os

from dotenv import load_dotenv
import httpx
import streamlit as st
from PIL import Image

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
    base64_image: str | None = None
) -> str:
    if base64_image is not None:
        payload = {
            "message": message,
            "base64_image": base64_image
        }
    response = httpx.post(
        f"{API_URL}/conversations/{conversation_id}/messages",
        json=payload if base64_image is not None else {"message": message},
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
        text_content = message.get(
            "text_content",
            message.get("content", ""),
        )
        if text_content:
            st.markdown(text_content)

        image_content = message.get("image_content")
        if image_content:
            st.image(base64.b64decode(image_content))
            

# Handle new user input
prompt = st.chat_input("Ask a question...", accept_file=True, file_type="image/*")

if prompt:
    text_content = prompt.get("text", "")
    uploaded_files = prompt.get("files") # Returns [] if no files were uploaded
    if not uploaded_files:
        uploaded_file = None
    else:
        uploaded_file = uploaded_files[0] 

    if uploaded_file is not None:
        # Convert image to JPEG if a file was uploaded
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            
            # Convert RGBA / PNG / other formats to RGB (JPEG doesn't support transparency)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            
            # Save it into an in-memory bytes buffer as JPEG
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            
            # Replace uploaded_file bytes/content or create a new handle/variable for it
            image_bytes = buffered.getvalue()
    else:
        text_content = text_content
        uploaded_file = None
        image_bytes = None

    with st.chat_message("user"):
        if text_content:
            st.markdown(text_content)
        if uploaded_file is not None:
            st.image(image_bytes)

    if uploaded_file is not None:
        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    else:
        base64_encoded = None

    st.session_state.messages.append({
        "role": "user",
        "text_content": text_content,
        "image_content": base64_encoded,
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = send_message(
                    st.session_state.conversation_id,
                    text_content,
                    base64_encoded
                )
                st.markdown(reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "text_content": reply,
                    "image_content": None,
                })
            except httpx.HTTPError:
                st.error(
                    "The agent API request failed. "
                    "Check the API process and its logs."
                )
