"""Local Streamlit front end for the ENT RAG SLM chatbot.

Run from the project root after running run_time_bootstrap.py:
    python -m streamlit run streamlit_ent_rag_app.py \
        --server.address=127.0.0.1 \
        --server.port=8501

This app is intentionally local-only.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image


# -----------------------------------------------------------------------------
# Import runtime modules
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
RUNTIME_SCRIPT_DIR = PROJECT_ROOT / "run_time" / "scripts"

for path in [RUNTIME_SCRIPT_DIR, PROJECT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import HF_TOKEN, SLM_CONFIG  # noqa: E402
from chatbot_module import ChatbotArchitecture  # noqa: E402
from session_module import ChatbotSession  # noqa: E402


# -----------------------------------------------------------------------------
# Page setup
st.set_page_config(
    page_title="ENT RAG SLM Chatbot",
    page_icon="🩺",
)

st.title("RAG-Enhanced SLM Otolaryngology Chatbot")
st.caption("The Running Prototype")


# -----------------------------------------------------------------------------
# Session-state helpers
def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def close_existing_session() -> None:
    session = st.session_state.get("chatbot_session")
    if session is not None:
        try:
            session.close()
        except Exception:
            pass

    st.session_state.pop("chatbot_session", None)
    st.session_state.pop("active_model_label", None)
    st.session_state.pop("active_model_card", None)
    clear_cuda_cache()


def initialize_session(model_label: str, model_card: str) -> None:
    with st.spinner(f"Loading {model_label}..."):
        chatbot = ChatbotArchitecture(
            model_card=model_card,
            is_eval_mode=False,
        )
        st.session_state.chatbot_session = ChatbotSession(chatbot=chatbot)
        st.session_state.active_model_label = model_label
        st.session_state.active_model_card = model_card


# -----------------------------------------------------------------------------
# Sidebar controls
model_cards = SLM_CONFIG["MODEL_CARDS"]
model_labels = list(model_cards.keys())

architecture_options = [
    "SLM",
    "Text-RAG",
    "Image-Text-RAG",
    "Image-RAG",
]

with st.sidebar:
    st.header("Runtime Controls")

    if not HF_TOKEN:
        st.warning(
            "HF_TOKEN was not found in the environment. "
            "Add it to .secrets or export it before loading the model."
        )

    selected_model_label = st.selectbox(
        "Model card",
        options=model_labels,
        index=model_labels.index("Llama 3.2 3B") if "Llama 3.2 3B" in model_labels else 0,
    )
    selected_model_card = model_cards[selected_model_label]

    selected_architecture = st.selectbox(
        "Architecture",
        options=architecture_options,
        index=1,
        help="SLM disables retrieval. Text-RAG retrieves text passages. Image-RAG retrieves image-linked neighbouring passages. Image-Text-RAG combines both retrieval routes.",
    )

    uploaded_image = None
    if selected_architecture in {"Image-RAG", "Image-Text-RAG"}:
        uploaded_image = st.file_uploader(
            "Image input",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )

    st.divider()

    load_clicked = st.button("Load / Switch Model", type="primary")
    clear_clicked = st.button("Clear Chat History")
    close_clicked = st.button("Close Model")

    if close_clicked:
        close_existing_session()
        st.success("Model session closed.")
        st.rerun()

    if clear_clicked and "chatbot_session" in st.session_state:
        st.session_state.chatbot_session.clear_chat_history()
        st.success("Chat history cleared.")
        st.rerun()

    st.divider()
    st.markdown("**Active model**")
    st.write(st.session_state.get("active_model_label", "No model loaded"))


# -----------------------------------------------------------------------------
# Model lifecycle
model_needs_loading = (
    "chatbot_session" not in st.session_state
    or st.session_state.get("active_model_card") != selected_model_card
)

if load_clicked or model_needs_loading:
    if st.session_state.get("active_model_card") != selected_model_card:
        close_existing_session()
    initialize_session(selected_model_label, selected_model_card)
    st.rerun()

session: ChatbotSession = st.session_state.chatbot_session


# -----------------------------------------------------------------------------
# Architecture routing
def build_generation_inputs(user_prompt: str):
    """Map the selected UI architecture to the session-level generate_response arguments."""

    rag_enabled = selected_architecture != "SLM"
    user_query = None
    user_image = None

    if selected_architecture in {"Text-RAG", "Image-Text-RAG"}:
        user_query = user_prompt

    if selected_architecture in {"Image-RAG", "Image-Text-RAG"}:
        if uploaded_image is None:
            raise ValueError("This architecture requires an uploaded image.")
        user_image = Image.open(uploaded_image).convert("RGB")

    return dict(
        user_prompt=user_prompt,
        user_query=user_query,
        user_image=user_image,
        RAG_enabled=rag_enabled,
    )


# -----------------------------------------------------------------------------
# Chat history rendering
for message in session.chat_history:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)


# -----------------------------------------------------------------------------
# Chat input and streaming response
placeholder = {
    "SLM": "Ask an ENT question without retrieval...",
    "Text-RAG": "Ask an ENT question using text retrieval...",
    "Image-Text-RAG": "Ask an ENT question using text and image-linked retrieval...",
    "Image-RAG": "Ask an ENT question using image-linked retrieval...",
}[selected_architecture]

user_prompt = st.chat_input(placeholder)

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        generation_inputs = build_generation_inputs(user_prompt)

        with st.chat_message("assistant"):
            st.write_stream(
                session.generate_response(**generation_inputs),
                cursor="▌",
            )

    except Exception as error:
        st.error(str(error))
