import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.errors import StreamlitSecretNotFoundError


MODEL = "openai/gpt-4o-mini"
CHAT_DIR = Path(__file__).parent / "chats"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, friendly assistant."


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_conversation() -> dict[str, Any]:
    timestamp = now_utc()
    return {
        "id": uuid.uuid4().hex,
        "title": "New chat",
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def load_conversations() -> tuple[dict[str, dict[str, Any]], list[str]]:
    conversations: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    try:
        paths = sorted(CHAT_DIR.glob("*.json"))
    except OSError as error:
        return conversations, [f"Could not read the chats folder: {error}"]

    for path in paths:
        if not re.fullmatch(r"[0-9a-fA-F]{32}", path.stem):
            errors.append(f"Skipped chat file with an invalid name: {path.name}")
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
                raise ValueError("the conversation data has an invalid format")

            messages = data["messages"]
            if any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError("the conversation contains an invalid message")

            conversation_id = path.stem.lower()
            conversations[conversation_id] = {
                "id": conversation_id,
                "title": str(data.get("title") or "New chat"),
                "created_at": str(data.get("created_at") or now_utc()),
                "updated_at": str(data.get("updated_at") or now_utc()),
                "messages": messages,
            }
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"Could not load {path.name}: {error}")

    return conversations, errors


def save_conversation(conversation: dict[str, Any]) -> None:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAT_DIR / f"{conversation['id']}.json"
    path.write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def initialize_state() -> None:
    if "conversations" in st.session_state:
        return

    conversations, errors = load_conversations()
    st.session_state.conversations = conversations
    st.session_state.chat_load_errors = errors
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    st.session_state.system_prompt_draft = DEFAULT_SYSTEM_PROMPT

    if conversations:
        st.session_state.active_chat_id = max(
            conversations,
            key=lambda chat_id: conversations[chat_id]["updated_at"],
        )
    else:
        conversation = new_conversation()
        conversations[conversation["id"]] = conversation
        st.session_state.active_chat_id = conversation["id"]
        try:
            save_conversation(conversation)
        except OSError as error:
            st.session_state.chat_load_errors.append(
                f"Could not save a new chat: {error}"
            )


def get_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        return str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def save_active_conversation() -> None:
    conversation = st.session_state.conversations[
        st.session_state.active_chat_id
    ]
    conversation["updated_at"] = now_utc()
    try:
        save_conversation(conversation)
    except OSError as error:
        st.error(f"Could not save this chat: {error}")


def start_new_chat() -> None:
    conversation = new_conversation()
    st.session_state.conversations[conversation["id"]] = conversation
    st.session_state.active_chat_id = conversation["id"]
    try:
        save_conversation(conversation)
    except OSError as error:
        st.error(f"Could not save the new chat: {error}")


def delete_active_chat() -> None:
    chat_id = st.session_state.active_chat_id
    try:
        (CHAT_DIR / f"{chat_id}.json").unlink(missing_ok=True)
    except OSError as error:
        st.error(f"Could not delete this chat: {error}")
        return

    del st.session_state.conversations[chat_id]
    if st.session_state.conversations:
        st.session_state.active_chat_id = max(
            st.session_state.conversations,
            key=lambda other_id: st.session_state.conversations[other_id][
                "updated_at"
            ],
        )
    else:
        start_new_chat()


def response_text_chunks(stream: Any):
    for chunk in stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            yield text


def request_reply(
    conversation: dict[str, Any],
    api_key: str,
    temperature: float,
    max_tokens: int,
    context_messages: int,
    streaming: bool,
) -> None:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        *conversation["messages"][-context_messages:],
    ]

    with st.chat_message("assistant"):
        try:
            if streaming:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                answer = st.write_stream(response_text_chunks(stream))
            else:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = response.choices[0].message.content
                if not answer:
                    st.error("The model returned an empty response.")
                    return
                st.markdown(answer)
        except OpenAIError as error:
            st.error(f"OpenRouter request failed: {error}")
            return

    if not answer:
        st.error("The model returned an empty response.")
        return

    conversation["messages"].append({"role": "assistant", "content": answer})
    save_active_conversation()


st.set_page_config(page_title="Mirvat_GPT", page_icon="💬")
initialize_state()

st.title("Mirvat_GPT")
st.caption(f"A beginner-friendly chat app powered by {MODEL} via OpenRouter.")

with st.sidebar:
    st.header("Chats")
    if st.button("＋ New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    ordered_chat_ids = sorted(
        st.session_state.conversations,
        key=lambda chat_id: st.session_state.conversations[chat_id]["updated_at"],
        reverse=True,
    )
    for chat_id in ordered_chat_ids:
        conversation = st.session_state.conversations[chat_id]
        label = conversation["title"] or "New chat"
        if st.button(
            label,
            key=f"select_{chat_id}",
            use_container_width=True,
            type=(
                "primary"
                if chat_id == st.session_state.active_chat_id
                else "secondary"
            ),
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    if st.button("Delete current chat", use_container_width=True):
        delete_active_chat()
        st.rerun()

    st.divider()
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Maximum output tokens", 128, 4096, 1024, 128)
    context_messages = st.slider("Recent messages sent as context", 2, 40, 12, 2)
    streaming = st.toggle("Stream responses", value=True)

    st.subheader("System prompt")
    st.text_area(
        "Edit the assistant's instructions",
        key="system_prompt_draft",
        height=140,
        label_visibility="collapsed",
    )
    if st.button("Save system prompt", use_container_width=True):
        st.session_state.system_prompt = st.session_state.system_prompt_draft
        st.success("System prompt saved for this session.")

    st.caption(
        "Chats are shared by everyone using this deployment. Streamlit Cloud "
        "may clear local chat files when the app restarts."
    )

for load_error in st.session_state.chat_load_errors:
    st.warning(load_error)

active_conversation = st.session_state.conversations[
    st.session_state.active_chat_id
]
for message in active_conversation["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Send a message")
if prompt:
    user_message = {"role": "user", "content": prompt}
    active_conversation["messages"].append(user_message)
    if active_conversation["title"] == "New chat":
        active_conversation["title"] = prompt.strip().replace("\n", " ")[:40]
        if len(prompt.strip()) > 40:
            active_conversation["title"] += "..."

    with st.chat_message("user"):
        st.markdown(prompt)
    save_active_conversation()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY in "
            "Streamlit secrets or your environment, then try again."
        )
    else:
        request_reply(
            active_conversation,
            api_key,
            temperature,
            max_tokens,
            context_messages,
            streaming,
        )
