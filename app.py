import json
import os
import re
import time
import uuid
from collections.abc import Iterator
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
        "transactions": [],
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

            transactions = data.get("transactions", [])
            if not isinstance(transactions, list) or any(
                not isinstance(transaction, dict)
                or not isinstance(transaction.get("request"), dict)
                or not isinstance(transaction.get("response"), str)
                or not isinstance(transaction.get("status"), str)
                or not isinstance(transaction.get("timestamp"), str)
                for transaction in transactions
            ):
                raise ValueError("the conversation contains invalid transaction logs")

            conversation_id = path.stem.lower()
            conversations[conversation_id] = {
                "id": conversation_id,
                "title": str(data.get("title") or "New chat"),
                "created_at": str(data.get("created_at") or now_utc()),
                "updated_at": str(data.get("updated_at") or now_utc()),
                "messages": messages,
                "transactions": transactions,
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


def capture_usage(source: Any, usage: dict[str, int | None]) -> None:
    if source is None:
        return
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(source, field, None)
        if isinstance(value, int):
            usage[field] = value


def format_response_metrics(
    response_time: float,
    usage: dict[str, int | None] | None,
) -> str:
    if usage and isinstance(usage.get("total_tokens"), int):
        token_summary = f"{usage['total_tokens']} tokens"
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            token_summary += (
                f" ({prompt_tokens} prompt + {completion_tokens} completion)"
            )
    else:
        token_summary = "token usage unavailable"
    return f"Response time: {response_time:.2f}s · {token_summary}"


def response_text_chunks(
    stream: Any,
    usage: dict[str, int | None],
) -> Iterator[str]:
    for chunk in stream:
        capture_usage(getattr(chunk, "usage", None), usage)
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
    messages = [
        {"role": "system", "content": st.session_state.system_prompt},
        *conversation["messages"][-context_messages:],
    ]
    transaction = {
        "timestamp": now_utc(),
        "status": "pending",
        "request": {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": streaming,
        },
        "response": "",
    }
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    answer = ""
    request_error: OpenAIError | None = None
    started_at = time.perf_counter()

    with st.chat_message("assistant", avatar="🌊"):
        with st.spinner("thinking..."):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
                if streaming:
                    stream = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    answer = st.write_stream(response_text_chunks(stream, usage))
                else:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    answer = response.choices[0].message.content or ""
                    capture_usage(response.usage, usage)
                    if answer:
                        st.markdown(answer)
            except OpenAIError as error:
                request_error = error

        elapsed = time.perf_counter() - started_at
        if request_error:
            st.error(f"OpenRouter request failed: {request_error}")
            transaction["status"] = "error"
            transaction["response"] = "The request failed; see the error above."
            transaction["response_time_seconds"] = round(elapsed, 2)
            conversation["transactions"].append(transaction)
            save_active_conversation()
            return

        if not answer:
            st.error("The model returned an empty response.")
            transaction["status"] = "empty"
            transaction["response"] = "The model returned an empty response."
            transaction["response_time_seconds"] = round(elapsed, 2)
            conversation["transactions"].append(transaction)
            save_active_conversation()
            return

        st.caption(format_response_metrics(elapsed, usage))

    conversation["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "response_time_seconds": round(elapsed, 2),
            "usage": usage,
        }
    )
    transaction["status"] = "completed"
    transaction["response"] = answer
    transaction["response_time_seconds"] = round(elapsed, 2)
    transaction["usage"] = usage
    conversation["transactions"].append(transaction)
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
avatar_by_role = {"user": "🧑‍🎓", "assistant": "🌊"}
for message in active_conversation["messages"]:
    with st.chat_message(
        message["role"],
        avatar=avatar_by_role[message["role"]],
    ):
        st.markdown(message["content"])
        if message["role"] == "assistant" and isinstance(
            message.get("response_time_seconds"), (int, float)
        ):
            st.caption(
                format_response_metrics(
                    message["response_time_seconds"],
                    message.get("usage"),
                )
            )

suggested_prompt = None
if not active_conversation["messages"]:
    st.markdown("## Welcome to Mirvat_GPT")
    st.write("What would you like to learn or work on today?")
    example_prompts = [
        "📚 Explain a difficult topic in simple terms",
        "🗓️ Help me plan a focused study session",
        "🧠 Quiz me on something I am learning",
        "✍️ Help me improve a paragraph I wrote",
    ]
    prompt_columns = st.columns(2)
    for index, example in enumerate(example_prompts):
        if prompt_columns[index % 2].button(
            example,
            key=f"example_{active_conversation['id']}_{index}",
            use_container_width=True,
        ):
            suggested_prompt = example

prompt = st.chat_input("Send a message")
if suggested_prompt:
    prompt = suggested_prompt
if prompt:
    user_message = {"role": "user", "content": prompt}
    active_conversation["messages"].append(user_message)
    if active_conversation["title"] == "New chat":
        active_conversation["title"] = prompt.strip().replace("\n", " ")[:40]
        if len(prompt.strip()) > 40:
            active_conversation["title"] += "..."

    with st.chat_message("user", avatar="🧑‍🎓"):
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

transactions = active_conversation["transactions"]
with st.expander(f"Transaction logs ({len(transactions)})"):
    if not transactions:
        st.caption("No request/response transactions yet.")

    request_tab, response_tab = st.tabs(["Request sent", "Response"])
    with request_tab:
        if transactions:
            for transaction in reversed(transactions):
                st.json(
                    {
                        "timestamp": transaction["timestamp"],
                        "status": transaction["status"],
                        "request": transaction["request"],
                    }
                )
        else:
            st.caption("No requests to display.")

    with response_tab:
        if transactions:
            for transaction in reversed(transactions):
                st.json(
                    {
                        "timestamp": transaction["timestamp"],
                        "status": transaction["status"],
                        "response": transaction["response"],
                    }
                )
        else:
            st.caption("No responses to display.")
