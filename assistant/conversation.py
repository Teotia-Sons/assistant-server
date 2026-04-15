import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from bson import ObjectId
from flask_login import current_user
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from assistant.models import get_model
from db.models.conversation import Conversation

SYSTEM_PROMPTS_DIR = Path(__file__).parent / "systemprompts"

type MessageType = Literal["system", "human", "ai"]


def _get_creation_time() -> str:
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat()


def get_conversation(conversation_id: str) -> dict | None:
    conversation = Conversation.objects(id=ObjectId(conversation_id)).first()
    if not conversation:
        return None
    return conversation.to_dict()


def list_conversations(query: str) -> list[dict]:
    mongo_query = Conversation.objects(username=current_user.id).exclude("messages")

    if query:
        mongo_query = mongo_query.filter(title__icontains=query)

    conversations = mongo_query.order_by("-created_at")

    return [conv.to_dict() for conv in conversations]


def _extract_text_from_content(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content

    text_parts = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif block.get("type") == "text":
            text_parts.append(block.get("text", ""))

    return "".join(text_parts)


def _get_selected_raw_content(
    content: str | list[str | dict], rendered_selection: str
) -> str:
    text = _extract_text_from_content(content)
    chars = [char for char in rendered_selection if char.isalnum()]
    pattern_str = r"\W*".join(re.escape(ch) for ch in chars)
    match = re.search(pattern_str, text, re.IGNORECASE)
    return match.group(0) if match else rendered_selection


def fork_conversation(
    source_conversation_id: str,
    message_id: str,
    target_conversation_id: str | None = None,
    prompt: str | None = None,
) -> dict:
    if target_conversation_id:
        if Conversation.objects(id=ObjectId(target_conversation_id)).first():
            raise ValueError(f"Conversation {target_conversation_id} already exists")

    conversation_messages = get_messages(source_conversation_id)
    if not conversation_messages:
        raise ValueError(f"Conversation {source_conversation_id} does not exist")

    for i, message in enumerate(conversation_messages):
        if message.id == message_id:
            forked_messages = conversation_messages[: i + 1]
            break
    else:
        raise ValueError(
            f"Message {message_id} not found in conversation {source_conversation_id}"
        )

    if prompt:
        last_message = forked_messages[-1]
        assert isinstance(last_message, AIMessage)
        creation_time = _get_creation_time()
        forked_messages[-1] = AIMessage(
            content=f"trimmed...\n\n{_get_selected_raw_content(last_message.content, prompt)}",
            id=last_message.id,
            additional_kwargs={"creation_time": creation_time},
        )

    return set_messages(target_conversation_id, forked_messages)


def trim_conversation(conversation_id: str, message_id: str) -> dict:
    messages = get_messages(conversation_id)
    if not messages:
        raise ValueError(f"Conversation {conversation_id} has no messages")

    for i, message in enumerate(messages):
        if message.id == message_id:
            messages = messages[: i + 1]
            return set_messages(conversation_id, messages)

    raise ValueError(
        f"Message {message_id} not found in conversation {conversation_id}"
    )


def get_messages(conversation_id: Optional[str]) -> list[BaseMessage]:
    if not conversation_id:
        return []
    conversation = Conversation.objects(id=ObjectId(conversation_id)).first()
    message_dicts = conversation.messages
    return messages_from_dict(message_dicts)


def get_next_message_id(messages: list[BaseMessage], message_type: MessageType) -> str:
    if message_type == "system":
        count = sum(1 for msg in messages if isinstance(msg, SystemMessage))
        return f"system-{count + 1}"
    if message_type == "human":
        count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        return f"human-{count + 1}"
    if message_type == "ai":
        count = sum(1 for msg in messages if isinstance(msg, AIMessage))
        return f"ai-{count + 1}"
    return f"{message_type}-{len(messages) + 1}"


def set_messages(conversation_id: str | None, messages: list[BaseMessage]) -> dict:
    if not conversation_id:
        conversation = Conversation(
            username=current_user.id,
            messages=[],
        )
    else:
        conversation = Conversation.objects(id=ObjectId(conversation_id)).first()
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
    conversation.messages = messages_to_dict(messages)
    conversation.save()
    return conversation.to_dict()


def append_message(
    conversation_id: Optional[str], message_type: MessageType, content: str
) -> dict:
    messages = get_messages(conversation_id)
    msg_id = get_next_message_id(messages, message_type)

    if message_type == "system":
        messages.append(
            SystemMessage(
                content=content,
                id=msg_id,
                additional_kwargs={"creation_time": _get_creation_time()},
            )
        )
    elif message_type == "human":
        messages.append(
            HumanMessage(
                content=content,
                id=msg_id,
                additional_kwargs={"creation_time": _get_creation_time()},
            )
        )
    else:
        raise ValueError(f"Unknown message type: {message_type}")

    return set_messages(conversation_id, messages)


def edit_message(conversation_id: str, message_id: str, new_content: str) -> dict:
    messages = get_messages(conversation_id)
    if not messages:
        raise ValueError(f"Conversation {conversation_id} has no messages")

    for i, message in enumerate(messages):
        if message.id == message_id:
            if not isinstance(message, (SystemMessage, HumanMessage)):
                raise ValueError(
                    f"Message {message_id} is not a system or human message"
                )
            messages[i].content = new_content
            return set_messages(conversation_id, messages)

    raise ValueError(
        f"Message {message_id} not found in conversation {conversation_id}"
    )


def _get_system_prompt(key: str) -> SystemMessage:
    prompt = (SYSTEM_PROMPTS_DIR / f"{key}.md").read_text()
    return SystemMessage(content=prompt)


def generate_title(conversation_id: str) -> None:
    system_msg = _get_system_prompt("title")

    messages = get_messages(conversation_id)
    formatted_content = "\n\n".join(
        f"{msg.type}:\n{_extract_text_from_content(msg.content)[:500]}"
        for msg in messages
    )
    human_msg = HumanMessage(content=formatted_content)

    model = get_model("GPT_OSS")
    response = model.invoke([system_msg, human_msg])

    response_data = json.loads(response.content)
    title = response_data["title"]

    conv_obj = Conversation.objects(id=ObjectId(conversation_id)).first()
    conv_obj.title = title
    conv_obj.save()


def delete_conversation(conversation_id: str) -> None:
    conversation = Conversation.objects(id=ObjectId(conversation_id)).first()
    conversation.delete()
