import logging
import os
import threading
from datetime import datetime
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage

from .conversation import (
    generate_title,
    get_messages,
    get_next_message_id,
    set_messages,
)
from .models import MODEL_PRICING, ModelConfig, ModelTag, get_model

logger = logging.getLogger(__name__)


def annotate_ai_message(
    ai_message: AIMessage, model_tag: ModelTag, invocation_time: datetime
) -> AIMessage:
    creation_time = datetime.now().astimezone()
    latency = (creation_time - invocation_time).total_seconds()

    ai_message.additional_kwargs = {
        **ai_message.additional_kwargs,
        "invocation_time": invocation_time.isoformat(),
        "creation_time": creation_time.isoformat(),
        "latency": latency,
    }

    model_name = ai_message.response_metadata.get("model_name")
    if model_name not in MODEL_PRICING:
        return ai_message

    pricing = MODEL_PRICING[model_name]
    usage_metadata = ai_message.usage_metadata
    if not usage_metadata:
        return ai_message

    input_tokens = usage_metadata["input_tokens"]
    output_tokens = usage_metadata["output_tokens"]

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    total_cost = input_cost + output_cost

    ai_message.additional_kwargs = {
        **ai_message.additional_kwargs,
        "cost": {
            "input": input_cost,
            "output": output_cost,
            "total": total_cost,
        },
    }

    return ai_message


def _get_system_prompt(key: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "systemprompts", f"{key}.md")
    with open(prompt_path, "r") as f:
        return f.read()


def generate_response(
    conversation_id: str, model_config: ModelConfig
) -> Generator[dict, None, None]:
    messages = get_messages(conversation_id)
    if not messages:
        raise ValueError(f"Conversation {conversation_id} has no messages.")

    model_tag = model_config.model
    model = get_model(model_tag, model_config.reasoning_effort)
    invocation_time = datetime.now().astimezone()
    stream = model.stream(messages)

    ai_message = None
    for chunk in stream:
        ai_message = chunk if not ai_message else ai_message + chunk
        ai_message.type = "ai"
        yield {"type": "message_chunk", "data": chunk.model_dump()}

    try:
        ai_message = annotate_ai_message(ai_message, model_tag, invocation_time)
    except Exception:
        logger.warning("Failed to annotate AI message cost", exc_info=True)

    ai_message.id = get_next_message_id(messages, "ai")
    messages.append(ai_message)
    conversation = set_messages(conversation_id, messages)
    yield {"type": "conversation", "data": conversation}

    threading.Thread(target=generate_title, args=(conversation_id,)).start()
