import logging
import os
import threading
from datetime import datetime
from typing import Generator

from langchain_core.messages import AIMessage, UsageMetadata

from .conversation import (
    generate_title,
    get_messages,
    get_next_message_id,
    set_messages,
)
from .models import ModelConfig, get_model

logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "grok-4.3": {
        "input_per_million": 1.25,
        "cached_per_million": 0.20,
        "output_per_million": 2.50,
    },
    "gemini-3.1-pro-preview": {
        "input_per_million": 2.0,
        "output_per_million": 12.0,
    },
    "claude-opus-4-8": {
        "input_per_million": 6.5,
        "cached_per_million": 0.50,
        "output_per_million": 25.0,
    },
    "gpt-5.5-2026-04-23": {
        "input_per_million": 5.0,
        "cached_per_million": 2.50,
        "output_per_million": 30.0,
    },
    "gpt-oss-120b": {
        "input_per_million": 0.35,
        "output_per_million": 0.75,
    },
    "gemini-3-flash-preview": {
        "input_per_million": 0.50,
        "output_per_million": 3.0,
    },
}


def calculate_pricing(model_name: str, usage_metadata: UsageMetadata) -> dict:
    pricing = MODEL_PRICING[model_name]
    input_tokens = usage_metadata["input_tokens"]
    output_tokens = usage_metadata["output_tokens"]

    cached_tokens = usage_metadata.get("input_token_details", {}).get("cache_read", 0)
    cached_cost = (cached_tokens / 1_000_000) * pricing.get("cached_per_million", 0)

    non_cached_input_tokens = input_tokens - cached_tokens
    input_cost = (non_cached_input_tokens / 1_000_000) * pricing["input_per_million"]

    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]

    return {
        "cached": cached_cost,
        "input": input_cost,
        "output": output_cost,
        "total": input_cost + output_cost + cached_cost,
    }


def annotate_ai_message(ai_message: AIMessage, invocation_time: datetime) -> AIMessage:
    creation_time = datetime.now().astimezone()
    latency = (creation_time - invocation_time).total_seconds()

    ai_message.additional_kwargs = {
        **ai_message.additional_kwargs,
        "invocation_time": invocation_time.isoformat(),
        "creation_time": creation_time.isoformat(),
        "latency": latency,
    }

    usage_metadata = ai_message.usage_metadata
    model_name = ai_message.response_metadata.get("model_name")
    if not usage_metadata or model_name not in MODEL_PRICING:
        return ai_message

    ai_message.additional_kwargs = {
        **ai_message.additional_kwargs,
        "cost": calculate_pricing(model_name, usage_metadata),
    }
    return ai_message


def _get_system_prompt(key: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "systemprompts", f"{key}.md")
    with open(prompt_path, "r") as f:
        return f.read()


def _clean_ai_message(ai_message: AIMessage) -> AIMessage:
    if not isinstance(ai_message.content, list):
        return ai_message
    ai_message.content = [
        block for block in ai_message.content
        if not (isinstance(block, str) and not block.strip())
    ]
    return ai_message


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
        ai_message = annotate_ai_message(ai_message, invocation_time)
    except Exception:
        logger.warning("Failed to annotate AI message cost", exc_info=True)

    ai_message.id = get_next_message_id(messages, "ai")
    ai_message = _clean_ai_message(ai_message)
    messages.append(ai_message)
    conversation = set_messages(conversation_id, messages)
    yield {"type": "conversation", "data": conversation}

    threading.Thread(target=generate_title, args=(conversation_id,)).start()
