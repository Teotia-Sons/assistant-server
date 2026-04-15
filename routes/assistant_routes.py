import json
from typing import Optional

from flask import Blueprint, Response, request
from xai_sdk.chat import ReasoningEffort

from assistant.conversation import (
    append_message,
    delete_conversation,
    edit_message,
    fork_conversation,
    get_conversation,
    list_conversations,
    trim_conversation,
)
from assistant.models import ModelConfig
from assistant.response import generate_response

assistant_bp = Blueprint("assistant_bp", __name__)


def translate_reasoning_effort(
    reasoning_effort: str,
) -> Optional[ReasoningEffort]:
    if reasoning_effort == "low":
        return "low"
    elif reasoning_effort == "high":
        return "high"

    return None


@assistant_bp.route("/conversation/<conversation_id>", methods=["GET"])
def handle_get_conversation(conversation_id):
    conversation = get_conversation(conversation_id)
    return {"conversation": conversation} if conversation else {}


@assistant_bp.route("/conversation/<conversation_id>", methods=["DELETE"])
def handle_delete_conversation(conversation_id):
    delete_conversation(conversation_id)
    return "", 204


@assistant_bp.route("/conversations", methods=["GET"])
def handle_list_conversations():
    query = request.args["query"]
    return {"conversations": list_conversations(query)}


@assistant_bp.route("/append", methods=["POST"])
def handle_append_message():
    data = request.json
    conversation_id = data.get("conversation_id")
    content = data["content"]
    message_type = data["message_type"]
    return append_message(conversation_id, message_type, content)


@assistant_bp.route("/invoke", methods=["POST"])
def handle_invoke():
    data = request.json
    conversation_id = data["conversation_id"]
    model_config = ModelConfig(
        model=data["model"],
        reasoning_effort=translate_reasoning_effort(data.get("reasoning_effort")),
    )

    def generate_sse_stream():
        for chunk in generate_response(conversation_id, model_config):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(
        generate_sse_stream(),
        content_type="text/event-stream",
    )


@assistant_bp.route("/edit", methods=["POST"])
def handle_edit_message():
    data = request.json
    conversation_id = data["conversation_id"]
    message_id = data["message_id"]
    content = data["content"]
    conversation = edit_message(conversation_id, message_id, content)
    return conversation


@assistant_bp.route("/fork", methods=["POST"])
def handle_fork_conversation():
    data = request.json
    source = data["source"]
    message_id = data["message_id"]
    target = data.get("target")
    prompt = data.get("prompt")
    return fork_conversation(source, message_id, target, prompt)


@assistant_bp.route("/trim", methods=["POST"])
def handle_trim_conversation():
    data = request.json
    conversation_id = data["conversation_id"]
    message_id = data["message_id"]
    return trim_conversation(conversation_id, message_id)
