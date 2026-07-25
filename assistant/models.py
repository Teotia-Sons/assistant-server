from dataclasses import dataclass
from typing import Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_cerebras import ChatCerebras
from langchain_core.language_models import BaseChatModel
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from xai_sdk.chat import ReasoningEffort

from config import Config

type ModelTag = Literal[
    "GEMINI_PRO",
    "GPT_OSS",
    "GROK",
    "OPUS",
    "FABLE",
    "GPT",
    "GPT_LUNA",
    "GLM",
]


@dataclass
class ModelConfig:
    model: ModelTag
    reasoning_effort: Optional[ReasoningEffort]


def get_model(
        model_tag: ModelTag, reasoning_effort: Optional[ReasoningEffort] = None
) -> BaseChatModel:
    if model_tag == "GEMINI_PRO":
        model = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro-preview",
            google_api_key=Config.GOOGLE_GENAI_API_KEY,
            thinking_level=reasoning_effort,
            include_thoughts=True,
        ).bind_tools([{"google_search": {}}])
        return model
    if model_tag == "GPT_OSS":
        return ChatCerebras(
            model="gpt-oss-120b",
            api_key=Config.CEREBRAS_API_KEY,
            reasoning_effort=reasoning_effort,
        )
    if model_tag == "GROK":
        assert reasoning_effort is None
        return ChatXAI(
            model="grok-4.5",
            api_key=Config.XAI_API_KEY,
            extra_body={"include": ["reasoning.encrypted_content"]},
        )
    if model_tag == "OPUS":
        assert reasoning_effort is None
        return ChatAnthropic(
            model="claude-opus-5",
            api_key=Config.ANTHROPIC_API_KEY,
            thinking={"type": "adaptive", "display": "summarized"},
        ).bind(cache_control={"type": "ephemeral", "ttl": "1h"})
    if model_tag == "FABLE":
        assert reasoning_effort is None
        return ChatAnthropic(
            model="claude-fable-5",
            api_key=Config.ANTHROPIC_API_KEY,
            thinking={"type": "adaptive", "display": "summarized"},
        ).bind(cache_control={"type": "ephemeral", "ttl": "1h"})
    if model_tag == "GPT":
        return ChatOpenAI(
            model="gpt-5.6",
            api_key=Config.OPENAI_API_KEY,
            reasoning_effort=reasoning_effort,
            use_responses_api=True,
        )
    if model_tag == "GPT_LUNA":
        return ChatOpenAI(
            model="gpt-5.6-luna",
            api_key=Config.OPENAI_API_KEY,
            reasoning_effort=reasoning_effort,
            use_responses_api=True,
        )
    if model_tag == "GLM":
        model_kwargs = {}
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        return ChatFireworks(
            model="accounts/fireworks/models/glm-5p2",
            api_key=Config.FIREWORKS_API_KEY,
            model_kwargs=model_kwargs,
        )
    raise ValueError(f"Unknown model tag: {model_tag}")
