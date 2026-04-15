from dataclasses import dataclass
from typing import Literal, Optional

from langchain_cerebras import ChatCerebras
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from xai_sdk.chat import ReasoningEffort

from config import Config

type ModelTag = Literal["GEMINI_PRO", "GEMINI_FLASH", "GPT_OSS"]


@dataclass
class ModelConfig:
    model: ModelTag
    reasoning_effort: Optional[ReasoningEffort]


MODEL_PRICING = {
    "gemini-3.1-pro-preview": {
        "input_per_million": 2.0,
        "output_per_million": 12.0,
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
    if model_tag == "GEMINI_FLASH":
        model = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
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
    raise ValueError(f"Unknown model tag: {model_tag}")
