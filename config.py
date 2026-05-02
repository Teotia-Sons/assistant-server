import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")

    MONGO_URI = os.getenv("MONGO_URI")

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")

    GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")

    GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    OTEL_EXPORTER_OTLP_API_KEY = os.getenv("OTEL_EXPORTER_OTLP_API_KEY")
