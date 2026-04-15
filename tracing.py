from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import Config


def setup_tracing(app):
    if not Config.OTEL_EXPORTER_OTLP_ENDPOINT:
        return

    tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": "assistant-server"})
    )
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{Config.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces",
                headers={"x-api-key": Config.OTEL_EXPORTER_OTLP_API_KEY},
            )
        )
    )
    trace.set_tracer_provider(tracer_provider)

    FlaskInstrumentor().instrument_app(app)
    LangchainInstrumentor().instrument()
    PymongoInstrumentor().instrument()
