import base64
import os
from typing import List

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables from .env file if it exists
load_dotenv()

EXPECTED_LIST_SIZE = 5


class LLMResponse(BaseModel):
    """Structured response model for NACE synthetic data generation"""

    code: str = Field(description="The NACE code provided in input (echo back for verification)")
    activity_descriptions: List[str] = Field(
        min_items=EXPECTED_LIST_SIZE,
        max_items=EXPECTED_LIST_SIZE,
        description=f"Exactly {EXPECTED_LIST_SIZE} synthetic activity descriptions with applied bias patterns",
    )


def setup_langfuse():
    # Validate required environment variables
    required_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "OPENAI_API_KEY",
    ]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    # Build Langfuse Basic Auth header
    LANGFUSE_AUTH = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()

    # Set OTEL exporter environment variables
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ["LANGFUSE_HOST"] + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"


setup_langfuse()
client = OpenAI(
    base_url="https://llm.lab.sspcloud.fr/api", api_key=os.environ.get("OPENAI_API_KEY")
)

prompt = Langfuse().get_prompt(name="llm-synthetic-data", label="production")

print(
    prompt.compile(
        nace_code="8891A",
        code_description="Accueil de jeunes enfants",
        expected_count=EXPECTED_LIST_SIZE,
    )
)
client.beta.chat.completions.parse(
    model="gemma3:27b",
    messages=prompt.compile(
        nace_code="8891A",
        code_description="Accueil de jeunes enfants",
        expected_count=EXPECTED_LIST_SIZE,
    ),
    temperature=0.5,
    max_tokens=500,
    response_format=LLMResponse,
)
