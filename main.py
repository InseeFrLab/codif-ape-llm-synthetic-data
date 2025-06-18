import os

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import OpenAI

from src.config import setup_langfuse
from src.output_models import BiasType, create_llm_response_model
from src.prompts import BIAS_INSTRUCTIONS

# Load environment variables from .env file if it exists
load_dotenv()

setup_langfuse()
client = OpenAI(
    base_url="https://vllm-generation.user.lab.sspcloud.fr/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def ask_llm(
    nace_code: str = "8891A",
    code_description: str = "Accueil de jeunes enfants",
    model_name: str = "mistralai/Mistral-Small-24B-Instruct-2501",
    expected_list_size: int = 10,
    bias_type: str = "Général",
    temprature: float = 0.8,
):
    """
    Ask the LLM to generate synthetic data for a given NACE code with specified bias.

    Args:
        nace_code (str): The NACE code for which to generate activity descriptions.
        code_description (str): A brief description of the NACE code.
        model_name (str): The name of the model to use for generation.
        expected_list_size (int): The expected number of activity descriptions to generate.
        bias_type (str): The type of bias to apply. Must be one of BiasType values (Général, Genre & Nombre, Typo & Registre).
        temperature (float): The temperature setting for the LLM generation, controlling randomness.

    Returns:
        LLMResponse: A structured response containing the generated activity descriptions and validation information.
    """

    bias_type = BiasType(bias_type).value  # validation
    LLMResponse = create_llm_response_model(expected_list_size)

    prompt = Langfuse().get_prompt(name="llm-synthetic-data", label="production")

    response = (
        client.beta.chat.completions.parse(
            model=model_name,
            messages=prompt.compile(
                nace_code=nace_code,
                code_description=code_description,
                expected_count=expected_list_size,
                bias=bias_type,
                bias_instructions=BIAS_INSTRUCTIONS[bias_type],
            ),
            temperature=temprature,
            max_tokens=500,
            response_format=LLMResponse,
        )
        .choices[0]
        .message.parsed
    )

    return response


if __name__ == "__main__":
    nace_code = "8891A"
    code_description = "Accueil de jeunes enfants"
    model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
    bias_type = "Général"
    expected_list_size = 10

    response = ask_llm(nace_code, code_description, model_name, expected_list_size, bias_type)
    print(response)
