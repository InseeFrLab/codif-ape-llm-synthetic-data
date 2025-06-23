import os

import hydra
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import OpenAI
from omegaconf import DictConfig

from src.config import setup_langfuse
from src.output_models import BiasType, create_llm_response_model
from src.prompts import BIAS_INSTRUCTIONS
from src.request_API import accuracy_score, purity_score, query_batchAPI
from src.utils import get_df_naf

# Load environment variables from .env file if it exists
load_dotenv()

setup_langfuse()
client = OpenAI(
    base_url="https://vllm-generation.user.lab.sspcloud.fr/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def ask_llm(
    nace_code: str,
    model_name: str,
    expected_list_size: int,
    bias_type: str,
    revision: str = "NAF2008",
    temperature: float = 0.8,
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

    if revision not in ["NAF2008", "NAF2025"]:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df_naf = get_df_naf(revision=revision)[["APE_NIV5", "LIB_NIV5"]]

    if nace_code not in df_naf["APE_NIV5"].values:
        raise ValueError(f"NACE code {nace_code} not found in the dataset.")
    code_description = df_naf.loc[df_naf["APE_NIV5"] == nace_code, "LIB_NIV5"].values[0]

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
            temperature=temperature,
            max_tokens=500,
            response_format=LLMResponse,
        )
        .choices[0]
        .message.parsed
    )

    return response


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that uses Hydra configuration."""

    nace_code = cfg.nace_code
    model_name = cfg.model_name
    expected_list_size = cfg.expected_list_size
    bias_type = cfg.bias_type
    revision = cfg.get("revision")
    temperature = cfg.get("temperature")
    nb_echos_max = cfg.get("nb_echos_max")

    response = ask_llm(
        nace_code=nace_code,
        model_name=model_name,
        expected_list_size=expected_list_size,
        bias_type=bias_type,
        revision=revision,
        temperature=temperature,
    )
    predictions = query_batchAPI(response, revision=revision, nb_echos_max=nb_echos_max)

    print(predictions)

    accuracies = accuracy_score(predictions, nace_code)
    print(f"Accuracies: {accuracies}")

    if bias_type == "Genre & Nombre" or bias_type == "Typo & Registre":
        purity = purity_score(predictions)
        print(f"Purity: {purity}")


if __name__ == "__main__":
    main()
