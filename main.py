import logging
import os
import time

import hydra
import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import OpenAI
from omegaconf import DictConfig
from pydantic import ValidationError

from src.config import setup_langfuse
from src.output_models import BiasType, create_llm_response_model
from src.prompts import BIAS_INSTRUCTIONS
from src.request_API import accuracy_score, purity_score, query_batchAPI
from src.utils import get_df_naf, get_file_system

# Load environment variables from .env file if it exists
load_dotenv()

logger = logging.getLogger(__name__)

setup_langfuse()
client = OpenAI(
    base_url="https://vllm-generation.user.lab.sspcloud.fr/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

MAX_RETRIES = 10
RETRY_DELAY = 1  # seconds


def ask_llm(
    df_naf: pd.DataFrame,
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

    if nace_code not in df_naf["APE_NIV5"].values:
        raise ValueError(f"NACE code {nace_code} not found in the dataset.")
    code_description = df_naf.loc[df_naf["APE_NIV5"] == nace_code, "LIB_NIV5"].values[0]

    bias_type = BiasType(bias_type).value  # validation
    LLMResponse = create_llm_response_model(expected_list_size)

    prompt = Langfuse().get_prompt(name="llm-synthetic-data", label="production")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
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

        except ValidationError as e:
            print(f"[Attempt {attempt}] Validation failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise ValueError(f"All {MAX_RETRIES} attempts failed to produce a valid output.")


def run_and_compute_metric(
    df_naf: pd.DataFrame,
    nace_code: str,
    model_name: str,
    expected_list_size: int,
    bias_type: str,
    revision: str = "NAF2008",
    temperature: float = 0.8,
    nb_echos_max: int = 3,
):
    response = ask_llm(
        df_naf=df_naf,
        nace_code=nace_code,
        model_name=model_name,
        expected_list_size=expected_list_size,
        bias_type=bias_type,
        revision=revision,
        temperature=temperature,
    )

    if response.code != nace_code:
        logger.warning(
            f"LLM response code {response.code} does not match expected NACE code {nace_code}."
        )

    predictions = query_batchAPI(response, revision=revision, nb_echos_max=nb_echos_max)

    accuracies = accuracy_score(predictions, nace_code)

    purity = purity_score(predictions)

    return response, predictions, accuracies, purity


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    run_type = cfg.run_type
    model_name = cfg.model_name
    expected_list_size = cfg.expected_list_size
    nb_echos_max = cfg.get("nb_echos_max")
    revision = cfg.get("revision")
    temperature = cfg.get("temperature")

    df_naf = get_df_naf(revision=revision)[["APE_NIV5", "LIB_NIV5"]]

    if run_type not in ["all", "single"]:
        raise ValueError("run_type must be either 'all' or 'single'.")

    if run_type == "all":
        df_res = pd.DataFrame(
            columns=[
                "NACE Code",
                "Bias Type",
                "Expected List Size",
                "Model Name",
                "Revision",
                "LLM Response",
                "Predictions",
            ]
        )
        for code in df_naf["APE_NIV5"]:
            for bias_type in BiasType:
                logger.info(f"Processing NACE code {code} with bias type {bias_type.value}...")
                llm_response, predictions, accuracies, purity = run_and_compute_metric(
                    df_naf=df_naf,
                    nace_code=code,
                    model_name=model_name,
                    expected_list_size=expected_list_size,
                    bias_type=bias_type.value,
                    revision=revision,
                    temperature=temperature,
                    nb_echos_max=nb_echos_max,
                )

                df_res = pd.concat(
                    [
                        df_res,
                        pd.DataFrame(
                            {
                                "NACE Code": code,
                                "Bias Type": bias_type.value,
                                "Expected List Size": expected_list_size,
                                "Model Name": model_name,
                                "Revision": revision,
                                "LLM Response": llm_response.activity_descriptions,
                                "Temperature": temperature,
                                "Predictions": [
                                    [pred[i] for pred in predictions]
                                    for i in range(
                                        len(predictions[0])
                                    )  # list of top_k pred for each libelle
                                ],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                logger.info(f"Accuracy Scores: {accuracies}")
                logger.info(f"Purity Scores: {purity}")
                logger.info(f"\n{df_res.head().to_string(index=False)}")

        # Save the results to a parquet file
        fs = get_file_system()
        date = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"projet-ape/synthetic_data_test/{date}.parquet"
        with fs.open(file_path, "wb") as f:
            df_res.to_parquet(f)

    else:
        nace_code = cfg.nace_code
        bias_type = cfg.bias_type

        llm_response, predictions, accuracies, purity = run_and_compute_metric(
            df_naf=df_naf,
            nace_code=nace_code,
            model_name=model_name,
            expected_list_size=expected_list_size,
            bias_type=bias_type,
            revision=revision,
            temperature=temperature,
            nb_echos_max=nb_echos_max,
        )

        df_res = pd.DataFrame(
            {
                "NACE Code": nace_code,
                "Bias Type": bias_type,
                "Expected List Size": expected_list_size,
                "Model Name": model_name,
                "Revision": revision,
                "LLM Response": llm_response.activity_descriptions,
                "Predictions": [
                    [pred[i] for pred in predictions]
                    for i in range(len(predictions[0]))  # list of top_k pred for each libelle
                ],
            }
        )

        logger.info(f"Accuracy Scores: {accuracies}")
        logger.info(f"Purity Scores: {purity}")
        logger.info(f"\n{df_res.to_string(index=False)}")


if __name__ == "__main__":
    main()
