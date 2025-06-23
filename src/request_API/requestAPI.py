from typing import List
from urllib.parse import urlencode

import numpy as np
import requests


def query_batchAPI(response, revision, nb_echos_max=3):
    """
    Query the batch API for NAF codes based on activity descriptions given by the LLM.
    Args:
        response: The response object containing activity descriptions.
        revision (str): The NAF revision to use, either "NAF2008" or "NAF2025".
        nb_echos_max (int): The maximum number of prediction levels to return
                            for each activity description.
    Returns:
        predictions (List[List[str]]):
            One list for each level i in 1...nb_echos_max, containing the predicted NAF code at
            level i for each activity description.
    """

    if revision not in ["NAF2008", "NAF2025"]:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    if revision == "NAF2008":
        base_url = "https://codification-ape2008-pytorch.lab.sspcloud.fr/predict"
    else:
        base_url = "https://codification-ape2025-pytorch.lab.sspcloud.fr/predict"

    batch = response.activity_descriptions

    json_data = {"forms": [{"description_activity": activity} for activity in batch]}

    params = {"nb_echos_max": nb_echos_max}
    url = f"{base_url}?{urlencode(params)}"

    response_api = requests.post(url, json=json_data)

    if response_api.status_code == 200:
        response_api = response_api.json()

        # For each level, extract the predicted NAF codes
        predictions = [
            [pred[str(i)]["code"] for pred in response_api] for i in range(1, nb_echos_max + 1)
        ]
        return predictions
    elif response_api.status_code == 400:
        print(response_api.json()["detail"])
    else:
        print(response_api.status_code)
        print("Error occurred while querying the API.")
        return None


def accuracy_score(predictions: List[List[str]], expected_code: str):
    # Cumulate accuracy over levels
    return np.cumsum(
        [sum([code == expected_code for code in pred]) / len(pred) for pred in predictions]
    )


def purity_score(predictions: List[List[str]]):
    # For each level, a metric of how the predicted codes are similar
    # 1 is perfectly pure (same code for all descriptions of the batch), 0 is completely impure (all have different codes)
    # Should be used when we expected the same code for all descriptions of the batch (variation mode such as Gender or Typo)
    return [
        np.round((len(pred) - len(np.unique(pred))) / (len(pred) - 1), 2) for pred in predictions
    ]
