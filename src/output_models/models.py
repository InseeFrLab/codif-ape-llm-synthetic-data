from enum import Enum
from typing import List, Type

from pydantic import BaseModel, Field


class BiasType(str, Enum):
    general = "Général"
    gender_and_number = "Genre & Nombre"
    typo_and_register = "Typo & Registre"


class ValidationBlock(BaseModel):
    count_check: str = Field(
        description="Validation message confirming the number of generated descriptions"
    )


def create_llm_response_model(expected_list_size: int) -> Type[BaseModel]:
    """
    Crée dynamiquement un modèle LLMResponse avec la taille de liste spécifiée
    """

    class LLMResponse(BaseModel):
        """Structured response model for NACE synthetic data generation"""

        code: str = Field(description="Le code NAF exact fourni en entrée")
        tested_bias: BiasType = Field(
            description="Tested bias: Général, Genre & Nombre ou Typo & Registre"
        )
        activity_descriptions: List[str] = Field(
            min_items=expected_list_size,
            max_items=expected_list_size,
            description=f"Exactement {expected_list_size} descriptions d'activité synthétiques avec biais appliqué",
        )
        validation: ValidationBlock

    return LLMResponse
