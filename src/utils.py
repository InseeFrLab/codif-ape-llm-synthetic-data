import os

import pandas as pd
import pyarrow.parquet as pq
from s3fs import S3FileSystem


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def get_df_naf(
    revision: str,
) -> pd.DataFrame:
    """
    Get detailed NAF data (lvl5).

    Args:
        path (str): Path to the data.

    Returns:
        pd.DataFrame: Detailed NAF data.
    """
    fs = get_file_system()

    if revision == "NAF2008":
        path = "projet-ape/data/naf2008_extended.parquet"
    elif revision == "NAF2025":
        path = "projet-ape/data/naf2025_extended.parquet"
    else:
        raise ValueError("Revision must be either 'NAF2008' or 'NAF2025'.")

    df = pq.read_table(path, filesystem=fs).to_pandas()

    return df
