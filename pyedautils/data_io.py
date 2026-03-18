import time
import os
import gzip
import logging
import pickle
import json
from typing import Any, Optional, Union

import pandas as pd
import math

logger = logging.getLogger(__name__)


def save_data(data: Any, file_path: str, index: bool = False) -> None:
    """
    Saves data to a file in various formats (CSV, Pickle, JSON) based on the
    given file extension of file_path.

    Args:
        data: The data to be saved. Should be a pandas DataFrame or a Python object.
        file_path (str): The path to the file where the data will be saved.
        index (bool, optional): Whether to write the DataFrame index. Default is False.

    Raises:
        ValueError: If the file format is not supported.

    Returns:
        None
    """
    start_time = time.time()

    try:
        # Create directory if it doesn't exist
        if os.path.dirname(file_path):
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
                logger.info("Directory did not exist, created it now")

        logger.info("Saving data to: %s", file_path)
        if file_path.endswith(".csv"):
            data.to_csv(file_path, index=index)
        elif file_path.endswith(".pklz"):
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif file_path.endswith(".pkl"):
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif file_path.endswith(".json"):
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:
            raise ValueError("Unsupported file format")

        logger.info(" -> Done, elapsed time: %.3f seconds", time.time() - start_time)
        logger.info(" -> File size in bytes: %d", math.trunc(os.path.getsize(file_path) / 1024))

    except Exception as e:
        logger.error("Error: %s", e)  # pragma: no cover


def load_data(file_path: str) -> Optional[Union[pd.DataFrame, Any]]:
    """
    Loads data from a file in various formats (CSV, Pickle, JSON) based on the
    given file extension of file_path.

    Args:
        file_path (str): The path to the file from which data will be loaded.

    Raises:
        ValueError: If the file format is not supported.

    Returns:
        data: The loaded data as a pandas DataFrame or a Python object.
    """
    start_time = time.time()

    try:
        logger.info("Loading data from: %s", file_path)
        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path, sep=None, engine='python')
        elif file_path.endswith(".pklz"):
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.endswith(".pkl"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format")

        logger.info(" -> Done, elapsed time: %.3f seconds", time.time() - start_time)
        logger.info(" -> File size in bytes: %d", math.trunc(os.path.getsize(file_path) / 1024))

        return data

    except Exception as e:
        logger.error("Error: %s", e)  # pragma: no cover
        return None
