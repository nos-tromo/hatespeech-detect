import logging
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from modules.ollama_cfg import (
    call_ollama_server,
    load_ollama_model,
    load_prompt
)
from utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Environment and file paths
# Set up environment variables in a .env file or export them in your shell
load_dotenv()
HOME_DIR: Path = Path.home()
DATA_DIR: str = os.getenv("DATA_DIR", "")
FILE: str = os.getenv("FILE", "")

if not DATA_DIR or not FILE:
    raise ValueError("Both DATA_DIR and FILE environment variables must be set")

DATA: Path = HOME_DIR / DATA_DIR / FILE
if not DATA.is_file():
    raise FileNotFoundError(f"Input data file '{DATA}' not found")

RESULTS: Path = DATA.with_name(f"{DATA.stem}_hatespeech.csv")


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the input data from a CSV file.

    Args:
        file_path (Path): The path to the input CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path, encoding="utf-8")


def _construct_prompt(text: str, language: str = "German") -> str:
    """
    Load and format the prompt.

    Args:
        text (str): The text to analyze.
        language (str, optional): The language of the text. Defaults to "German".

    Returns:
        str: The formatted prompt for the hate speech detection model.
    """
    return load_prompt("hate").format(language=language, text=text)


def _parse_binary_label(resp: str) -> int:
    """
    Parse the model response to extract a binary label (0 or 1).

    Args:
        resp (str): The raw response from the model.

    Returns:
        int: The parsed binary label (0 or 1), or -1 if parsing failed.
    """
    if resp is None:
        logger.error("Ollama returned None for label")
        return -1

    # Normalize to first line, strip quotes/whitespace
    text = str(resp).splitlines()[0].strip().strip("\"'")

    # Strict match: exactly one char 0 or 1
    m = re.fullmatch(r"[01]", text)
    if m:
        return int(text)

    # If the model emitted extra tokens (e.g., 'label: 1'), try to find a lone 0/1 token
    tokens = text.replace(":", " ").replace(",", " ").split()
    for tok in tokens:
        if re.fullmatch(r"[01]", tok):
            return int(tok)

    logger.error("Non-binary Ollama response: %r", resp)
    return -1


def run_inference(text: str, model: str | None, keyword: str = "hate") -> int:
    """
    Run inference on the given text using the Ollama model.

    Args:
        text (str): The text to analyze for hate speech.
        model (str | None): The name of the Ollama model to use (or None).

    Returns:
        int: The hate speech detection label (0 or 1), or -1 if detection failed.
    """
    prompt = _construct_prompt(text)
    raw_response = call_ollama_server(model=model, prompt=prompt, think=False)
    return _parse_binary_label(raw_response)


def store_output(df: pd.DataFrame, file_path: Path = RESULTS) -> None:
    """
    Store the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (Path, optional): The path to the output CSV file. Defaults to RESULTS.
    """
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Results saved to '{file_path}'")


def main() -> None:
    """
    Main function to load data, run hate speech detection, and store results.
    """
    df = load_data(DATA)

    # Pre-create result column as nullable integer to avoid float coercion
    df["class"] = pd.Series([pd.NA] * len(df), dtype="Int8")
    model = load_ollama_model()
    for row in df.iterrows():
        print(f"Processing row {row[0]}: {row[1]['text'][:50]}...")
        label = run_inference(text=row[1]["text"], model=model)
        if label in (0, 1):
            df.at[row[0], "class"] = int(label)
        else:
            # keep as <NA> instead of writing -1, preserves Int8 dtype
            df.at[row[0], "class"] = pd.NA
    
    store_output(df)


if __name__ == "__main__":
    main()
