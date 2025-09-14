import logging
import re
from pathlib import Path

import pandas as pd

from modules.ollama_cfg import call_ollama_server, hate_detection_prompt
from utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA = DATA_DIR / ""
RESULTS = DATA_DIR / ""


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the input data from a CSV file.

    Args:
        file_path (Path): The path to the input CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path, encoding="utf-8")


def _load_prompt(text: str, language: str = "German") -> str:
    """
    Load and format the hate speech detection prompt.

    Args:
        text (str): The text to analyze.
        language (str, optional): The language of the text. Defaults to "German".

    Returns:
        str: The formatted prompt for the hate speech detection model.
    """
    return hate_detection_prompt.format(language=language, text=text)


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

    logger.error(f"Non-binary Ollama response: {resp!r}")
    return -1


def run_hate_speech_detection(text: str) -> int:
    """
    Run hate speech detection on the given text using the Ollama model.

    Args:
        text (str): The text to analyze for hate speech.

    Returns:
        int: The hate speech detection label (0 or 1), or -1 if detection failed.
    """
    prompt = _load_prompt(text)
    raw_response = call_ollama_server(prompt=prompt, think=False)
    return _parse_binary_label(raw_response)


def store_output(df: pd.DataFrame, file_path: Path = RESULTS):
    """
    Store the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (Path, optional): The path to the output CSV file. Defaults to RESULTS.
    """
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Results saved to '{file_path}'")


def main():
    """
    Main function to load data, run hate speech detection, and store results.
    """
    df = load_data(DATA)
    # Pre-create result column as nullable integer to avoid float coercion
    df["class"] = pd.Series([pd.NA] * len(df), dtype="Int8")
    for row in df.iterrows():
        print(f"Processing row {row[0]}: {row[1]['text'][:50]}...")
        label = run_hate_speech_detection(row[1]["text"])
        if label in (0, 1):
            df.at[row[0], "ollama_response"] = int(label)
        else:
            # keep as <NA> instead of writing -1, preserves Int8 dtype
            df.at[row[0], "ollama_response"] = pd.NA
    store_output(df)


if __name__ == "__main__":
    main()
