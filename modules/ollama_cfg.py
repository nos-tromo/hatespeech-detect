import logging
import os

import ollama
import requests
import torch

from utils.mappings_loader import load_mappings

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _get_ollama_health(url: str = OLLAMA_HOST) -> bool:
    """
    Perform a health check by querying Ollama's /api/tags endpoint.

    Args:
        url (str, optional): The base URL of the Ollama server. Defaults to environment variable or localhost.

    Returns:
        bool: True if the Ollama server responds with model tags, False otherwise.
    """
    response = requests.get(f"{url}/api/tags", timeout=5)
    return response.status_code == 200 and "models" in response.json()


def load_ollama_model(
    filename: str = "ollama_models.json", fallback: str = "gemma3n:e4b"
) -> str | None:
    """
    Load the specified ollama model.

    Args:
        filename (str): The name of the JSON file containing model mappings. Defaults to "ollama_models.json".
        fallback (str): The fallback model to use if no suitable model is found. Defaults to "gemma3n:e4b".

    Returns:
        str | None: The name of the model if loaded successfully, None otherwise.

    Raises:
        RuntimeError: If the model file is not found or empty.
    """
    models = load_mappings(filename)
    if not models:
        raise RuntimeError(
            f"Model file '{filename}' not found or empty. Please check the file."
        )
    model = (
        models.get("cuda")
        if torch.cuda.is_available()
        else models.get("mps")
        if torch.backends.mps.is_available()
        else models.get("cpu", fallback)
    )
    logger.info("Loaded Ollama model: %s", model)
    return model


def call_ollama_server(
    model: str | None,
    prompt: str,
    think: bool = False,
    num_ctx: int = 16384,
    seed: int = 42,
    temperature: float = 0.1,
    top_k: int = 1,
    top_p: float = 0.0,
    num_predict: int = 1,
    stop: list[str] = ["\n"],
) -> str:
    """
    Call the ollama server with the given model and prompt.

    Args:
        model (str | None): The name of the model to use.
        prompt (str): The prompt to send to the model.
        think (bool): Whether to enable "think" mode for the model. Defaults to False.
        num_ctx (int): The number of context tokens to use. Defaults to 16384.
        seed (int): The random seed for the model's response. Defaults to 42.
        temperature (float): The temperature for the model's response. Defaults to 0.1.
        top_k (int): The top_k parameter for the model's response. Defaults to 1.
        top_p (float): The top_p parameter for the model's response. Defaults to 0.0.
        num_predict (int): The number of tokens to predict. Defaults to 1.
        stop (list[str]): A list of stop sequences for the model's response. Defaults to ["\n"].

    Returns:
        str: The response from the ollama server, or an empty string if an error occurs.

    Raises:
        RuntimeError: If the Ollama model cannot be loaded or the server is unreachable.
    """
    if not model:
        model = load_ollama_model()

    if not model:
        raise RuntimeError(
            "Ollama model could not be loaded. Please check your configuration."
        )
    if not _get_ollama_health():
        logger.error(
            "Ollama server does not respond. Please ensure it is running and accessible."
        )
        raise RuntimeError(
            "Ollama server is not reachable. Please check your configuration."
        )

    # Ensure environment variable is set for ollama library
    os.environ["OLLAMA_HOST"] = OLLAMA_HOST
    response = ollama.chat(
        model=model,
        think=think,
        messages=[{"role": "user", "content": prompt}],
        options={
            # Only include num_ctx when positive; -1 or 0 lets the server/model default
            **({"num_ctx": num_ctx} if isinstance(num_ctx, int) and num_ctx > 0 else {}),
            # Deterministic, single-token classification
            "seed": seed,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": num_predict,
            # Stop at newline just in case a model tries to ramble
            "stop": stop,
        },
    )
    return response["message"]["content"].strip()


# System prompt
system_prompt = """
You are a highly proficient assistant that strictly follows instructions and provides only the requested output.
Do not include interpretations, comments, or acknowledgments unless explicitly asked.
Do not use confirmation phrases such as "Sure, here it comes:", "Got it.", "Here is the translation:", or similar expressions.
Responses shall be generated without any markdown formatting unless specified otherwise.
All your outputs must be in {language} language regardless of the input language.
"""

# Violence and hate speech detection
hate_detection = """
You are an expert in identifying and classifying content related to violence and hate speech.
Analyze the following text and determine if it contains any violent or hate speech content. 
For each identified instance, classify it into the following categories: 
"1" - if it contains violent or hate speech content; 
"0" -  if no such content is present.

Text to analyze:
"{text}"
"""

# Text summarization
text_summarization = """
You are an expert summarizer. Create a concise and coherent summary of the following text, capturing all key points and essential information.

Instructions:
1. Content Coverage: Ensure that the summary includes all main ideas and important details from the original text.
2. Brevity: The summary should be no longer than 15 sentences.
3. Clarity: Use clear and straightforward language. All your outputs must be in {language} language.
4. No Additional Information: Do not include personal opinions, interpretations, or external information.
5. No Extraneous Information: Do not include any Markdown code blocks, additional formatting, or extraneous information.

Text to Summarize:
"{text}"
"""

# Topic summarization
topic_titles = """
You are an expert for topic modeling that is highly proficient in generating topic titles from raw text.
I have a topic that is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, generate a short label of the topic of at most 5 words.
"""

topic_summaries = """
You are an expert for topic modeling that is highly proficient in summarizing topics from raw text.
I have a topic that is described by the following title: "{title}"
The topic is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, create a short summary of the topic.
"""

# Combine prompts with system prompt
hate_detection_prompt = system_prompt + "\n\n" + hate_detection
text_summarization_prompt = system_prompt + "\n\n" + text_summarization
topic_titles_prompt = system_prompt + "\n\n" + topic_titles
topic_summaries_prompt = system_prompt + "\n\n" + topic_summaries
