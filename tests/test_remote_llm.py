"""Send authenticated requests to a future permitted vLLM HTTPS endpoint.

This client is intentionally not pointed at Kaggle. Kaggle does not permit
tunneling or public API hosting, so this script has not been tested against
the Kaggle-hosted server. It validates HTTPS and uses the API key from `.env`.

Required `.env` values:
    VLLM_API_BASE_URL=https://your-secure-endpoint.example/v1
    VLLM_API_KEY=your_vllm_api_key
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

REQUEST_TIMEOUT_SECONDS = 120


def get_settings():
    """Read and validate the remote endpoint settings without printing secrets."""
    base_url = os.getenv("VLLM_API_BASE_URL", "").strip().rstrip("/")
    api_key = os.getenv("VLLM_API_KEY", "").strip()

    if not base_url:
        raise ValueError("VLLM_API_BASE_URL is missing from .env.")
    if not api_key:
        raise ValueError("VLLM_API_KEY is missing from .env.")

    parsed_url = urlparse(base_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("VLLM_API_BASE_URL must be a valid HTTPS URL.")

    # Accept a base URL with or without /v1, but never create /v1/v1.
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    return base_url, api_key


def main():
    try:
        base_url, api_key = get_settings()
    except ValueError as error:
        print(f"Configuration error: {error}")
        return 1

    headers = {"Authorization": f"Bearer {api_key}"}

    # /v1/models confirms that the endpoint is reachable and authenticated.
    try:
        models_response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=30,
        )
        models_response.raise_for_status()
    except requests.exceptions.Timeout:
        print("Timed out while contacting the model endpoint.")
        return 1
    except requests.exceptions.ConnectionError:
        print("Could not connect to the model endpoint. Check the URL and network.")
        return 1
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        if status == 401:
            print("Authentication failed. Check VLLM_API_KEY in .env.")
        else:
            print(f"The endpoint returned HTTP {status}.")
        return 1
    except requests.exceptions.RequestException as error:
        print(f"Request failed: {type(error).__name__}")
        return 1

    try:
        models = models_response.json().get("data", [])
    except (AttributeError, ValueError):
        print("The endpoint returned an unexpected JSON response.")
        return 1

    if not models:
        print("The endpoint returned no models.")
        return 1

    try:
        model_id = models[0]["id"]
    except (KeyError, IndexError, TypeError):
        print("The endpoint returned a model entry without a usable ID.")
        return 1

    print(f"Authenticated successfully. Using model: {model_id}")

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Explain what a remote LLM API does in one short sentence.",
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }

    # /v1/chat/completions runs the model and returns an OpenAI-style response.
    try:
        chat_response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        chat_response.raise_for_status()
    except requests.exceptions.Timeout:
        print("The model took too long to respond.")
        return 1
    except requests.exceptions.ConnectionError:
        print("The connection closed before the model responded.")
        return 1
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        if status == 401:
            print("Authentication failed. Check VLLM_API_KEY in .env.")
        else:
            print(f"The model request returned HTTP {status}.")
        return 1
    except requests.exceptions.RequestException as error:
        print(f"Model request failed: {type(error).__name__}")
        return 1

    try:
        answer = chat_response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        print("The endpoint returned an unexpected JSON response.")
        return 1

    if not isinstance(answer, str):
        print("The endpoint returned a non-text response.")
        return 1

    print("\nGenerated response:")
    print(answer.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
