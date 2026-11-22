import time

from google.genai import errors, types

from src import config

RETRY_CODES = {429, 500, 503, 504}


def make_config(**kwargs) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="low"), **kwargs
    )


def generate_with_fallback(client, primary_model: str, **kwargs):
    models = dict.fromkeys([primary_model, *config.FALLBACK_MODELS])
    last_error = None

    for model in models:
        start = time.time()
        try:
            response = client.models.generate_content(model=model, **kwargs)
        except Exception as e:
            code = getattr(e, "code", None)
            if isinstance(e, errors.APIError) and code not in RETRY_CODES:
                raise
            print(f"   [{model}: lỗi {code or type(e).__name__}, thử model khác]")
            last_error = e
            continue

        if config.DEBUG:
            print(f"   [{model}: {time.time() - start:.1f}s]")
        return response

    raise last_error


def stream_with_fallback(client, primary_model: str, **kwargs):
    models = dict.fromkeys([primary_model, *config.FALLBACK_MODELS])
    last_error = None

    for model in models:
        try:
            return client.models.generate_content_stream(model=model, **kwargs)
        except Exception as e:
            code = getattr(e, "code", None)
            if isinstance(e, errors.APIError) and code not in RETRY_CODES:
                raise
            print(f"   [{model}: lỗi {code or type(e).__name__}, thử model khác]")
            last_error = e
            continue

    raise last_error