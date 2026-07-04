import hashlib
import logging
import os
from typing import Any, Optional, TypeVar, Union

from dotenv import load_dotenv
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from rexpand_pyutils_file import read_file, write_file

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable is not set")

DEEPSEEK_API_BASE = os.getenv(
    "DEEPSEEK_API_BASE", "https://api.deepseek.com/beta"
).strip()

LLM_USE_CACHE: bool = os.getenv("LLM_USE_CACHE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

LLM_INCLUDE_DEBUG_FIELDS: bool = os.getenv(
    "LLM_INCLUDE_DEBUG_FIELDS", "true"
).strip().lower() in ("1", "true", "yes", "y", "on")

LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat").strip()
try:
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0").strip())
except ValueError:
    LLM_TEMPERATURE = 0.0

logging.info(f"LLM_MODEL: {LLM_MODEL}")
logging.info(f"LLM_TEMPERATURE: {LLM_TEMPERATURE}")
logging.info(f"LLM_USE_CACHE: {LLM_USE_CACHE}")
logging.info(f"LLM_INCLUDE_DEBUG_FIELDS: {LLM_INCLUDE_DEBUG_FIELDS}")
logging.info(f"DEEPSEEK_API_BASE: {DEEPSEEK_API_BASE}")

T = TypeVar("T", bound=BaseModel)
OutputSchema = Union[type[T], dict[str, Any]]

default_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
)


def prune_debug_fields_from_schema(schema: dict) -> dict:
    """Return a copy of the JSON schema without debug fields (reason, confidence)."""
    DEBUG_FIELDS = {"reason", "confidence"}

    def _prune(node: Any) -> Any:
        if isinstance(node, dict):
            new_node: dict[str, Any] = {}
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    filtered_props = {
                        prop_key: _prune(prop_val)
                        for prop_key, prop_val in value.items()
                        if prop_key not in DEBUG_FIELDS
                    }
                    new_node[key] = filtered_props
                elif key == "required" and isinstance(value, list):
                    new_node[key] = [r for r in value if r not in DEBUG_FIELDS]
                else:
                    new_node[key] = _prune(value)
            return new_node
        if isinstance(node, list):
            return [_prune(item) for item in node]
        return node

    return _prune(schema)


def resolve_output_schema(model_class: type[T]) -> OutputSchema:
    if LLM_INCLUDE_DEBUG_FIELDS:
        return model_class
    return prune_debug_fields_from_schema(model_class.model_json_schema())


def _coerce_structured_result(
    result: Any, schema: OutputSchema, model_class: type[T]
) -> T:
    if isinstance(result, model_class):
        return result
    if isinstance(result, dict):
        return model_class.model_validate(result)
    raise TypeError(f"Unexpected structured output type: {type(result)}")


def invoke_structured_llm(
    input: LanguageModelInput,
    schema: OutputSchema,
    *,
    model_class: type[T],
    use_cache: bool = LLM_USE_CACHE,
    verbose: bool = False,
    llm: Optional[ChatOpenAI] = None,
) -> T:
    """Call DeepSeek with tool-based structured output (strict schema on beta API)."""
    llm = llm or default_llm
    cache_material = f"{input}|{schema}|{model_class.__name__}"

    if use_cache:
        input_hash = hashlib.md5(cache_material.encode()).hexdigest()
        filepath = f"./.cache/{input_hash}.json"
        cached_response = read_file(filepath)
        if cached_response is not None:
            if verbose:
                logging.info(f"Cache hit: {filepath}")
            return model_class.model_validate(cached_response)

    structured_llm = llm.with_structured_output(
        schema,
        method="function_calling",
        strict=True,
    )
    result = structured_llm.invoke(input)

    if use_cache:
        payload = (
            result.model_dump()
            if isinstance(result, BaseModel)
            else result
        )
        write_file(filepath, payload)
        if verbose:
            logging.info(f"Cache miss: {filepath}")

    return _coerce_structured_result(result, schema, model_class)


def invoke_llm(
    input: LanguageModelInput,
    config: Optional[RunnableConfig] = None,
    *,
    use_cache: bool = LLM_USE_CACHE,
    verbose: bool = False,
    llm: Optional[ChatOpenAI] = None,
    **kwargs: Any,
) -> BaseMessage:
    """Low-level LLM invoke for unstructured prompts."""
    llm = llm or default_llm

    if use_cache:
        input_hash = hashlib.md5((str(input) + "|" + str(config)).encode()).hexdigest()
        filepath = f"./.cache/{input_hash}.json"

        cached_response = read_file(filepath)
        if cached_response is not None:
            if verbose:
                logging.info(f"Cache hit: {filepath}")
            return AIMessage(**cached_response)

        if verbose:
            logging.info(f"Cache miss: {filepath}")

        response: BaseMessage = llm.invoke(input, config, **kwargs)
        write_file(filepath, response.model_dump())
        return response

    return llm.invoke(input, config, **kwargs)
