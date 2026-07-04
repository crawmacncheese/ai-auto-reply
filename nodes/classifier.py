from langchain_core.messages import HumanMessage, SystemMessage

from models.category import Category
from models.context import Context
from models.llm_result import ClassifierResult
from utils.llm import (
    LLM_INCLUDE_DEBUG_FIELDS,
    invoke_structured_llm,
    resolve_output_schema,
)


def classify_conversation(
    context: Context, categories: list[Category], dry_run: bool = False
) -> ClassifierResult:
    system_prompt = f"""\
You are a helpful assistant that classifies the whole conversation between job seeker and referrer into one of the following categories.
When classifying, always evaluate the **last** messages.
If multiple categories are applicable, you should choose the one indicating the last status of the conversation.
Never make up facts.

Category Definition:
{categories}
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
You will need to provide confidence score, reason, and referenced message ids (only include the most relevant message ids to the classification).
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

The last message is from {context.messages[-1].role}.
"""

    if context.messages[-1].role == "job seeker":
        user_prompt += "`no_reply` or `followed_up_no_reply` is highly likely to be the category.\n"

    if context.user_profile:
        user_prompt += f"""
Job Seeker Profile:
{context.user_profile}
"""

    if context.referrer_profile:
        user_prompt += f"""
Referrer Profile:
{context.referrer_profile}
"""

    output_schema = resolve_output_schema(ClassifierResult)

    if dry_run:
        return system_prompt, user_prompt, output_schema

    return invoke_structured_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        output_schema,
        model_class=ClassifierResult,
    )
