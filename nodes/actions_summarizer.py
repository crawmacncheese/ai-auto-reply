from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ActionsSummarizerResult, ClassifierResult
from utils.llm import (
    LLM_INCLUDE_DEBUG_FIELDS,
    invoke_structured_llm,
    resolve_output_schema,
)


def summarize_required_actions(
    context: Context,
    classified_category: ClassifierResult,
    dry_run: bool = False,
) -> ActionsSummarizerResult:
    system_prompt = """\
You are a helpful assistant that summarizes what the job seeker must do or provide before they can reply to the referrer.
You will be given the conversation history and the classified category of the conversation.
Focus on the referrer's **last** message(s) and extract only explicit requests directed at the job seeker.
Each action should be an extremely short simple phrase (not a full sentence).
Use `details` for any helpful extra context quoted or paraphrased from the referrer's message.
Never invent actions that are not clearly requested in the conversation.
Never make assumptions.
Never make up facts.

Action Examples (but not limited to):
1. Send job link
2. Send resume
3. Provide email
4. Provide phone number
5. Answer question about target role
6. Answer question about team preference
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
You will need to provide each action along with confidence score and reason.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

The last message is from {context.messages[-1].role}.
"""

    if classified_category.category == "agree_on_condition":
        user_prompt += """
The referrer agreed to help conditionally. Summarize each condition or item the job seeker must provide before the referrer will proceed.
"""
    elif classified_category.category == "request_additional_info":
        user_prompt += """
The referrer asked for more information before committing. Summarize each question or information request the job seeker must answer.
"""

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

    output_schema = resolve_output_schema(ActionsSummarizerResult)

    if dry_run:
        return system_prompt, user_prompt, output_schema

    return invoke_structured_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        output_schema,
        model_class=ActionsSummarizerResult,
    )
