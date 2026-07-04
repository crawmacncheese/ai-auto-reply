from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import (
    ActionsSummarizerResult,
    ClassifierResult,
    FulfilledActionsResult,
    InferencerResult,
)
from utils.llm import invoke_structured_llm, resolve_output_schema


def infer_outcome(
    context: Context,
    classified_category: ClassifierResult,
    required_actions: ActionsSummarizerResult,
    fulfilled_actions: FulfilledActionsResult,
    dry_run: bool = False,
) -> InferencerResult:
    system_prompt = """\
You are a helpful assistant that determines whether a job referral is still possible after the job seeker has fulfilled the referrer's requests.
You will be given the conversation history, classified category, required actions from the referrer, and the job seeker's fulfilled responses.
Return only whether `referral_possible` is true or false based on the conversation and fulfilled actions.
Never make assumptions beyond what is supported by the conversation and fulfilled actions.
Never make up facts.

Set `referral_possible` to true when the referrer has agreed to help (conditionally or otherwise) and nothing in the conversation or fulfilled actions blocks a referral.

Set `referral_possible` to false when the conversation or fulfilled actions clearly indicate a referral cannot proceed. Important example:
- The referrer states the company does not support H1B sponsorship (or similar visa sponsorship limits) and asks whether the job seeker is on an F1 visa (or similar work-authorization status).
- The job seeker's fulfilled answer confirms they need sponsorship the company cannot provide (e.g., yes, they are on F1 and need H1B sponsorship).
In that case, referral is impossible even if the referrer was otherwise willing to help.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Required Actions:
{required_actions}

Fulfilled Actions:
{fulfilled_actions}
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

    output_schema = resolve_output_schema(InferencerResult)

    if dry_run:
        return system_prompt, user_prompt, output_schema

    return invoke_structured_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        output_schema,
        model_class=InferencerResult,
    )
