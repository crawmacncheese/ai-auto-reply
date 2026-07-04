from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import (
    ActionsSummarizerResult,
    ClassifierResult,
    FulfilledActionsResult,
    InferencerResult,
    MessageGeneratorResult,
)
from utils.llm import (
    LLM_INCLUDE_DEBUG_FIELDS,
    invoke_structured_llm,
    resolve_output_schema,
)


def generate_message(
    context: Context,
    classified_category: ClassifierResult,
    selected_topics: list[str],
    *,
    required_actions: ActionsSummarizerResult | None = None,
    fulfilled_actions: FulfilledActionsResult | None = None,
    inference: InferencerResult | None = None,
    dry_run: bool = False,
) -> MessageGeneratorResult:
    system_prompt = """\
You are a helpful assistant that generates a message for the job seeker to reply for a potential referral or intro call.
You will be given the conversation history, classified category, selected topics for new message, actions required by the referrer and fulfilled by the job seeker (optional), inferred result regarding the referral possibility (optional), job seeker profile (optional), and referrer profile (optional).
When generating the message, always evaluate the latest existing messages first.
Never make up facts.
Never mention anything you don't know based on the user prompt.
Never use placeholders in the message since this will be the final message content sent to the referrer.
If needed to ask some questions to the referrer, never ask more than 2 questions at the same time since the referrer might get overwhelmed.
"""

    if inference is not None and not inference.referral_possible:
        system_prompt += """
REFERRAL IS NOT POSSIBLE (referral_possible=false).
Write a gracious, professional closing message.
Do NOT ask for a referral, internal referral submission, or a call to discuss a referral.
Do NOT express continued expectation of a referral.
You may thank the referrer, acknowledge the visa or eligibility constraint if present in the conversation, and optionally ask for alternative referrers or general advice.
"""
    else:
        system_prompt += """
The referral is still possible. Write a message aligned with the selected topics and move the conversation forward professionally.
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
You will need to generate a message along with confidence score and reason.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Selected Topics:
{selected_topics}
"""

    if required_actions is not None:
        user_prompt += f"""
Required Actions:
{required_actions}
"""

    if fulfilled_actions is not None:
        user_prompt += f"""
Fulfilled Actions:
{fulfilled_actions}
"""

    if inference is not None:
        user_prompt += f"""
Referral Possible:
{inference.referral_possible}
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

    output_schema = resolve_output_schema(MessageGeneratorResult)

    if dry_run:
        return system_prompt, user_prompt, output_schema

    return invoke_structured_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        output_schema,
        model_class=MessageGeneratorResult,
    )
