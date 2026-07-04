from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import (
    ActionsSummarizerResult,
    ClassifierResult,
    FulfilledActionsResult,
    InferencerResult,
    TopicSuggesterResult,
)
from utils.llm import (
    LLM_INCLUDE_DEBUG_FIELDS,
    invoke_structured_llm,
    resolve_output_schema,
)


def suggest_topics(
    context: Context,
    classified_category: ClassifierResult,
    *,
    required_actions: ActionsSummarizerResult | None = None,
    fulfilled_actions: FulfilledActionsResult | None = None,
    inference: InferencerResult | None = None,
    dry_run: bool = False,
) -> TopicSuggesterResult:
    system_prompt = """\
You are a helpful assistant that suggests at most 4 conversation topic bullet points for the job seeker to reply for a potential referral or intro call.
Each bullet point should be an extremely short simple phrase (not sentence).
You will be given the conversation history, classified category, actions required by the referrer and fulfilled by the job seeker (optional), inferred result regarding the referral possibility (optional), job seeker profile (optional), and referrer profile (optional).
When suggesting topics, always evaluate the latest messages first and from the existing facts.
Never suggest todo items that are not in the conversation history or fulfilled actions.
Never make assumptions.
Never make up facts.
"""

    if inference is not None and not inference.referral_possible:
        system_prompt += """
REFERRAL IS NOT POSSIBLE (referral_possible=false).
Do NOT suggest topics about securing a referral, scheduling a call for referral purposes, or following up to obtain a referral.
Focus on gracious closure topics such as Thank you, Ask for alternative referrers, or Express appreciation for their time.
Do NOT suggest `Ask for a brief call`.

Topic Examples when referral is not possible:
1. Thank you
2. Ask for alternative referrers
3. Express appreciation
"""
    else:
        system_prompt += """
As long as the referral is possible, suggest `Ask for a brief call` in the topics to help they keep the conversation going and know each other better.

Topic Examples (but not limited to):
1. Thank you
2. Express interest
3. Express background match
4. Ask for a brief call
5. Ask for alternative referrers
6. Follow-up
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
You will need to provide conversation topics along with confidence score and reason.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}
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

    output_schema = resolve_output_schema(TopicSuggesterResult)

    if dry_run:
        return system_prompt, user_prompt, output_schema

    return invoke_structured_llm(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        output_schema,
        model_class=TopicSuggesterResult,
    )
