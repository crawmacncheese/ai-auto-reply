from models.base import BaseModel
from models.context import Context
from models.llm_result import (
    ActionsSummarizerResult,
    ClassifierResult,
    FulfilledActionsResult,
    InferencerResult,
    MessageGeneratorResult,
    TopicSuggesterResult,
)


class State(BaseModel):
    step: str | None = None
    context: Context
    classified_category: ClassifierResult | None = None
    required_actions: ActionsSummarizerResult | None = None
    fulfilled_actions: FulfilledActionsResult | None = None
    inference: InferencerResult | None = None
    suggested_topics: TopicSuggesterResult | None = None
    selected_topics: list[str] | None = None
    generated_reply_message: MessageGeneratorResult | None = None
