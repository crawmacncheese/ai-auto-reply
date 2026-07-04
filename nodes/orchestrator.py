from functools import lru_cache

from langgraph.graph import END, StateGraph
from rexpand_pyutils_file import read_file

from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.actions_summarizer import summarize_required_actions
from nodes.classifier import classify_conversation
from nodes.inferencer import infer_outcome
from nodes.message_generator import generate_message
from nodes.topic_suggester import suggest_topics


CATEGORIES = [Category(**category) for category in read_file("./input/categories.json")]
EXTENDED_CATEGORY_LOOKUP = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("./input/categories.json")
}


def _get_extended_category(state: State) -> ExtendedCategory:
    return EXTENDED_CATEGORY_LOOKUP[state.classified_category.category]


def classify_node(state: State) -> State:
    if state.classified_category is None:
        state.classified_category = classify_conversation(
            state.context, CATEGORIES, dry_run=False
        )
    return state


def summarize_actions_node(state: State) -> State:
    if state.required_actions is None:
        state.required_actions = summarize_required_actions(
            state.context, state.classified_category, dry_run=False
        )
    return state


def await_user_node(state: State) -> State:
    state.step = "awaiting_user_actions"
    return state


def infer_node(state: State) -> State:
    if state.inference is None:
        state.inference = infer_outcome(
            state.context,
            state.classified_category,
            state.required_actions,
            state.fulfilled_actions,
            dry_run=False,
        )
    return state


def suggest_topics_node(state: State) -> State:
    if state.suggested_topics is None:
        state.suggested_topics = suggest_topics(
            state.context,
            state.classified_category,
            required_actions=state.required_actions,
            fulfilled_actions=state.fulfilled_actions,
            inference=state.inference,
            dry_run=False,
        )
    return state


def generate_message_node(state: State) -> State:
    if state.generated_reply_message is not None:
        state.step = "end: reply generated"
        return state

    state.selected_topics = [topic.topic for topic in state.suggested_topics.topics]
    state.generated_reply_message = generate_message(
        state.context,
        state.classified_category,
        state.selected_topics,
        required_actions=state.required_actions,
        fulfilled_actions=state.fulfilled_actions,
        inference=state.inference,
        dry_run=False,
    )
    state.step = "end: reply generated"
    return state


def end_no_reply_node(state: State) -> State:
    state.step = "end: no reply needed"
    return state


def route_after_classify(state: State) -> str:
    extended_category = _get_extended_category(state)

    if not extended_category.reply_needed:
        return "end_no_reply"

    if not extended_category.human_action_required:
        return "suggest_topics"

    if state.required_actions is None:
        return "summarize_actions"

    if state.fulfilled_actions is None:
        return "await_user"

    if state.inference is None:
        return "infer"

    return "suggest_topics"


def route_after_summarize(state: State) -> str:
    if state.fulfilled_actions is None:
        return "await_user"

    if state.inference is None:
        return "infer"

    return "suggest_topics"


@lru_cache(maxsize=1)
def _build_workflow_graph():
    graph = StateGraph(State)

    graph.add_node("classify", classify_node)
    graph.add_node("summarize_actions", summarize_actions_node)
    graph.add_node("await_user", await_user_node)
    graph.add_node("infer", infer_node)
    graph.add_node("suggest_topics", suggest_topics_node)
    graph.add_node("generate_message", generate_message_node)
    graph.add_node("end_no_reply", end_no_reply_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "end_no_reply": "end_no_reply",
            "summarize_actions": "summarize_actions",
            "await_user": "await_user",
            "infer": "infer",
            "suggest_topics": "suggest_topics",
        },
    )
    graph.add_conditional_edges(
        "summarize_actions",
        route_after_summarize,
        {
            "await_user": "await_user",
            "infer": "infer",
            "suggest_topics": "suggest_topics",
        },
    )
    graph.add_edge("await_user", END)
    graph.add_edge("end_no_reply", END)
    graph.add_edge("infer", "suggest_topics")
    graph.add_edge("suggest_topics", "generate_message")
    graph.add_edge("generate_message", END)

    return graph.compile()


def _to_state(result) -> State:
    if isinstance(result, State):
        return result
    return State(**result)


def orchestrate(state: State) -> State:
    return _to_state(_build_workflow_graph().invoke(state))
