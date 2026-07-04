from typing import Optional

from pydantic import ConfigDict

from models.base import BaseModel


class ClassifierResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["category", "referenced_message_ids", "confidence", "reason"]
        },
    )
    category: str
    confidence: Optional[float] = None
    reason: Optional[str] = None
    referenced_message_ids: list[str]


class Topic(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["topic", "confidence", "reason"]},
    )
    topic: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


class TopicSuggesterResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["topics"]},
    )
    topics: list[Topic]


class MessageGeneratorResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["message", "confidence", "reason"]
        },
    )
    message: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


class RequiredAction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["action", "details", "confidence", "reason"]
        },
    )
    action: str
    details: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None


class ActionsSummarizerResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["actions"]},
    )
    actions: list[RequiredAction]


class FulfilledAction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["action", "response"]},
    )
    action: str
    response: str


class FulfilledActionsResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["actions"]},
    )
    actions: list[FulfilledAction]


class InferencerResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["referral_possible"]},
    )
    referral_possible: bool
