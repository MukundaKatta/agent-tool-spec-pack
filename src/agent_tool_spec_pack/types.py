"""Types shared across the package: Format enum and InspectionResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Format(str, Enum):
    """Provider tool-schema shape to emit.

    `BEDROCK_ANTHROPIC` is identical to `ANTHROPIC` (Bedrock just forwards the
    Anthropic shape). `BEDROCK_LLAMA` emits the Llama-on-Bedrock function
    shape (similar to OpenAI but flat at the top level).
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    BEDROCK_LLAMA = "bedrock_llama"


@dataclass
class ParamInfo:
    """One parameter as the packer sees it before provider shaping."""

    name: str
    annotation: Any
    has_default: bool
    default: Any
    description: str
    schema: dict[str, Any]


@dataclass
class InspectionResult:
    """Intermediate packer output. Useful for debugging or building custom shapes."""

    name: str
    description: str
    params: list[ParamInfo] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    raw_doc: str = ""
