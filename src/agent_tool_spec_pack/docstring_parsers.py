"""Docstring parsing for Google, reST, and numpy styles.

Each parser returns a tuple `(description, params)` where `description` is the
short first-paragraph summary and `params` is a dict `{name: description}`.

We try each style in order. The first style that finds any structured params
wins. If none match, we fall back to first-paragraph-only.
"""

from __future__ import annotations

import inspect
import re

ParamDescriptions = dict[str, str]


def parse_docstring(doc: str | None) -> tuple[str, ParamDescriptions]:
    """Parse a docstring into (description, {param_name: param_doc}).

    Returns ("", {}) when `doc` is None or empty.
    """
    if not doc:
        return "", {}

    cleaned = inspect.cleandoc(doc)

    for parser in (_parse_google, _parse_rest, _parse_numpy):
        desc, params = parser(cleaned)
        if params:
            return desc, params

    # Nothing structured. Fall back to the first paragraph.
    return _first_paragraph(cleaned), {}


def _first_paragraph(doc: str) -> str:
    """Return the first blank-line-delimited paragraph, joined onto one line."""
    if not doc:
        return ""
    paragraph_chunks: list[str] = []
    for line in doc.splitlines():
        if not line.strip():
            if paragraph_chunks:
                break
            continue
        paragraph_chunks.append(line.strip())
    return " ".join(paragraph_chunks).strip()


# Google-style:
#
#     Short summary line.
#
#     Args:
#         name: Description of name. Can wrap onto
#             the next line if it stays indented.
#         other (int): Description.
#
#     Returns:
#         ...
_GOOGLE_SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Params)\s*:\s*$",
    re.MULTILINE,
)
_GOOGLE_END_SECTION_RE = re.compile(
    r"^(Returns|Yields|Raises|Examples?|Notes?|See Also|Attributes|Todo)\s*:\s*$",
    re.MULTILINE,
)
_GOOGLE_PARAM_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.*)$",
)


def _parse_google(doc: str) -> tuple[str, ParamDescriptions]:
    head = _first_paragraph(doc)
    m = _GOOGLE_SECTION_RE.search(doc)
    if not m:
        return head, {}

    args_body_start = m.end()
    end_m = _GOOGLE_END_SECTION_RE.search(doc, args_body_start)
    args_body = doc[args_body_start : end_m.start() if end_m else len(doc)]

    params: ParamDescriptions = {}
    current_name: str | None = None
    current_indent: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_name is not None:
            params[current_name] = " ".join(current_lines).strip()

    for raw_line in args_body.splitlines():
        if not raw_line.strip():
            continue
        line_indent = len(raw_line) - len(raw_line.lstrip(" \t"))
        param_match = _GOOGLE_PARAM_RE.match(raw_line)
        if param_match and (current_indent is None or line_indent <= current_indent):
            # Start a new param.
            flush()
            current_name = param_match.group("name")
            current_indent = line_indent
            current_lines = [param_match.group("desc").strip()]
        elif current_name is not None:
            # Continuation line for the current param.
            current_lines.append(raw_line.strip())
    flush()

    return head, params


# reST-style:
#
#     :param name: Description of name. Can wrap, and the
#         continuation must be indented.
#     :type name: int
#     :param other: ...
_REST_PARAM_RE = re.compile(
    r"^\s*:param\s+(?:(?P<typename>[A-Za-z_][\w.\[\], ]*)\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<desc>.*)$",  # noqa: E501
)
_REST_FIELD_RE = re.compile(r"^\s*:[A-Za-z]+(?:\s+[\w.\[\], ]+)?:\s*")


def _parse_rest(doc: str) -> tuple[str, ParamDescriptions]:
    head = _first_paragraph(doc)
    params: ParamDescriptions = {}
    current_name: str | None = None
    current_indent: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_name is not None:
            params[current_name] = " ".join(current_lines).strip()

    for raw_line in doc.splitlines():
        if not raw_line.strip():
            continue
        param_match = _REST_PARAM_RE.match(raw_line)
        if param_match:
            flush()
            current_name = param_match.group("name")
            current_indent = len(raw_line) - len(raw_line.lstrip(" \t"))
            current_lines = [param_match.group("desc").strip()]
        elif _REST_FIELD_RE.match(raw_line):
            # A different reST field like :type:, :returns:, :raises:. End the current param.
            flush()
            current_name = None
            current_indent = None
            current_lines = []
        elif current_name is not None:
            line_indent = len(raw_line) - len(raw_line.lstrip(" \t"))
            if current_indent is None or line_indent > current_indent:
                current_lines.append(raw_line.strip())
            else:
                flush()
                current_name = None
                current_indent = None
                current_lines = []
    flush()
    return head, params


# numpy-style:
#
#     Short summary.
#
#     Parameters
#     ----------
#     name : int
#         Description.
#     other : str, optional
#         Another description.
_NUMPY_HEADING_RE = re.compile(
    r"^(Parameters|Params|Arguments|Args)\s*\n[-=]{2,}\s*$",
    re.MULTILINE,
)
_NUMPY_END_HEADING_RE = re.compile(
    r"^(Returns|Yields|Raises|Examples?|Notes?|See Also|Attributes)\s*\n[-=]{2,}\s*$",
    re.MULTILINE,
)
_NUMPY_PARAM_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^\n]*)?$",
)


def _parse_numpy(doc: str) -> tuple[str, ParamDescriptions]:
    head = _first_paragraph(doc)
    m = _NUMPY_HEADING_RE.search(doc)
    if not m:
        return head, {}

    body_start = m.end()
    end_m = _NUMPY_END_HEADING_RE.search(doc, body_start)
    body = doc[body_start : end_m.start() if end_m else len(doc)]

    params: ParamDescriptions = {}
    current_name: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_name is not None:
            params[current_name] = " ".join(current_lines).strip()

    for raw_line in body.splitlines():
        if not raw_line.strip():
            continue
        # A header line is non-indented and matches NAME or NAME : TYPE.
        is_unindented = raw_line == raw_line.lstrip()
        param_match = _NUMPY_PARAM_RE.match(raw_line) if is_unindented else None
        if param_match:
            flush()
            current_name = param_match.group("name")
            current_lines = []
        elif current_name is not None:
            current_lines.append(raw_line.strip())
    flush()
    return head, params
