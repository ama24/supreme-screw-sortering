"""Form fields generated from configs/config.yaml, so the GUI shows every setting of every tool.

The kind of each field comes from its value (number, switch, range, text, choice, YAML) and its
help text from the `#` comments in config.yaml. A comment of the form `a | b | c` turns a text
setting into a choice. New keys in config.yaml therefore appear in the GUI without any GUI code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHOICES_RE = re.compile(r"^\s*([\w.-]+(?:\s*\|\s*[\w.-]+)+)")


@dataclass
class Field:
    path: str  # dotted config path, e.g. scenegen.render.samples
    key: str
    kind: str  # bool | int | float | str | choice | pair | yaml | section
    value: Any  # the value in config.yaml (the default)
    help: str = ""
    choices: list[str] = field(default_factory=list)
    children: list[Field] = field(default_factory=list)


def config_comments(path: str | Path) -> dict[str, str]:
    """dotted path -> comment text (the comment lines above a key, then its end-of-line comment)."""
    from ruamel.yaml import YAML

    with Path(path).open(encoding="utf-8") as f:
        doc = YAML().load(f)
    out: dict[str, str] = {}
    _collect(doc, "", out, [])
    return out


def _last_paragraph(text: str) -> list[str]:
    """Comment lines of the last block (blank lines separate blocks), without the '#'."""
    blocks: list[list[str]] = [[]]
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("#"):
            blocks[-1].append(s.lstrip("#").strip())
        elif not s and blocks[-1]:
            blocks.append([])
    blocks = [b for b in blocks if b]
    return blocks[-1] if blocks else []


def _collect(node: Any, prefix: str, out: dict[str, str], inherited: list[str]) -> list[str]:
    """Comments of one mapping; returns the comment lines after its last key (they describe the
    parent's next key). ruamel stores a key's end-of-line comment together with the full-line
    comments below it, and the comments above the first key in the mapping's own `ca.comment`."""
    keys = list(node)
    if not keys or not hasattr(node, "ca"):
        return []
    head = node.ca.comment[1] if node.ca.comment and len(node.ca.comment) > 1 else None
    if head:
        inherited = _last_paragraph("".join(str(tok.value) for tok in head))
    above: dict[str, list[str]] = {str(keys[0]): list(inherited)} if inherited else {}
    trailing: list[str] = []
    for i, key in enumerate(keys):
        items = node.ca.items.get(key)
        token = items[2] if items and len(items) > 2 else None
        if token is not None:
            lines = str(token.value).split("\n")
            key_line = node.lc.key(key)[0] if hasattr(node, "lc") else -1
            on_key_line = token.start_mark.line == key_line
            eol = lines[0].strip().lstrip("#").strip() if on_key_line else ""
            if eol:
                above.setdefault(str(key), []).append(eol)
            below = _last_paragraph("\n".join(lines[1:] if on_key_line else lines))
        else:
            below = []
        value = node[key]
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            # comments right after "key:" open the section: they describe its first child
            first = below if token is not None and not isinstance(node[key], str) else []
            below = _collect(value, path + ".", out, first)
        if below:
            if i + 1 < len(keys):
                above[str(keys[i + 1])] = below + above.get(str(keys[i + 1]), [])
            else:
                trailing = below
    for key in keys:
        if above.get(str(key)):
            out[f"{prefix}{key}"] = "\n".join(above[str(key)])
    return trailing


def choices_from(comment: str) -> list[str]:
    m = CHOICES_RE.match(comment or "")
    return [c.strip() for c in m.group(1).split("|")] if m else []


def build_fields(
    data: Mapping[str, Any],
    comments: Mapping[str, str],
    prefix: str = "",
    parent_help: str = "",
    skip: frozenset[str] = frozenset(),
) -> list[Field]:
    """One Field per key of `data` (recursing into mappings)."""
    out = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        if path in skip:
            continue
        help_ = comments.get(path, "")
        f = Field(path, str(key), kind_of(value), value, help_)
        if f.kind == "section":
            f.children = build_fields(value, comments, path + ".", help_, skip)
        elif f.kind == "str":
            opts = choices_from(help_) or choices_from(parent_help)
            if value in opts:
                f.kind, f.choices = "choice", opts
        out.append(f)
    return out


def kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, Mapping):
        return "section" if value else "yaml"
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        return "pair"
    return "yaml"


def coerce(f: Field, raw: Any) -> Any:
    """GUI input -> config value of the field's kind. Raises ValueError on bad input."""
    if f.kind == "bool":
        return bool(raw)
    if f.kind == "int":
        if raw is None or raw == "":
            raise ValueError("a whole number is required")
        if float(raw) != int(float(raw)):
            raise ValueError("a whole number is required")
        return int(float(raw))
    if f.kind == "float":
        if raw is None or raw == "":
            raise ValueError("a number is required")
        return float(raw)
    if f.kind == "pair":
        a, b = raw
        if a is None or b is None:
            raise ValueError("both numbers are required")
        ints = all(isinstance(v, int) for v in f.value)
        return [int(a), int(b)] if ints else [float(a), float(b)]
    if f.kind == "yaml":
        import yaml

        try:
            return yaml.safe_load(raw) if isinstance(raw, str) else raw
        except yaml.YAMLError as exc:
            raise ValueError(f"not valid YAML: {exc}") from None
    return "" if raw is None else str(raw)


def to_yaml_text(value: Any) -> str:
    import yaml

    if value in ({}, [], None):
        return {dict: "{}", list: "[]"}.get(type(value), "null")
    return yaml.safe_dump(value, sort_keys=False, default_flow_style=None).strip()
