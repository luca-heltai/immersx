#!/usr/bin/env python3
"""Append the ImmersX debugger configuration to a VS Code launch file."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path


CONFIGURATION_NAME = "ImmersX (CMake target)"


def matching_array_end(text: str, opening: int) -> int:
    """Return the index of the closing bracket for the array at opening."""
    depth = 0
    in_string = False
    escaped = False
    in_line_comment = False
    in_block_comment = False
    index = opening

    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""

        if in_line_comment:
            if character in "\r\n":
                in_line_comment = False
        elif in_block_comment:
            if character == "*" and following == "/":
                in_block_comment = False
                index += 1
        elif in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "/" and following == "/":
            in_line_comment = True
            index += 1
        elif character == "/" and following == "*":
            in_block_comment = True
            index += 1
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return index

        index += 1

    raise ValueError("the VS Code configurations array is not closed")


def append_configuration(launch_text: str, configuration: dict) -> str:
    """Append configuration while preserving existing JSONC text."""
    if re.search(
        rf'"name"\s*:\s*"{re.escape(CONFIGURATION_NAME)}"', launch_text
    ):
        return launch_text

    configurations_key = re.search(r'"configurations"\s*:', launch_text)
    if configurations_key is None:
        raise ValueError('launch.json has no "configurations" array')

    opening = launch_text.find("[", configurations_key.end())
    if opening == -1:
        raise ValueError('launch.json has no "configurations" array')
    closing = matching_array_end(launch_text, opening)

    existing = launch_text[opening + 1 : closing]
    has_trailing_comma = existing.rstrip().endswith(",")
    separator = "" if not existing.strip() or has_trailing_comma else ","
    rendered = json.dumps(configuration, indent=2)
    rendered = "\n" + "\n".join(f"  {line}" for line in rendered.splitlines())
    rendered += "\n"

    return launch_text[:closing] + separator + rendered + launch_text[closing:]


def write_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(contents)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} LAUNCH_JSON CONFIGURATION_JSON", file=sys.stderr)
        return 2

    launch_path = Path(sys.argv[1])
    configuration_path = Path(sys.argv[2])
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))

    if launch_path.exists():
        launch_text = launch_path.read_text(encoding="utf-8")
        updated_text = append_configuration(launch_text, configuration)
    else:
        launch_text = '{\n  "version": "0.2.0",\n  "configurations": []\n}\n'
        updated_text = append_configuration(launch_text, configuration)

    if updated_text != launch_text:
        write_atomically(launch_path, updated_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
