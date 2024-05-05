from typing import Iterable
from os import get_terminal_size
from functools import cache


def readLines(path: str, max_lines: int = 0) -> list[str]:
    """Read the file from the given `path` and return a list of **non-empty** lines."""
    result = []
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
                result.append(line)
                if max_lines > 0 and count >= max_lines:
                    break
    return result


def makeTable(header: Iterable[str], data: Iterable[tuple]) -> str:
    """Make a nice Markdown table from the given header and data."""
    table_head = f"| {' | '.join(header)} |"
    table_sep = "| --- " * len(header) + "|"
    table_body = ""
    for row in data:
        assert len(row) == len(header), "Row length does not match header length."
        table_body += f"| {' | '.join(row)} |\n"
    table = f"{table_head}\n{table_sep}\n{table_body}"
    return table


def banner(text: str):
    """Print a nice banner centered with the given text."""
    width = get_terminal_size().columns
    if len(text) > width - 2:
        text = text[: width - 5] + "..."
    text = f" {text} "
    print(text.center(width, "="))


@cache # Cache the result of this function
def getPrompt(role: str, scene: str) -> str:
    """Get the prompt for the given role and scene."""
    with open(f"./prompts/{role}/{scene}.md") as f:
        return f.read()
    
def formatResponse(response: str) -> dict[str, str]:
    """Format the response into a dictionary."""
    # Agent response be like:
    # **Thoughts**: <thoughts>
    # **Prompt**: <prompt>
    # **...**: ...
    result = {}
    for line in response.split("\n"):
        if not line.strip():
            continue
        pair = line.split(":", 1)
        if len(pair) != 2:
            continue
        key, value = pair
        key = key.removeprefix("**").removesuffix("**")
        result[key.strip()] = value.strip()
    return result


