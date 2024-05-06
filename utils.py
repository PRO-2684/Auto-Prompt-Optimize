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


def makeMd(header: Iterable[str], data: Iterable[tuple]) -> str:
    """Make a nice Markdown display from the given header and data."""
    result = ""
    i = 0
    for row in data:
        i += 1
        result += f"## Example {i}\n"
        for head, value in zip(header, row):
            result += f"### {head}\n{value}\n"
    return result


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
    # # Thoughts
    # <thoughts>
    # # Prompt
    # <prompt>
    # # ...
    # ...
    result = {}
    lines = response.strip().split("\n")
    key = ""
    for line in lines:
        if line.startswith("# "):
            key = line[2:]
            result[key] = ""
        else:
            result[key] += line + "\n"
    # Strip the values
    for key in result:
        result[key] = result[key].strip()
    return result
