from typing import Iterable
from os import get_terminal_size
from functools import cache
from random import sample


@cache
def readLines(path: str) -> list[str]:
    """Read the file from the given `path` and return a list of **non-empty** lines."""
    with open(path) as f:
        result = f.readlines()
    result = map(str.strip, result)
    result = filter(None, result)
    return list(result)


def sampleLines(path: str, n: int) -> list[str]:
    """Sample `n` lines from the file at the given `path`."""
    lines = readLines(path)
    return sample(lines, n)


def makeMd(header: Iterable[str], data: Iterable[tuple]) -> str:
    """Make a nice Markdown display from the given header and data."""
    result = ""
    i = 0
    for row in data:
        i += 1
        result += f"## Example {i}\n"
        for head, value in zip(header, row):
            result += f"### {head}\n```text\n{value}\n```\n\n"
    return result.strip()


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
