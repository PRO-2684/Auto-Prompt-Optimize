from typing import Iterable
from os import get_terminal_size
from functools import cache
from random import sample, choices
from dataclasses import dataclass


@cache
def readLines(path: str) -> list[str]:
    """Read the file from the given `path` and return a list of **non-empty** lines."""
    with open(path) as f:
        result = f.readlines()
    result = map(str.strip, result)
    result = filter(None, result)
    return list(result)


def sampleLines(paths: list[str], n: int) -> list[list[str]]:
    """Sample `n` identical lines from the given `paths`."""
    data = [readLines(path) for path in paths]
    minLineCnt = len(data[0])
    for lines in data:
        minLineCnt = min(minLineCnt, len(lines))
    indices = sample(range(minLineCnt), n)
    sampled = [[lines[i] for i in indices] for lines in data]
    return sampled


def weighted_sample_without_replacement(population, weights, k=1):
    # https://stackoverflow.com/a/43649323/16468609
    weights = list(weights)
    positions = range(len(population))
    indices = []
    while True:
        needed = k - len(indices)
        if not needed:
            break
        for i in choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return [population[i] for i in indices]


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


def truncate(text: str, length: int = 64) -> str:
    """Truncate the given text to the given length."""
    if len(text) > length:
        text = text[: length - 3] + "..."
    return text


def banner(text: str):
    """Print a nice banner centered with the given text."""
    width = get_terminal_size().columns
    if len(text) > width - 2:
        text = text[: width - 5] + "..."
    text = f" {text} "
    print(text.center(width, "="))


@cache  # Cache the result of this function
def getPrompt(role: str, scene: str) -> str:
    """Get the prompt for the given role and scene."""
    with open(f"./prompts/{role}/{scene}.md") as f:
        return f.read()


def formatResponse(response: str) -> dict[str, str]:
    """Format the response into a dictionary."""
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


@dataclass
class Commandline:
    """Dummy class for "typing" commandline arguments."""

    task: str
    """Path to the task directory."""
    rounds: int
    """Maximum number of rounds to find the best prompt."""
    population: int
    """Number of prompts to keep after each iteration."""
    train_sample: int
    """Maximum number of examples to use when training on each iteration."""
    eval_sample: int
    """Maximum number of examples to use on evaluation."""
    cross_ratio: float
    """The ratio of cross-enhancement."""
    verbose: int
    """Verbosity level."""
