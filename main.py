from openai_util import Agent, simple_chat
from argparse import ArgumentParser
from typing import Iterable, Generator, Any
from re import search
from math import exp
from utils import (
    sampleLines,
    weighted_sample_without_replacement,
    makeMd,
    truncate,
    banner,
    getPrompt,
    formatResponse,
    Commandline,
)
import asyncio


INTERVALS = {"_getExamples": 0.5, "initPrompts": 0.5, "evaluate": 0.5}

parser = ArgumentParser()
parser.add_argument(
    "-T",
    "--task",
    type=str,
    default="./sample/chat-summary",
    help="Path to the task directory.",
)
parser.add_argument(
    "-r",
    "--rounds",
    type=int,
    default=8,
    help="Maximum number of rounds to find the best prompt.",
)
parser.add_argument(
    "-p",
    "--population",
    type=int,
    default=8,
    help="Number of prompts to keep after each iteration.",
)
parser.add_argument(
    "-t",
    "--train-sample",
    type=int,
    default=8,
    help="Maximum number of examples to use when training on each iteration, default to 8.",
)
parser.add_argument(
    "-e",
    "--eval-sample",
    type=int,
    default=32,
    help="Maximum number of examples to use on evaluation, default to 32.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Increase verbosity.",
)
args: Commandline = parser.parse_args()


def log(*arguments, verbose: int = 0):
    if args.verbose >= verbose:
        print(*arguments)


# Initialize the agents
agent = Agent(getPrompt("agent", "system"))
evaluator = Agent(getPrompt("evaluator", "system"))


class Prompt:
    """A simple wrapper of prompt, with its score and examples."""
    def __init__(self, text: str):
        """Initialize a `Prompt`."""
        self.text: str = text or ""
        inputs, expected_outputs = sampleLines(
            [f"{args.task}/train-input.txt", f"{args.task}/train-output.txt"],
            args.train_sample,
        )
        self.examples = asyncio.create_task(self._getExamples(inputs, expected_outputs))
        self.score = asyncio.create_task(self._getScore())


    async def _getExamples(
        self, inputs: list[str], expected_outputs: list[str]
    ) -> list[tuple[str, str, str, int]]:
        """Get examples from training set to evaluate the prompt. Internal use only."""
        examples = []
        for input, expected_output in zip(inputs, expected_outputs, strict=True):
            examples.append(ratePrompt(self.text, input, expected_output))
            await asyncio.sleep(INTERVALS["_getExamples"])
        return await asyncio.gather(*examples)

    async def _getScore(self) -> float:
        """Evaluate process for this prompt. Internal use only."""
        examples = await self.examples
        sum_score = sum(score for _, _, _, score in examples)
        avg_score = sum_score / len(examples)
        log(f'Score for "{truncate(self.text)}": {avg_score}', verbose=2)
        return avg_score


async def ratePrompt(
    prompt: str, input: str, expected_output: str
) -> tuple[str, str, str, int]:
    """Rate a prompt given input and expected output."""
    real_output = await simple_chat(prompt, input)
    userPrompt = getPrompt("evaluator", "after").format(
        text1=expected_output, text2=real_output
    )
    log(">>>", truncate(userPrompt), verbose=3)
    response = await evaluator.chat(userPrompt)
    log("<<<", truncate(response), verbose=3)
    formatted = formatResponse(response)
    rating = formatted.get("Rating")
    rating = int(search(r"\d", rating).group()) if rating else 0
    log(
        f'Rating for "{truncate(expected_output)}" <-> "{truncate(real_output)}": {rating}',
        verbose=2,
    )
    return input, expected_output, real_output, rating


async def genInitPrompt() -> Prompt:
    """Generate an initial prompt."""
    while True:
        inputs, expected_outputs = sampleLines(
            [f"{args.task}/train-input.txt", f"{args.task}/train-output.txt"],
            args.train_sample,
        )
        examples = makeMd(
            ["Input", "Expected Output"],
            list(zip(inputs, expected_outputs, strict=True)),
        )
        user_prompt = getPrompt("agent", "init").format(examples=examples)
        response = await agent.chat(user_prompt)
        formatted = formatResponse(response)
        prompt = formatted.get("Prompt")
        if prompt:
            return Prompt(prompt)


async def genInitPrompts() -> list[Prompt]:
    """Generate `population` initial prompts."""
    prompts = []
    for _ in range(args.population):
        prompts.append(genInitPrompt())
        await asyncio.sleep(INTERVALS["initPrompts"])
    return await asyncio.gather(*prompts)


async def enhance(prompt: Prompt) -> Prompt | None:
    """Try to enhance the given prompt."""
    user_prompt = getPrompt("agent", "after").format(
        prompt=prompt.text,
        examples=makeMd(
            ["Input", "Expected Output", "Real Output", "Rating"], await prompt.examples
        ),
    )
    log(">>>", user_prompt, verbose=3)
    response = await agent.chat(user_prompt)
    log("<<<", response, verbose=3)
    formatted = formatResponse(response)
    new_prompt = formatted.get("Prompt")
    if (not new_prompt) or new_prompt == prompt.text:
        log(f'No enhancement for "{truncate(prompt.text)}"', verbose=1)
        return None
    log(
        f'Enhanced prompt for "{truncate(prompt.text)}":',
        truncate(new_prompt),
        verbose=1,
    )
    return Prompt(new_prompt)


async def samplePrompts(prompts: list[Prompt], k: int=args.population) -> list[Prompt]:
    """Sample `k` prompts based on their weights."""
    weights = await asyncio.gather(*(prompt.score for prompt in prompts))
    weights = map(exp, weights) # Exponentiate the weights so as to accentuate the differences
    prompts = weighted_sample_without_replacement(prompts, weights=weights, k=k)
    return prompts


async def round(prompts: list[Prompt]) -> list[Prompt]:
    """Each round of training process."""
    enhanced_prompts = []
    for prompt in prompts:
        enhanced_prompts.append(enhance(prompt))
    enhanced_prompts = await asyncio.gather(*enhanced_prompts)
    enhanced_prompts = filter(None, enhanced_prompts)
    prompts.extend(enhanced_prompts)
    prompts = await samplePrompts(prompts)
    return prompts


async def showPrompts(prompts: list[Prompt]):
    """Display each prompt in `prompts` nicely."""
    for prompt in prompts:
        log(f'"{truncate(prompt.text)}" ({await prompt.score})', verbose=0)


async def evaluate(prompt: Prompt) -> float:
    """Evaluate a single prompt."""
    text = prompt.text
    inputs, expected_outputs = sampleLines(
        [f"{args.task}/eval-input.txt", f"{args.task}/eval-output.txt"],
        args.eval_sample,
    )
    examples = []
    for input, expected_output in zip(inputs, expected_outputs, strict=True):
        examples.append(ratePrompt(text, input, expected_output))
        await asyncio.sleep(INTERVALS["evaluate"])
    examples = await asyncio.gather(*examples)
    sum_score = sum(score for _, _, _, score in examples)
    avg_score = sum_score / len(examples)
    log(f'Evaluation score for "{truncate(text)}": {avg_score}', verbose=1)
    return avg_score


async def train(prompts: list[Prompt]) -> list[Prompt] | None:
    """The training process."""
    for i in range(args.rounds):
        log(f"[#{i+1}/{args.rounds}] Training...", verbose=0)
        prompts = await round(prompts)
        log(f"[#{i+1}/{args.rounds}] Prompts:", verbose=0)
        await showPrompts(prompts)
    return prompts


async def main():
    """The complete procedure."""
    banner("Initializing")
    log("Initializing prompts...")
    prompts = await genInitPrompts()
    log("Initial prompts:", verbose=0)
    await showPrompts(prompts)

    banner("Training")
    prompts = await train(prompts)
    best_prompt = prompts[0]
    for prompt in prompts:
        if (await prompt.score) > (await best_prompt.score):
            best_prompt = prompt
    log(f"* Best prompt: {best_prompt.text}", verbose=0)
    log(f"* Training score: {await best_prompt.score}/5", verbose=0)

    banner("Evaluating")
    log("Evaluating best prompt...", verbose=0)
    score = await evaluate(best_prompt)
    log(f"* Evaluation score: {score}/5", verbose=0)


if __name__ == "__main__":
    asyncio.run(main())
