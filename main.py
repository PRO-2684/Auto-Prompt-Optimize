from openai_util import Agent, simple_chat
from argparse import ArgumentParser
from typing import Iterable, Generator, Any
from utils import (
    sampleLines,
    makeMd,
    truncate,
    banner,
    getPrompt,
    formatResponse,
    Commandline,
)
from re import search
from random import choices
import asyncio


INTERVALS = {"_getExamples": 0.5, "initPrompts": 0.5}

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
    def __init__(self, text: str):
        self.text: str = text or ""
        inputs, expected_outputs = sampleLines(
            [f"{args.task}/train-input.txt", f"{args.task}/train-output.txt"],
            args.train_sample,
        )
        self.examples = asyncio.create_task(self._getExamples(inputs, expected_outputs))
        self.score = asyncio.create_task(self._getScore())

    async def _getExample(
        self, input: str, expected_output: str
    ) -> tuple[str, str, str, int]:
        real_output = await simple_chat(self.text, input)
        prompt = getPrompt("evaluator", "after").format(
            text1=expected_output, text2=real_output
        )
        log(">>>", truncate(prompt), verbose=3)
        response = await evaluator.chat(prompt)
        log("<<<", truncate(response), verbose=3)
        formatted = formatResponse(response)
        rating = formatted.get("Rating")
        rating = int(search(r"\d", rating).group()) if rating else 0
        log(
            f'Rating for "{truncate(expected_output)}" <-> "{truncate(real_output)}": {rating}',
            verbose=2,
        )
        return input, expected_output, real_output, rating

    async def _getExamples(
        self, inputs: list[str], expected_outputs: list[str]
    ) -> list[tuple[str, str, str, int]]:
        examples = []
        for input, expected_output in zip(inputs, expected_outputs):
            examples.append(self._getExample(input, expected_output))
            await asyncio.sleep(INTERVALS["_getExamples"])
        return await asyncio.gather(*examples)

    async def _getScore(self) -> float:
        examples = await self.examples
        sum_score = sum(score for _, _, _, score in examples)
        avg_score = sum_score / len(examples)
        log(f'Score for "{truncate(self.text)}": {avg_score}', verbose=2)
        return avg_score


async def genInitPrompt() -> Prompt:
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
    prompts = []
    for _ in range(args.population):
        prompts.append(genInitPrompt())
        await asyncio.sleep(INTERVALS["initPrompts"])
    return await asyncio.gather(*prompts)


async def enhance(prompt: Prompt) -> Prompt | None:
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
    if (not new_prompt) or "DONE" in new_prompt or new_prompt == prompt.text:
        log(f'No enhancement for "{truncate(prompt.text)}"', verbose=1)
        return None
    log(f'Enhanced prompt for "{truncate(prompt.text)}":', truncate(new_prompt), verbose=1)
    return Prompt(new_prompt)


async def round(prompts: list[Prompt]) -> list[Prompt]:
    enhanced_prompts = []
    for prompt in prompts:
        enhanced_prompts.append(enhance(prompt))
    enhanced_prompts = await asyncio.gather(*enhanced_prompts)
    enhanced_prompts = filter(None, enhanced_prompts)
    prompts.extend(enhanced_prompts)
    # Sample the best prompts with weighted probability
    weights = await asyncio.gather(*(prompt.score for prompt in prompts))
    prompts = choices(prompts, weights=weights, k=args.population)
    return prompts


async def train() -> list[Prompt] | None:
    log("Initializing prompts", verbose=0)
    prompts = await genInitPrompts()
    log("Initial prompts:", [f'"{truncate(prompt.text)}" ({await prompt.score})' for prompt in prompts], verbose=0)
    for i in range(args.rounds):
        log(f"[#{i+1}/{args.rounds}] Training", verbose=0)
        prompts = await round(prompts)
    return prompts

async def main():
    prompts = await train()
    best_prompt = prompts[0]
    for prompt in prompts:
        if (await prompt.score) > (await best_prompt.score):
            best_prompt = prompt
    log(f"* Best prompt: {best_prompt.text}", verbose=0)
    log(f"* Score: {await best_prompt.score}/5", verbose=0)

if __name__ == "__main__":
    asyncio.run(main())


# def getRealOutputsAndRatings(
#     evaluator: Agent, sys_prompt: str, data: Iterable[tuple[str, str]], log: bool = False
# ) -> Generator[tuple[str, str, str, int], Any, None]:
#     """Append the real output and rating at the end of the given data."""
#     after_prompt = getPrompt("evaluator", "after")
#     for input, expected_output in data:
#         real_output = simple_chat(sys_prompt, input)
#         prompt = after_prompt.format(text1=expected_output, text2=real_output)
#         log and print(">>>", prompt)
#         response = evaluator.chat(prompt)
#         log and print("<<<", response)
#         formatted = formatResponse(response)
#         rating = formatted.get("Rating")
#         rating = int(search(r"\d", rating).group()) if rating else 0
#         yield input, expected_output, real_output, rating


# def getAverageScore(evaluated_data: list[tuple[str, str, str, int]]) -> float:
#     sum_score = 0
#     for _, _, _, score in evaluated_data:
#         sum_score += score
#     return sum_score / len(evaluated_data)


# def train(
#     agent: Agent, evaluator: Agent, task: str, sample: int = 8, rounds: int = 16
# ) -> str | None:
#     """Train the agent with the given data and return the best prompt it finds."""
#     train_input, train_output = sampleLines([f"{task}/train-input.txt", f"{task}/train-output.txt"], sample)
#     train_data = list(zip(train_input, train_output, strict=True))
#     banner("Training")
#     init_prompt = getPrompt("agent", "init")
#     after_prompt = getPrompt("agent", "after")
#     user_prompt = init_prompt.format(
#         examples=makeMd(["Input", "Expected Output"], train_data)
#     )
#     print(">>>", user_prompt)
#     response = agent.chat(user_prompt)
#     print("<<<", response)
#     formatted = formatResponse(response)
#     prev_prompt = formatted.get("Prompt")
#     print("*prev_prompt:", prev_prompt)
#     if not prev_prompt:
#         return None
#     best_score = -1
#     best_prompt = prev_prompt
#     best_evaluated_data = []
#     for i in range(rounds):
#         prev_evaluated_data = list(getRealOutputsAndRatings(evaluator, prev_prompt, train_data))
#         prev_score = getAverageScore(prev_evaluated_data)
#         if prev_score > best_score:
#             best_score = prev_score
#             best_prompt = prev_prompt
#             best_evaluated_data = prev_evaluated_data
#         examples = makeMd(
#             ["Input", "Expected Output", "Real Output", "Rating"],
#             best_evaluated_data
#         )
#         user_prompt = after_prompt.format(prompt=best_prompt, examples=examples)
#         print(f"[#{i+1}/{rounds}] >>>", user_prompt)
#         print("*best_prompt:", best_prompt)
#         response = agent.chat(user_prompt)
#         print(f"[#{i+1}/{rounds}] <<<", response)
#         formatted = formatResponse(response)
#         new_prompt = formatted.get("Prompt")
#         print("*new_prompt:", new_prompt)
#         if (not new_prompt) or "DONE" in new_prompt or new_prompt == prev_prompt:
#             break
#         prev_prompt = new_prompt
#     return best_prompt


# def evaluate(
#     evaluator: Agent, system_prompt: str, task: str, sample: int = 8
# ) -> float:
#     """Evaluate the agent with the given data and return the score."""
#     eval_input, eval_output = sampleLines([f"{task}/eval-input.txt", f"{task}/eval-output.txt"], sample)
#     eval_data = list(zip(eval_input, eval_output, strict=True))
#     banner("Evaluation")
#     rating_sum = 0
#     for i, output in enumerate(getRealOutputsAndRatings(evaluator, system_prompt, eval_data, True)):
#         _, _, _, rating = output
#         rating_sum += rating
#         print(f"[#{i+1}/{sample}] {rating}/5")
#     return rating_sum / sample if sample else 0


# def main(
#     task: str = "./sample/chat-summary",
#     rounds: int = 8,
#     train_sample: int = 8,
#     eval_sample: int = 8,
# ):
#     """Main function to train and evaluate the agent on the given task.

#     `task`: The path to the task directory.
#     `rounds`: The maximum number of rounds to find the best prompt."""

#     # Training
#     best_prompt = train(agent, evaluator, task, sample=train_sample, rounds=rounds)
#     if not best_prompt:
#         print("Cannot find a suitable prompt.")
#         return
#     print(f"* Best prompt: {best_prompt}")

#     # Evaluation
#     score = evaluate(evaluator, best_prompt, task, sample=eval_sample)
#     print(f"* Score: {score}/5")


# if __name__ == "__main__":
#     main(args.task, args.rounds, args.train_sample, args.eval_sample)
