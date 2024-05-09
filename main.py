from openai_util import Agent, simple_chat
from argparse import ArgumentParser
from typing import Iterable, Generator, Any
from utils import sampleLines, makeMd, banner, getPrompt, formatResponse
from re import search


def getRealOutputs(
    prompt: str, data: Iterable[tuple[str, str]]
) -> Generator[tuple[str, str, str], Any, None]:
    """Get the real output for the given data."""
    for input, expected_output in data:
        real_output = simple_chat(prompt, input)
        yield input, expected_output, real_output


def getRealOutputsAndRatings(
    evaluator: Agent, sys_prompt: str, data: Iterable[tuple[str, str]], log: bool = False
) -> Generator[tuple[str, str, str, int], Any, None]:
    """Append the real output and rating at the end of the given data."""
    after_prompt = getPrompt("evaluator", "after")
    for input, expected_output in data:
        real_output = simple_chat(sys_prompt, input)
        prompt = after_prompt.format(text1=expected_output, text2=real_output)
        log and print(">>>", prompt)
        response = evaluator.chat(prompt)
        log and print("<<<", response)
        formatted = formatResponse(response)
        rating = formatted.get("Rating")
        rating = int(search(r"\d", rating).group()) if rating else 0
        yield input, expected_output, real_output, rating


def getAverageScore(evaluated_data: list[tuple[str, str, str, int]]) -> float:
    sum_score = 0
    for _, _, _, score in evaluated_data:
        sum_score += score
    return sum_score / len(evaluated_data)


def train(
    agent: Agent, evaluator: Agent, task: str, sample: int = 8, rounds: int = 16
) -> str | None:
    """Train the agent with the given data and return the best prompt it finds."""
    train_input, train_output = sampleLines([f"{task}/train-input.txt", f"{task}/train-output.txt"], sample)
    train_data = list(zip(train_input, train_output, strict=True))
    banner("Training")
    init_prompt = getPrompt("agent", "init")
    after_prompt = getPrompt("agent", "after")
    user_prompt = init_prompt.format(
        examples=makeMd(["Input", "Expected Output"], train_data)
    )
    print(">>>", user_prompt)
    response = agent.chat(user_prompt)
    print("<<<", response)
    formatted = formatResponse(response)
    prev_prompt = formatted.get("Prompt")
    print("*prev_prompt:", prev_prompt)
    if not prev_prompt:
        return None
    best_score = -1
    best_prompt = prev_prompt
    best_evaluated_data = []
    for i in range(rounds):
        prev_evaluated_data = list(getRealOutputsAndRatings(evaluator, prev_prompt, train_data))
        prev_score = getAverageScore(prev_evaluated_data)
        if prev_score > best_score:
            best_score = prev_score
            best_prompt = prev_prompt
            best_evaluated_data = prev_evaluated_data
        examples = makeMd(
            ["Input", "Expected Output", "Real Output", "Rating"],
            best_evaluated_data
        )
        user_prompt = after_prompt.format(prompt=best_prompt, examples=examples)
        print(f"[#{i+1}/{rounds}] >>>", user_prompt)
        print("*best_prompt:", best_prompt)
        response = agent.chat(user_prompt)
        print(f"[#{i+1}/{rounds}] <<<", response)
        formatted = formatResponse(response)
        new_prompt = formatted.get("Prompt")
        print("*new_prompt:", new_prompt)
        if (not new_prompt) or "DONE" in new_prompt or new_prompt == prev_prompt:
            break
        prev_prompt = new_prompt
    return best_prompt


def evaluate(
    evaluator: Agent, system_prompt: str, task: str, sample: int = 32
) -> float:
    """Evaluate the agent with the given data and return the score."""
    eval_input, eval_output = sampleLines([f"{task}/eval-input.txt", f"{task}/eval-output.txt"], sample)
    eval_data = list(zip(eval_input, eval_output, strict=True))
    banner("Evaluation")
    rating_sum = 0
    for i, output in enumerate(getRealOutputsAndRatings(evaluator, system_prompt, eval_data, True)):
        input, expected_output, real_output, rating = output
        rating_sum += rating
        print(f"[#{i+1}/{sample}] {rating}/5")
    return rating_sum / sample if sample else 0


def main(
    task: str = "./sample/chat-summary",
    rounds: int = 8,
    train_sample: int = 8,
    eval_sample: int = 8,
):
    """Main function to train and evaluate the agent on the given task.

    `task`: The path to the task directory.
    `rounds`: The maximum number of rounds to find the best prompt."""
    # Initialize the agents
    agent = Agent(getPrompt("agent", "system"))
    evaluator = Agent(getPrompt("evaluator", "system"))

    # Training
    best_prompt = train(agent, evaluator, task, sample=train_sample, rounds=rounds)
    if not best_prompt:
        print("Cannot find a suitable prompt.")
        return
    print(f"* Best prompt: {best_prompt}")

    # Evaluation
    score = evaluate(evaluator, best_prompt, task, sample=eval_sample)
    print(f"* Score: {score}/5")


if __name__ == "__main__":
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
    args = parser.parse_args()
    main(args.task, args.rounds, args.train_sample, args.eval_sample)
