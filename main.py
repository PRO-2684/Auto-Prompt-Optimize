from openai_util import Agent, simple_chat
from argparse import ArgumentParser
from typing import Iterable, Generator, Any
from utils import sampleLines, makeMd, banner, getPrompt, formatResponse


def getRealOutput(
    prompt: str, data: Iterable[tuple[str, str]]
) -> Generator[tuple[str, str, str], Any, None]:
    """Get the real output for the given data."""
    for input, expected_output in data:
        real_output = simple_chat(prompt, input)
        yield input, expected_output, real_output


def train(
    agent: Agent, task: str, sample: int = 8, rounds: int = 16
) -> str | None:
    """Train the agent with the given data and return the best prompt it finds."""
    train_input = sampleLines(f"{task}/train-input.txt", sample)
    train_output = sampleLines(f"{task}/train-output.txt", sample)
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
    best_prompt = formatted.get("Prompt")
    if not best_prompt:
        return None
    for i in range(rounds):
        examples = makeMd(
            ["Input", "Expected Output", "Real Output"],
            getRealOutput(best_prompt, train_data),
        )
        user_prompt = after_prompt.format(prompt=best_prompt, examples=examples)
        print(f"[#{i+1}/{rounds}] >>>", user_prompt)
        response = agent.chat(user_prompt)
        print(f"[#{i+1}/{rounds}] <<<", response)
        formatted = formatResponse(response)
        new_prompt = formatted.get("Prompt")
        if (not new_prompt) or "DONE" in new_prompt or new_prompt == best_prompt:
            break
        best_prompt = new_prompt
    return best_prompt


def evaluate(
    evaluator: Agent, system_prompt: str, task: str, sample: int = 32
) -> float:
    """Evaluate the agent with the given data and return the score."""
    eval_input = sampleLines(f"{task}/eval-input.txt", sample)
    eval_output = sampleLines(f"{task}/eval-output.txt", sample)
    eval_data = list(zip(eval_input, eval_output, strict=True))
    banner("Evaluation")
    after_prompt = getPrompt("evaluator", "after")
    rating_sum = 0
    for i, output in enumerate(getRealOutput(system_prompt, eval_data)):
        input, expected_output, real_output = output
        prompt = after_prompt.format(text1=expected_output, text2=real_output)
        print(f"[#{i+1}/{sample}] >>>", prompt)
        response = evaluator.chat(prompt)
        print(f"[#{i+1}/{sample}] <<<", response)
        formatted = formatResponse(response)
        rating = formatted.get("Rating")
        rating_sum += int(rating)
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
    # Initialize the agent
    agent = Agent(getPrompt("agent", "system"))

    # Training
    best_prompt = train(agent, task, sample=train_sample, rounds=rounds)
    if not best_prompt:
        print("Cannot find a suitable prompt.")
        return
    print(f"* Best prompt: {best_prompt}")

    # Initialize the evaluator
    evaluator = Agent(getPrompt("evaluator", "system"))

    # Evaluation
    score = evaluate(evaluator, best_prompt, task, sample=eval_sample)
    print(f"* Score: {score}/5")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="./sample/chat-summary",
        help="Path to the task directory.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="Maximum number of rounds to find the best prompt.",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=8,
        help="Maximum number of training examples to use, default to 8.",
    )
    parser.add_argument(
        "--eval-sample",
        type=int,
        default=32,
        help="Maximum number of evaluation examples to use, default to 32.",
    )
    args = parser.parse_args()
    main(args.task, args.rounds, args.train_sample, args.eval_sample)
