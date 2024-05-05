from openai_util import Agent, simple_chat
from argparse import ArgumentParser
from typing import Iterable, Generator, Any
from utils import readLines, makeTable, banner, getPrompt, formatResponse


def getRealOutput(
    prompt: str, data: Iterable[tuple[str, str]]
) -> Generator[tuple[str, str, str], Any, None]:
    """Get the real output for the given data."""
    for input, expected_output in data:
        real_output = simple_chat(prompt, input)
        yield input, expected_output, real_output


def train(
    agent: Agent, data: Iterable[tuple[str, str]], rounds: int = 16
) -> str | None:
    """Train the agent with the given data and return the best prompt it finds."""
    banner("Training")
    init_prompt = getPrompt("agent", "init")
    after_prompt = getPrompt("agent", "after")
    user_prompt = init_prompt.format(table=makeTable(["Input", "Expected Output"], data))
    print(">>>", user_prompt)
    response = agent.chat(user_prompt)
    print("<<<", response)
    formatted = formatResponse(response)
    best_prompt = formatted.get("Prompt")
    if not best_prompt:
        return None
    for i in range(rounds):
        table = makeTable(
            ["Input", "Expected Output", "Real Output"],
            getRealOutput(best_prompt, data),
        )
        user_prompt = after_prompt.format(prompt=best_prompt, table=table)
        print(f"[#{i+1}/{rounds}] >>>", user_prompt)
        response = agent.chat(user_prompt)
        print(f"[#{i+1}/{rounds}] <<<", response)
        formatted = formatResponse(response)
        best_prompt = formatted.get("Prompt")
        if not best_prompt or best_prompt == "DONE":
            break
    return best_prompt


def evaluate(system_prompt: str, data: Iterable[tuple[str, str]]) -> float:
    """Evaluate the agent with the given data and return the score."""
    banner("Evaluation")
    return 0 # TODO: Implement this


def main(
    task: str = "./sample/chat-summary",
    rounds: int = 16,
    train_clip: int = 0,
    eval_clip: int = 0,
):
    """Main function to train and evaluate the agent on the given task.

    `task`: The path to the task directory.
    `rounds`: The maximum number of rounds to find the best prompt."""
    # Initialize the agent
    agent = Agent(getPrompt("agent", "system"))

    # Training
    train_input = readLines(f"{task}/train-input.txt", max_lines=train_clip)
    train_output = readLines(f"{task}/train-output.txt", max_lines=train_clip)
    train_data = zip(train_input, train_output, strict=True)
    best_prompt = train(agent, train_data, rounds=rounds)
    if not best_prompt:
        print("Cannot find a suitable prompt.")
        return
    print(f"Best prompt: {best_prompt}")

    # Evaluation
    eval_input = readLines(f"{task}/eval-input.txt", max_lines=eval_clip)
    eval_output = readLines(f"{task}/eval-output.txt", max_lines=eval_clip)
    eval_data = zip(eval_input, eval_output)
    score = evaluate(best_prompt, eval_data)
    print(f"Score: {score}")


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
        default=16,
        help="Maximum number of rounds to find the best prompt.",
    )
    parser.add_argument(
        "--train-clip",
        type=int,
        default=0,
        help="Maximum number of training examples to use, 0 for all.",
    )
    parser.add_argument(
        "--eval-clip",
        type=int,
        default=0,
        help="Maximum number of evaluation examples to use, 0 for all.",
    )
    args = parser.parse_args()
    main(args.task, args.rounds, args.train_clip, args.eval_clip)
