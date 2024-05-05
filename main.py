from openai_util import Agent, simple_chat
from argparse import ArgumentParser
from typing import Iterable

def initAgent(system_prompt_path: str):
    '''Initializes the agent with the system prompt from the given path.'''
    with open(system_prompt_path) as f:
        system_prompt = f.read()
    agent = Agent(system_prompt)
    return agent

def read(path: str):
    '''Read the file from the given `path` and return a list of lines.'''
    result = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(line)
    return result

def train(agent: Agent, data: Iterable[tuple[str, str]], rounds: int=10) -> str | None:
    '''Train the agent with the given data and return the best prompt it finds.'''
    pass

def evaluate(system_prompt: str, data: Iterable[tuple[str, str]]) -> float:
    '''Evaluate the agent with the given data and return the score.'''
    pass

def main(task: str="./sample/chat-summary", rounds: int=16, train_clip: int=0, eval_clip: int=0):
    '''Main function to train and evaluate the agent on the given task.
    
    `task`: The path to the task directory.
    `rounds`: The maximum number of rounds to find the best prompt.'''
    agent = initAgent("./prompts/agent.md")

    train_input = read(f"{task}/train-input.txt")
    train_output = read(f"{task}/train-output.txt")
    if train_clip > 0: # Clip the training data
        train_input = train_input[:train_clip]
        train_output = train_output[:train_clip]
    train_data = zip(train_input, train_output, strict=True)
    best_prompt = train(agent, train_data, rounds=rounds)
    if not best_prompt:
        print("Cannot find a suitable prompt.")
        return
    print(f"Best prompt: {best_prompt}")

    eval_input = read(f"{task}/eval-input.txt")
    eval_output = read(f"{task}/eval-output.txt")
    if eval_clip > 0: # Clip the evaluation data
        eval_input = eval_input[:eval_clip]
        eval_output = eval_output[:eval_clip]
    eval_data = zip(eval_input, eval_output)
    score = evaluate(best_prompt, eval_data)
    print(f"Score: {score}")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="./sample/chat-summary", help="Path to the task directory.")
    parser.add_argument("--rounds", type=int, default=16, help="Maximum number of rounds to find the best prompt.")
    parser.add_argument("--train-clip", type=int, default=0, help="Maximum number of training examples to use, 0 for all.")
    parser.add_argument("--eval-clip", type=int, default=0, help="Maximum number of evaluation examples to use, 0 for all.")
    args = parser.parse_args()
    main(args.task, args.rounds, args.train_clip, args.eval_clip)
