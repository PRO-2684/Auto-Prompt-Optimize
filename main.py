from openai_util import Agent

def initAgent(system_prompt_path: str):
    with open(system_prompt_path) as f:
        system_prompt = f.read()
    agent = Agent(system_prompt)
    return agent

def main():
    agent = initAgent("./data/prompts/agent.md")
    ...

if __name__ == "__main__":
    main()

