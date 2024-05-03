import openai
import atexit
from json import load, dump

# Load configuration
with open("config.json") as f:
    config = load(f)

client = openai.OpenAI(api_key=config["openai_key"], base_url=config["openai_endpoint"])

COST_FACTOR = config["cost_factor"] # Cost per token
tokenLimit = config["single_token_limit"] # Token limit per run
tokensUsed = 0 # Tokens used in this run


def incrementTokensUsed(incr):
    global tokensUsed
    tokensUsed += incr
    if tokensUsed > tokenLimit:
        raise RuntimeError("Token limit exceeded")


def getTokensUsed():
    return tokensUsed


def printUsage():
    print(
        f"Tokens used: {tokensUsed} / {tokenLimit}, estimated cost: ￥{tokensUsed * COST_FACTOR}"
    )


def beforeExit():
    printUsage()
    config["tokens_used"] += tokensUsed
    config["cost"] = config["tokens_used"] * COST_FACTOR
    with open("config.json", "w") as f:
        dump(config, f, indent=4)
    print("Statistics saved.")


atexit.register(beforeExit)


class Agent:
    def __init__(self, system_prompt: str, model="gpt-3.5-turbo"):
        '''
        system_prompt: The system prompt
        model: The model to use'''
        self.system_prompt = system_prompt
        self.memory = []
        self.model = model

    def chat(self, text: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        self.memory.append({"role": "user", "content": text})
        messages.extend(self.memory)
        r = client.chat.completions.create(
            model = self.model,
            messages = messages
        )
        incrementTokensUsed(r.usage.total_tokens)
        self.memory.append({"role": "assistant", "content": r.choices[0].message.content})
        return r.choices[0].message.content

if __name__ == "__main__":
    # Test the agent
    agent = Agent("You are a helpful assistant.", memory_rounds=5)
    while True:
        text = input("You: ")
        if text == "exit":
            break
        print("Assistant:", agent.chat(text))

