import openai
import atexit
from json import load, dump

# Load configuration
with open("config.json") as f:
    config = load(f)

client = openai.OpenAI(api_key=config["openai_key"], base_url=config["openai_endpoint"])

# Token limit
COST_FACTOR = 0.002 / 1000
tokensLimit = 1000000
tokensUsed = 0


def incrementTokensUsed(incr):
    global tokensUsed
    tokensUsed += incr
    if tokensUsed > tokensLimit:
        raise Exception("Token limit exceeded")


def getTokensUsed():
    return tokensUsed


def printUsage():
    print(
        f"Tokens used: {tokensUsed} / {tokensLimit}, estimated cost: ￥{tokensUsed * COST_FACTOR}"
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
    def __init__(self, system_prompt: str, memory_rounds: int = 0, model="gpt-3.5-turbo"):
        '''
        system_prompt: The prompt to be used for the system messages
        memory_rounds: The maximum rounds of chats to store in the memory'''
        self.system_prompt = system_prompt
        self.memory = []
        self.memory_size = memory_rounds * 2
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
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
            self.memory.pop(0)
        return r.choices[0].message.content

if __name__ == "__main__":
    # Test the agent
    agent = Agent("You are a helpful assistant.", memory_rounds=5)
    while True:
        text = input("You: ")
        if text == "exit":
            break
        print("Assistant:", agent.chat(text))

