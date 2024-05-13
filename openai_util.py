import openai
from backoff import on_exception, expo
import asyncio
from ratelimit import limits, RateLimitException
from atexit import register as on_exit
from json import load, dump

# Load configuration
with open("config.json") as f:
    config: dict = load(f)

client = openai.AsyncOpenAI(
    api_key=config["openai_key"], base_url=config.get("openai_endpoint", None)
)

COST_FACTOR = config.get("cost_factor", 2e-06)  # Cost per token
tokenLimit = config.get("single_token_limit", 300000)  # Token limit per run
tokensUsed = 0  # Tokens used in this run


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
    config["tokens_used"] = config.get("tokens_used", 0) + tokensUsed
    config["cost"] = config["tokens_used"] * COST_FACTOR
    with open("config.json", "w") as f:
        dump(config, f, indent=4)
    print("Statistics saved.")


on_exit(beforeExit)


@on_exception(lambda: expo(max_value=60), (RateLimitException, openai.RateLimitError), max_tries=32)
@limits(calls=config.get("rpm", 60), period=60)
async def chat_with_backoff(**kwargs):
    return await client.chat.completions.create(**kwargs)


class Agent:
    def __init__(self, system_prompt: str, model="gpt-3.5-turbo"):
        """
        system_prompt: The system prompt
        model: The model to use"""
        self.system_prompt = system_prompt
        self.model = model

    async def chat(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        r = await chat_with_backoff(model=self.model, messages=messages)
        incrementTokensUsed(r.usage.total_tokens)
        return r.choices[0].message.content


async def simple_chat(system_prompt, msg, model="gpt-3.5-turbo") -> str:
    r = await chat_with_backoff(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ],
    )
    incrementTokensUsed(r.usage.total_tokens)
    return r.choices[0].message.content


if __name__ == "__main__":
    async def main():
        # Test the agent
        agent = Agent("You are a helpful assistant.")
        while True:
            text = input("You: ")
            if text == "exit":
                break
            print("Assistant:", await agent.chat(text))

    asyncio.run(main())
