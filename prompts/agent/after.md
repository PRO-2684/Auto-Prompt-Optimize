I'll provide you with an existing deduced prompt, and the inputs, expected outputs, real outputs from the LLM app following that prompt and a similarity score. Please:

1. Observe the differences between expected outputs and real outputs with low similarity score (Max score is 5)
2. Propose a better prompt based on your observations. *Completely abandon* the original prompt and craft a new prompt if necessary.
3. If all are similar enough, answer `DONE` in the `Prompt` section instead.

You should always answer in the format of:

```markdown
# Thoughts
Your thoughts, observations, and reasoning.

# Prompt
Your new deduced prompt, or `DONE`.
```

You should **only provide the deduced prompt**, without any prologue in the `Prompt` section. Do not explain your reasoning in the `Prompt` section - use the `Thoughts` section for that.

Here's the existing prompt:

```text
{prompt}
```

Here's our `<input, expected_output, real_output, score>` pairs:

{examples}
