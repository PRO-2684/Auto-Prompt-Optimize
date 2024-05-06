I'll provide you with an existing deduced prompt, and the inputs, expected outputs and real outputs from the LLM app following that prompt. Please:

1. Evaluate the real outputs and see if they're similar enough to the expected outputs.
2. If they're not similar enough, propose a better prompt in the `Prompt` section. (you can *completely abandon* the original prompt if the outputs are very different)
3. If they're similar enough, answer `DONE` in the `Prompt` section.

Here's the existing prompt:

```text
{prompt}
```

Here's our `<input, expected_output, real_output>` pairs:

{examples}

You should always answer in the format of:

```markdown
# Thoughts
Your thoughts, observations, and reasoning.

# Prompt
Your new deduced prompt, or `DONE`.
```

You should **only provide the deduced prompt**, without any special prefix in the `Prompt` section. Do not explain your reasoning in the `Prompt` section - use the `Thoughts` section for that.
