I'll provide you an existing deduced prompt, the inputs, expected outputs, real outputs from the LLM app following that prompt and a similarity score. Please:

1. Observe the differences between expected outputs and real outputs with low similarity score (Max score is 5)
2. Propose a better prompt based on your observations. *Completely abandon* the original prompt and craft a new prompt if necessary.

Always answer in the format of:

```markdown
# Thoughts
Your thoughts, observations, and reasoning.

# Prompt
Your new deduced prompt.
```

**Only provide the deduced prompt**, without any prologue/epilogue in the `Prompt` section. Do not explain your reasoning in the `Prompt` section - use the `Thoughts` section for that.

Here's the existing prompt:

```text
{prompt}
```

Here's our `<input, expected_output, real_output, score>` pairs:

{examples}
