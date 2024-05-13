# Auto Prompt Optimization

## 📥 Setup

### 📦 Dependencies

```bash
pip install -r requirements.txt
```

### ⚙️ Configuration

Create a file named `config.json` in the root directory of this repo. Here's how your `config.json` should look like:

```jsonc
{
    "openai_key": "sk-...", // OpenAI API key (Required)
    "openai_endpoint": "https://api.openai.com", // OpenAI API endpoint
    "rpm": 60, // Requests per minute
    "cost_factor": 2e-06, // Cost factor
    "single_token_limit": 300000, // Token limit per run
    "tokens_used": 0, // Tokens used
    "cost": 0 // Total cost
}
```

## 🤔 Usage

```text
$ python3 main.py --help
usage: main.py [-h] [-T TASK] [-r ROUNDS] [-p POPULATION] [-t TRAIN_SAMPLE] [-e EVAL_SAMPLE] [-v]

options:
  -h, --help            show this help message and exit
  -T TASK, --task TASK  Path to the task directory.
  -r ROUNDS, --rounds ROUNDS
                        Maximum number of rounds to find the best prompt.
  -p POPULATION, --population POPULATION
                        Number of prompts to keep after each iteration.
  -t TRAIN_SAMPLE, --train-sample TRAIN_SAMPLE
                        Maximum number of examples to use when training on each iteration, default to 8.
  -e EVAL_SAMPLE, --eval-sample EVAL_SAMPLE
                        Maximum number of examples to use on evaluation, default to 32.
  -v, --verbose         Increase verbosity.
  ```

The task directory shall be a folder containing the following files:

- `train-input.txt`: input examples
- `train-output.txt`: output examples
- `eval-input.txt`: input to be evaluated
- `eval-output.txt`: expected output

## 🔄️ Procedure

### Overview

1. Given $t$ samples from the training set, let the agent generate $k$ initial prompts.
2. [Evaluate](#evaluation) each prompt on randomly-selected $e$ samples from the training set.
3. Let the agent enhance each prompt based on corresponding evaluation result, so we now have $2k$ prompts.
4. [Evaluate](#evaluation) each prompt (if not already evaluated) on randomly-selected $e$ samples from the training set, and randomly select $k$ prompts with the weighted probability of their scores.
5. Repeat steps 3-4 for $r$ rounds.
6. Output the best prompt with the highest score.

### Evaluation

Given a prompt and a set of example inputs and outputs:

1. Generate real outputs using the prompt and given inputs.
2. Rate the similarity between the generated outputs and the expected outputs, and assign a score $\in\{0, 1, 2, 3, 4, 5\}$ to each output.
3. Assign the prompt's score as the average of aforementioned scores.

## 📃 TODO

- Make self-enhancement probabilistic. The probability of enhancing a prompt should be inversely proportional to its score.
- Add probabilistic cross-enhancement. The probability of enhancing a prompt with another prompt should be proportional to their scores.

## 🎉 Acknowledgements

- [microsoft/EvoPrompt: Automatic Prompt Optimization (github.com)](https://github.com/microsoft/EvoPrompt)
- [HouYi (github.com)](https://github.com/LLMSecurity/HouYi)
