# Auto Prompt Optimization

## 🤔 Usage

```text
$ python3 main.py --help
usage: main.py [-h] [--task TASK] [--rounds ROUNDS] [--train-clip TRAIN_CLIP] [--eval-clip EVAL_CLIP]

options:
  -h, --help            show this help message and exit
  --task TASK           Path to the task directory.
  --rounds ROUNDS       Maximum number of rounds to find the best prompt.
  --train-clip TRAIN_CLIP
                        Maximum number of training examples to use, 0 for all.
  --eval-clip EVAL_CLIP
                        Maximum number of evaluation examples to use, 0 for all.
```

The task directory shall be a folder containing the following files:

- `train-input.txt`: input examples
- `train-output.txt`: output examples
- `eval-input.txt`: input to be evaluated
- `eval-output.txt`: expected output

## 🎉 Acknowledgements

- [microsoft/EvoPrompt: Automatic Prompt Optimization (github.com)](https://github.com/microsoft/EvoPrompt)
- [HouYi (github.com)](https://github.com/LLMSecurity/HouYi)
