# Auto Prompt Optimization

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

## 🤔 Usage

```text
$ python3 main.py --help
usage: main.py [-h] [-T TASK] [-r ROUNDS] [-t TRAIN_SAMPLE] [-e EVAL_SAMPLE]

options:
  -h, --help            show this help message and exit
  -T TASK, --task TASK  Path to the task directory.
  -r ROUNDS, --rounds ROUNDS
                        Maximum number of rounds to find the best prompt.
  -t TRAIN_SAMPLE, --train-sample TRAIN_SAMPLE
                        Maximum number of examples to use when training on each iteration, default to 8.
  -e EVAL_SAMPLE, --eval-sample EVAL_SAMPLE
                        Maximum number of examples to use on evaluation, default to 32.
```

The task directory shall be a folder containing the following files:

- `train-input.txt`: input examples
- `train-output.txt`: output examples
- `eval-input.txt`: input to be evaluated
- `eval-output.txt`: expected output

## 🎉 Acknowledgements

- [microsoft/EvoPrompt: Automatic Prompt Optimization (github.com)](https://github.com/microsoft/EvoPrompt)
- [HouYi (github.com)](https://github.com/LLMSecurity/HouYi)
