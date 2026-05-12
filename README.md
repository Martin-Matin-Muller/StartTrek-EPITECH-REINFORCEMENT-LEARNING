# Start Trek (2026)

This repository contains all the implementation of the training policy and evaluation protocol, as well as all the different agents and neural networks code.\
You will also find a scientific like report of the project with a jupyter notebook which explain how to implement a neural network model in the report folder.

# Setup

Create the venv :
```bash
python3 -m venv StartTrekVenv

source StartTrekVenv/bin/activate

pip install -r requirement.txt
```

Create mutliple configs:
```bash
python3 codebase/utils/config_gen.py 
```

# Usage:

Training:

```bash
python3 -m codebase.train config_name "number of episode" "seed"
```

See one model in action:

```bash
python3 -m codebase.visualization [agent].pt "number of episode" "seed"
```

Evaluate all the models:

```bash
python3 -m codebase.eval "number of episode" "seed"
```

# Video report

Here is a timelapse video of a training : [Luni's training](https://youtu.be/Sm6sS1p6iwI?si=YYU6UAXroNc8NrUP)
\
This video is inspired by [Ai warehouse channel](https://www.youtube.com/@aiwarehouse)