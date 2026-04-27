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

Training (default episode value 1):
```bash
python3 -m codebase.train config_name --train
```

Training (5000 as the number of episode):

```bash
python3 -m codebase.train config_name --train 5000
```

See one model in action:

```bash
python3 -m codebase.train config_name
```

Run multiple train:

```bash
python3 codebase/run_multiple_train.py
```

See multiple model in action:

```bash
python3 codebase/run_multiple_eval.py
```