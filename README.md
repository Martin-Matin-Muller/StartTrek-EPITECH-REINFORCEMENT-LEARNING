Create the venv :
```bash
python3 -m venv StartTrekVenv

source StartTrekVenv/bin/activate

pip install -r requirement.txt
```

Create mutliple configs: need to change config generator
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
python3 -m codebase.visualization config_name "number of episode" "seed"
```

Evaluate all the models:

```bash
python3 -m codebase.eval "number of episode" "seed"
```