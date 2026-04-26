Create the venv :
```bash
python3 -m venv StartTrekVenv

source StartTrekVenv/bin/activate

pip install -r requirement.txt
```

Create mutliple configs:
python codebase/utils/config_gen.py 

Run:
    training (default episode value 1):
    python3 codebase/train.py model_name --train

    training (5000 as the number of episode):
    python3 codebase/train.py model_name --train 5000

    check one model in action:
    python3 codebase/train.py model_name

    run multiple train:
    python3 codebase/run_multiple_train.py

    check multiple model in action:
    python3 codebase/run_multiple_eval.py