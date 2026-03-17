# CVDrift Suggested

Repository is now organized in the requested style:

```text
project/
│
├── main.py
├── cvdrift/
│   ├── preprocessing.py
│   ├── window_selection.py
│   ├── drift_detection.py
│
├── experiments/
│   └── run_experiments.py
│
├── datasets/
│
├── requirements.txt
├── README.md
```

## Usage

Single file:

```bash
python main.py --file "path/to/log.xes" --drift routing
```

Multiple files:

```bash
python main.py --files "log1.xes" "log2.xes" --drift duration routing --out output/results.csv
```

Folder batch:

```bash
python main.py --dir "path/to/folder" --drift duration routing arrival --out output/batch.csv --gt-mode ceravolo
```

## Install

```bash
pip install -r requirements.txt
```
