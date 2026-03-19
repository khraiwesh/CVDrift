# CVDrift 

## Install

Clone the project and initialize submodules first:

```bash
git clone https://github.com/khraiwesh/CVDrift.git
cd CVDrift
git submodule update --init --recursive
```
Create environment:

```bash
python -m venv venv
```

Then install dependencies:

```bash
pip install -r requirements.txt
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