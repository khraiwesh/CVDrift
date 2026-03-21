# CVDrift

A concept drift detection framework for process mining event logs. CVDrift uses coefficient of variation (CV) analysis over rolling time-series windows to detect temporal and control-flow concept drifts in business processes.

## Project Structure

```text
cvdrift-suggested/
│
├── main.py                        # CLI entry point (single file, batch, folder modes)
├── requirements.txt               # Python dependencies
├── README.md
│
├── cvdrift/                       # Core library
│   ├── __init__.py
│   ├── preprocessing.py           # Log loading & preprocessing utilities
│   ├── window_selection.py        # Sliding/rolling window strategies
│   ├── drift_detection.py         # CV-based drift detection logic
│   └── pipeline/                  # Modular pipeline components
│       ├── io.py                  #   Event log I/O (XES, CSV, MXML)
│       ├── preprocessing.py       #   Pipeline-level preprocessing
│       ├── window_selection.py    #   Window selection for pipeline
│       ├── rolling.py             #   Rolling window computations
│       ├── series_duration.py     #   Duration time-series extraction
│       ├── series_arrival.py      #   Arrival time-series extraction
│       ├── series_routing.py      #   Routing time-series extraction
│       ├── drift_detection.py     #   Pipeline drift detection step
│       ├── consensus.py           #   Multi-perspective consensus logic
│       └── runner.py              #   Pipeline orchestration
│

│
└── experiments/                   # Reproducible experiment scripts & data
    │
    ├── Experiment#1 (Temporal Perspective)/
    │   ├
    │   ├── run_experiments_CVDrift.py           # CVDrift experiment driver
    │   ├── run_experiments_OCPA.py              # OCPA baseline driver
    │   ├── compare_clean_tolerance_micro.py     # Tolerance comparison analysis
    │   ├── Datasets/                            # XES logs (100/1000/3000 cases,
    │   │   └── *.xes                            #   10–60 min, 0–20% noise)
    |   |   |__ Dataset generator                # Dataset script generator + the process model
    │   ├── output/                              # CSV results & evaluation files
    │   └── ex_concept_drift (OCPA)/             # OCPA baseline code & config
                                              (https://github.com/niklasadams/ex_concept_drift)

    │
    └── Experiment#2 (Control Flow Perspective)/
        ├── run_experiments.py                   # Experiment runner
        ├── evaluate.py                          # Evaluation logic as porposed Cdrift benchmark
        ├── plot_noise_accuracy_raw200.py         # Accuracy vs. noise plots
        ├── environment.yaml                     # Conda environment spec
        ├── algorithm_results_with_CVDrift.csv    # Combined algorithm results
        ├── Datasets/                            # Benchmark event logs
        │   ├── Bose/                            #   Bose et al. logs
        │   ├── Ceravolo/                        #   Ceravolo et al. logs
        │   └── Ostovar/                         #   Ostovar et al. logs (Atomic,
        │       └── *.xes                        #     Composite, Nested patterns)
        └── output/                              # Results & evaluation output
            ├── my_results.csv
            
```

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, matplotlib, ruptures, python-dateutil, pm4py

## Usage

**Single file:**

```bash
python main.py --file "path/to/log.xes" --drift duration
```

**Multiple files:**

```bash
python main.py --files "log1.xes" "log2.xes" --drift duration routing --out output/results.csv
```

**Folder batch:**

```bash
python main.py --dir "path/to/folder" --drift duration routing arrival --out output/batch.csv --gt-mode ceravolo
```

## Experiments

- **Experiment #1 – Temporal Perspective:** Evaluates CVDrift on duration-based concept drift using synthetic logs with varying case counts, time intervals, and noise levels. Compares against the OCPA baseline.
- **Experiment #2 – Control Flow Perspective:** Evaluates CVDrift on control-flow drift using benchmark datasets from Bose, Ceravolo, and Ostovar with atomic, composite, and nested change patterns.
