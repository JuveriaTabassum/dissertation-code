# Shadow AI and Agentic AI Misuse Detection and Governance using a Neurosymbolic Approach

This repository contains the source code and supporting data for my MSc Cybersecurity dissertation (Middlesex University, CST4990), *Shadow AI and Agentic AI Misuse Detection and Governance using a Neuro-Symbolic Approach*.

The project compares three approaches to detecting AI misuse:

- a regularised nine-class MLP neural classifier using MiniLM embeddings and operational features
- a symbolic engine using explicit event-level and session-level rules
- a neurosymbolic fusion system combining neural predictions with symbolic evidence

Shadow AI is handled as a separate governance issue by recording whether activity takes place on a sanctioned or unsanctioned AI platform. The project also includes a Streamlit dashboard for reviewing governance outcomes and individual detection decisions.

## Project structure

```text
src/
  core.py                     Shared event schema, taxonomy and constants
  dataset_scenarios.yaml      Scenario templates used by the simulator
  dataset_simulator.py        Generates the simulated dataset
  neural_classifier.py       Trains and evaluates the neural classifier
  symbolic_engine.py          Applies and evaluates the symbolic rules
  neurosymbolic_fusion.py     Combines neural and symbolic decisions
  dashboard.py                Displays saved governance decisions
  plot_category_recall.py     Creates the category recall comparison figure for the report

experiments/
  classifier_experiments.py   Compares candidate encoders, architectures and feature sets
  prepare_external_dataset.py Prepares the external validation dataset
  external_validation_exp1.py Combined simulated and external-data experiment
  external_validation_exp2.py External-only experiment
  external_validation_events.jsonl  Saved external validation dataset

outputs/
  simulated_events.csv              Saved simulated dataset in CSV format
  simulated_events.jsonl            Saved simulated dataset used by the system
  neural_classifier_model.joblib    Saved trained neural classifier
  fusion_dashboard_data.json        Saved fusion decisions used by the dashboard
```

## Setup

The project was developed using Python 3.13.

Install the required packages from the project folder:

```bash
pip install -r requirements.txt
```

The sentence-transformer model is downloaded automatically the first time the neural classifier is run, so an internet connection is required for the initial run.

## Main workflow

The files can be opened and run individually in PyCharm in this order:

1. `src/dataset_simulator.py`
2. `src/neural_classifier.py`
3. `src/symbolic_engine.py`
4. `src/neurosymbolic_fusion.py`
5. `src/dashboard.py`

The simulator creates the dataset in `outputs`. The classifier saves the trained model, and the fusion file creates the JSON data used by the dashboard. The included output files can also be used to inspect the completed system without regenerating every result.

## Supplementary experiments

`classifier_experiments.py` performs grouped Development-set comparisons of the text encoders, classifier architectures and feature sets to select the most suitable configuration for the final neural classifier.

The two external-validation files provide supplementary evaluation beyond the simulated dataset:

- `external_validation_exp1.py` trains and evaluates using combined simulated and external data
- `external_validation_exp2.py` trains and evaluates using only the external dataset

`prepare_external_dataset.py` takes the externally sourced datasets mapped to each misuse category and preprocesses them into a single `external_validation_events.jsonl` file for use in the supplementary experiments.

## External datasets

The raw external datasets are not stored in this repository because of their combined size. They are available here:

**Google Drive:** [Download the external datasets](https://drive.google.com/file/d/1NhL2eF4_nANb5jjeVh7AcjHI9FQNo2Dp/view?usp=sharing)

The included `external_validation_events.jsonl` file can be used directly to run `external_validation_exp1.py` and `external_validation_exp2.py`.

The raw datasets are only needed to rebuild this file. Download `Datasets.zip` from the Google Drive link and extract the `Datasets` folder inside the repository's `experiments` folder. Then run `prepare_external_dataset.py`, which searches this folder and its subfolders for the required source files.

The external validation dataset was prepared from LMSYS-Chat-1M, InjecAgent, AgentLeak, ASSEBench, CIRCLE, HaluEval, TensorTrust, SafeRAG and RedCode. `prepare_external_dataset.py` shows how each source is mapped to the project taxonomy.

## Notes

- The external-data experiments are supplementary and contain limited operational telemetry
- No real organisational or personal data was collected for this project
- The repository is provided for dissertation assessment and reproducibility
