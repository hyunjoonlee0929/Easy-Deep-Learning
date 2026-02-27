# OmniInsight

OmniInsight is a modular AI analysis platform for tabular and multi-omics workflows.
It runs end-to-end:

1. Data validation
2. Auto preprocessing (imputation, scaling, one-hot encoding, split)
3. Model training (XGBoost or PyTorch DNN with early stopping)
4. SHAP feature attribution
5. Agent-driven structured report generation

## Project Structure

```text
OmniInsight/
├── core/
├── adapters/
├── interpretation/
├── agents/
├── dashboard/
├── config/
├── data/
└── main.py
```

## Installation

```bash
pip install -r OmniInsight/requirements.txt
```

## CLI Demo (example_dataset.csv)

```bash
python OmniInsight/main.py \
  --data OmniInsight/data/example_dataset.csv \
  --config OmniInsight/config/model_config.yaml
```

Run with DNN:

```bash
python OmniInsight/main.py \
  --data OmniInsight/data/example_dataset.csv \
  --model-type dnn
```

## Streamlit Dashboard

```bash
streamlit run OmniInsight/dashboard/app.py
```

In the dashboard you can upload any CSV, choose target/task/model, and execute the full pipeline.

## OpenAI Agent Behavior

- If `OPENAI_API_KEY` is set, agents call OpenAI and return JSON outputs.
- If no API key is set, agents return deterministic structured mock outputs.
