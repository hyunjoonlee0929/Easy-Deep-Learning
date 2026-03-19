# Dependency Compatibility Matrix

This project mixes classic ML, Torch ecosystem, Transformers/PEFT, and Streamlit UI.  
To reduce environment drift, use the matrix below and install with constraints.

## Recommended Base Matrix (Python 3.12)

| Component | Recommended |
|---|---|
| Python | 3.12.x |
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| torchaudio | 2.3.1 |
| transformers | 4.43.3 |
| peft | 0.12.0 |
| streamlit | 1.32.0 |
| scikit-learn | 1.5.1 |
| xgboost | 2.1.1 |

## Installation

```bash
pip install -r Easy_Deep_Learning/requirements.txt -c Easy_Deep_Learning/constraints/py312.txt
```

## Notes

- `torch/torchvision/torchaudio` must be installed as a matching set.
- `streamlit==1.32.0` does not support `st.audio_input`; app handles this via compatibility fallback.
- If you update Streamlit or Transformers stack, run test suite before use.
