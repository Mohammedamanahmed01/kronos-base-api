# Kronos-base API

FastAPI server hosting Kronos-base financial forecasting model.

## Endpoints

- `GET /health` - Check if model is alive
- `POST /predict` - Get price prediction for OHLCV data

## Example Usage

```python
import requests

response = requests.post(
    "https://your-username-kronos-base-api.hf.space/predict",
    json={"ohlcv": [[100, 102, 99, 101, 1000], [101, 103, 100, 102, 1100]]}
)

print(response.json())
# Output: {"up_prob": 0.72, "down_prob": 0.28, "confidence": 0.85}# kronos-base-api
