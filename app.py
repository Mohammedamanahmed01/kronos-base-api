from fastapi import FastAPI
from kronos import KronosPredictor
import torch

app = FastAPI()

# Load model once at startup
predictor = None

@app.on_event("startup")
async def startup():
    global predictor
    print("🚀 Loading Kronos-base model...")
    try:
        predictor = KronosPredictor.from_pretrained(
            "NeoQuasar/Kronos-base",
            device="cpu"
        )
        print("✅ Kronos-base loaded and ready!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/health")
async def health():
    """Endpoint for keepalive pings"""
    return {
        "status": "alive",
        "model": "kronos-base",
        "device": "cpu"
    }

@app.post("/predict")
async def predict(data: dict):
    """
    Input: {"ohlcv": [[o,h,l,c,v], [o,h,l,c,v], ...]}
    Output: {"up_prob": 0.72, "down_prob": 0.28, "confidence": 0.85}
    """
    if predictor is None:
        return {"error": "Model not loaded"}
    
    try:
        ohlcv_data = data.get("ohlcv", [])
        
        if not ohlcv_data:
            return {"error": "No OHLCV data provided"}
        
        # Run prediction
        result = predictor.predict(ohlcv_data)
        
        return {
            "up_prob": float(result['up']),
            "down_prob": float(result['down']),
            "confidence": float(result['confidence']),
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "status": "error"
        }

@app.get("/")
async def root():
    return {
        "message": "Kronos-base API",
        "endpoints": {
            "/health": "GET - Check if model is alive",
            "/predict": "POST - Get price prediction"
        }
    }