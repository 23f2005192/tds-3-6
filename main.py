import base64
import os
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
import librosa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ------------------------------------------------------------
# 1. INIT APP
# ------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------
# 2. REQUEST SCHEMA
# ------------------------------------------------------------
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# ------------------------------------------------------------
# 3. DECODE BASE64 TO TEMPORARY AUDIO FILE
# ------------------------------------------------------------
def decode_audio(audio_base64: str) -> str:
    audio_bytes = base64.b64decode(audio_base64)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp.write(audio_bytes)
    temp.close()
    return temp.name

# ------------------------------------------------------------
# 4. EXTRACT AUDIO FEATURES AND COMPUTE ALL REQUIRED STATISTICS
#    Features: duration, rms, zero_crossing_rate, 13 MFCCs
#    Returns JSON with exactly the required keys.
# ------------------------------------------------------------
def compute_stats_from_audio(file_path: str) -> dict:
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Feature extraction
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y)[0]                     # shape (n_frames,)
    zcr = librosa.feature.zero_crossing_rate(y)[0]       # shape (n_frames,)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # shape (13, n_frames)
    
    # Build a DataFrame: each column is a feature, each row is a time frame
    df = pd.DataFrame({
        "duration": np.full(mfccs.shape[1], duration),   # constant across frames
        "rms": rms,
        "zcr": zcr,
        **{f"mfcc_{i+1}": mfccs[i] for i in range(mfccs.shape[0])}
    })
    
    # If no frames (shouldn't happen), add a dummy row
    if df.empty:
        df = pd.DataFrame([[0]*len(df.columns)], columns=df.columns)
    
    # Initialize result skeleton
    result = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": {},
        "std": {},
        "variance": {},
        "min": {},
        "max": {},
        "median": {},
        "mode": {},
        "range": {},
        "allowed_values": {},
        "value_range": {},
        "correlation": []
    }
    
    # Compute per-column statistics
    for col in df.columns:
        data = df[col].values
        # Mode: use scipy.stats.mode (returns array, pick first)
        mode_val = float(stats.mode(data)[0][0]) if len(data) > 0 else 0.0
        
        result["mean"][col] = float(np.mean(data))
        result["std"][col] = float(np.std(data, ddof=1) if len(data) > 1 else 0.0)
        result["variance"][col] = float(np.var(data, ddof=1) if len(data) > 1 else 0.0)
        result["min"][col] = float(np.min(data))
        result["max"][col] = float(np.max(data))
        result["median"][col] = float(np.median(data))
        result["mode"][col] = mode_val
        result["range"][col] = float(np.max(data) - np.min(data))
        # allowed_values: unique values sorted (convert to float for JSON)
        result["allowed_values"][col] = sorted([float(v) for v in set(data)])
        result["value_range"][col] = [float(np.min(data)), float(np.max(data))]
    
    # Correlation matrix (only if more than one column)
    if len(df.columns) > 1:
        # Compute correlation matrix and convert to list of lists
        corr_matrix = df.corr().values.tolist()
        result["correlation"] = corr_matrix
    else:
        result["correlation"] = []   # empty list as per spec
    
    return result

# ------------------------------------------------------------
# 5. API ENDPOINTS
# ------------------------------------------------------------
@app.post("/")
async def process_audio(request: AudioRequest):
    try:
        # Decode base64 to temp file
        temp_path = decode_audio(request.audio_base64)
        # Compute statistics from the audio file
        stats_response = compute_stats_from_audio(temp_path)
        # Clean up
        os.unlink(temp_path)
        return stats_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy", "features": "duration, rms, zcr, 13 MFCCs"}

# ------------------------------------------------------------
# 6. RUN SERVER (binds to $PORT for Render)
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
