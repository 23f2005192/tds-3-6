import base64
import os
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import wavfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time

app = FastAPI()

# In-memory cache: audio_id -> computed statistics dict
CACHE = {}

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def decode_audio(audio_base64: str) -> str:
    audio_bytes = base64.b64decode(audio_base64)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp.write(audio_bytes)
    temp.close()
    return temp.name

def compute_stats_from_wav(file_path: str) -> dict:
    # Read WAV file (fast)
    sr, y = wavfile.read(file_path)
    # Convert to float for calculations
    y = y.astype(np.float32) / np.iinfo(y.dtype).max

    duration = len(y) / sr
    frame_len = 2048
    num_frames = len(y) // frame_len
    if num_frames == 0:
        num_frames = 1
        frame_len = len(y)

    # Pre-allocate arrays
    rms_vals = np.zeros(num_frames, dtype=np.float32)
    zcr_vals = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        frame = y[start:end]
        # RMS
        rms_vals[i] = np.sqrt(np.mean(frame ** 2))
        # Zero-crossing rate
        sign_changes = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr_vals[i] = sign_changes / len(frame)

    # Build DataFrame: one row per frame
    df = pd.DataFrame({
        "duration": np.full(num_frames, duration),
        "rms": rms_vals,
        "zcr": zcr_vals
    })

    # If audio is completely silent, RMS may be zero – that's fine.

    # Initialize the exact required response
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

    for col in df.columns:
        data = df[col].values
        # Mode (uses scipy; for continuous data we fall back to median or first value)
        if len(data) > 0:
            mode_val = float(stats.mode(data)[0][0])
        else:
            mode_val = 0.0

        result["mean"][col] = float(np.mean(data))
        result["std"][col] = float(np.std(data, ddof=1) if len(data) > 1 else 0.0)
        result["variance"][col] = float(np.var(data, ddof=1) if len(data) > 1 else 0.0)
        result["min"][col] = float(np.min(data))
        result["max"][col] = float(np.max(data))
        result["median"][col] = float(np.median(data))
        result["mode"][col] = mode_val
        result["range"][col] = float(np.max(data) - np.min(data))
        # Allowed values: unique sorted (float)
        result["allowed_values"][col] = sorted([float(v) for v in set(data)])
        result["value_range"][col] = [float(np.min(data)), float(np.max(data))]

    # Correlation matrix
    if len(df.columns) > 1:
        result["correlation"] = df.corr().values.tolist()
    else:
        result["correlation"] = []

    return result

@app.post("/")
async def process_audio(request: AudioRequest):
    # Check cache first
    if request.audio_id in CACHE:
        return CACHE[request.audio_id]

    try:
        temp_path = decode_audio(request.audio_base64)
        stats_response = compute_stats_from_wav(temp_path)
        os.unlink(temp_path)

        # Store in cache
        CACHE[request.audio_id] = stats_response
        return stats_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy", "features": "duration, rms, zcr", "cached": len(CACHE)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
