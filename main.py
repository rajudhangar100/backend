from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import jiwer
import librosa
import joblib
import numpy as np


app = FastAPI()
model0 = joblib.load("dyslexia_model.pkl")

# 👇 Add this CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure this directory exists
# directory = 'C:\\Users\\Raju Dhangar\\Documents\\DyslexiaAi\\serverpy'
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "App is working"}


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    import whisper
    model = whisper.load_model("tiny")
    
    # Path to save uploaded file
    # audio_path = os.path.join(directory, f"temp_{file.filename}")
    audio_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")

    # Save the uploaded file
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    #load the audio into lebrosa(used for pauses)
    audio, sr = librosa.load(audio_path)    

    # Transcribe the audio
    result = model.transcribe(audio_path)

    # Delete the file after use (optional)
    os.remove(audio_path)

    #add error rate 
    actualtext='The fox ran fast through the fog.He saw five frogs near a log.One frog fell and flipped on a rock.“Funny frog!” said the fox, with a smile.The wind whooshed, and all frogs hopped away'
    transcribedtext=result["text"]
    error_rate = jiwer.wer(actualtext, transcribedtext)

    #add pauses
    pauses = librosa.effects.split(audio, top_db=20)
    long_pauses = [pause[1] - pause[0] for pause in pauses if (pause[1] - pause[0]) > 2]

    #add reading speed
    words = len(transcribedtext.split())
    duration_seconds = librosa.get_duration(y=audio, sr=sr)
    reading_speed=(words / duration_seconds) * 60

    input_data = np.array([[reading_speed, error_rate, len(long_pauses)]])
    prediction = model0.predict(input_data)[0]

    return {
        "text":result["text"],
        "prediction":prediction
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use PORT from environment
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    # uvicorn.run(app, host="0.0.0.0", port=8000)