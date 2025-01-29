from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, pipeline
import torch
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentenceRequest(BaseModel):
    sentence: str

config = AutoConfig.from_pretrained("./saved_model")

model = AutoModelForSequenceClassification.from_pretrained("./saved_model", config=config)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./saved_model")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
@app.get("/")
async def root():
    return {"message": "Running"}
@app.post("/predict/")
async def predict(request: SentenceRequest):
    print(f"Received request: {request}")
    if re.match(r"^\d+$", request.sentence.strip()):  
        return {"prediction": "Number"}
    
    try:
        result = classifier(request.sentence)
        label = result[0]['label']
        print(label)
        if label == "LABEL_1":
            return {"prediction": "negative"}
        elif label == "LABEL_0":
            return {"prediction": "positive"}
        else:
            raise HTTPException(status_code=500, detail="Unexpected label from model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
