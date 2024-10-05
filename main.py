from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, pipeline
import torch

app = FastAPI()
print("1")
# Define a request model
class SentenceRequest(BaseModel):
    sentence: str
print("1")
# Load model and tokenizer
config = AutoConfig.from_pretrained("./saved_model")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("./saved_model", config=config)
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

# Create a text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define the endpoint
@app.post("/predict/")
async def predict(request: SentenceRequest):
    print(f"Received request: {request}")
    # Use the classifier pipeline for prediction
    try:
        # Predict using the classifier pipeline
        result = classifier(request.sentence)
        label = result[0]['label']
        print(label)
        # Return response based on the predicted label
        if label == "LABEL_1":
            return {"prediction": "negative"}
        elif label == "LABEL_0":
            return {"prediction": "positive"}
        else:
            raise HTTPException(status_code=500, detail="Unexpected label from model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
