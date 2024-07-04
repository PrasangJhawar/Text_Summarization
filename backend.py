from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    # Add other origins as needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load tokenizer and model from local directory
model_path = r'C:\Users\DEll\Downloads\Text Summarization\fine_tuned_t5_small_model'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Pydantic model for request body
class SummarizeRequest(BaseModel):
    article: str

# Endpoint for summarization
@app.post('/summarize')
def summarize(data: SummarizeRequest):
    article = data.article

    inputs = tokenizer.encode("summarize: " + article, return_tensors='pt', max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {'summary': summary}

if __name__ == '__main__':
    uvicorn.run(app, host="192.168.1.6", port=8000)
