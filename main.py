from fastapi import FastAPI, UploadFile
from llm_endpoint import generate_response
from vision_endpoint import detect_defect

app = FastAPI()

@app.post("/ask")
def ask(question: str):
    return generate_response(question)

@app.post("/detect")
def detect(file: UploadFile):
    return detect_defect(file)
