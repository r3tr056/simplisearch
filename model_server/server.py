#!/usr/bin/python

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_206_PARTIAL_CONTENT
import os

app = FastAPI()

MODEL_DIR = "models"

def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)

def file_iterator(file_path: str, chunk_size: int = 1024 * 1024):
    with open(file_path, mode="rb") as file:
        while chunk := file.read(chunk_size):
            yield chunk

@app.get("/")
def read_root():
    return {"message": "Welcome to the SimpliSearch model server!"}

@app.get("/models/{model_name}")
def get_model(model_name: str, request: Request):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.onnx")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    # get file size
    file_size = get_file_size(model_path)

    # handle the range header (for partial content requests)
    range_header = request.headers.get('Range', None)
    if range_header:
        byte_range = range_header.replace("bytes=", "")
        start_byte = int(byte_range[0])
        end_byte = int(byte_range[1]) if len(byte_range) > 1 else file_size - 1

        # ensure the range is within file bounds
        start_byte = max(0, start_byte)
        end_byte = min(file_size - 1, end_byte)

        # calculate content length
        content_length = end_byte - start_byte + 1
        headers = {
            "Content-Range": f"bytes {start_byte}-{end_byte}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length)
        }

        return StreamingResponse(
            file_iterator(model_path)[start_byte:end_byte],
            headers=headers,
            status_code=HTTP_206_PARTIAL_CONTENT
        )
    
    headers = {
        "Content-Length": str(file_size)
    }
    return StreamingResponse(file_iterator(model_path), headers=headers)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)