"""OpenAI-compatible HTTP server backed by a local Transformers VLM."""

import argparse
import os
import time
import uuid
from typing import Any, Dict, List

import torch
import transformers
from fastapi import FastAPI, HTTPException
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL_PATH = (
    "/data/home/x_long32/workspace/models/Qwen3-VL-4B-Instruct"
)

app = FastAPI(title="Local Transformers VLM")
model = None
processor = None
model_path = None


def convert_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI image_url blocks to the format used by Qwen processors."""
    converted = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": content})
            continue

        converted_content = []
        for item in content:
            item_type = item.get("type")
            if item_type == "text":
                converted_content.append({"type": "text", "text": item["text"]})
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                converted_content.append({"type": "image", "image": url})
            elif item_type == "image":
                converted_content.append(item)
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported content type: {item_type}")
        converted.append({"role": message["role"], "content": converted_content})
    return converted


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "loading",
        "model_path": model_path,
        "transformers_source": transformers.__file__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/v1/chat/completions")
def chat_completions(request: Dict[str, Any]):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if request.get("n", 1) != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")

    messages = convert_openai_messages(request.get("messages", []))
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    temperature = float(request.get("temperature", 0.0))
    generation_kwargs = {
        "max_new_tokens": int(request.get("max_tokens", 1000)),
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    started = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    prompt_tokens = int(inputs.input_ids.shape[-1])
    completion_tokens = len(generated_ids[0])
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.get("model", os.path.basename(model_path)),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "server_timing": {"generation_seconds": time.time() - started},
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Local Hugging Face model directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main():
    global model, processor, model_path

    args = parse_args()
    model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    print(f"Loading model from: {model_path}")
    print(f"Using Transformers source: {transformers.__file__}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="sdpa",
    )
    model.eval()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
