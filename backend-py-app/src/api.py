import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datasets import load_dataset

from abliterators.abliterator import ModelAbliterator
from abliterators.reverseAbliterator import ReverseAbliterator

app = FastAPI()

# Pydantic models for request bodies
class AbliteratorConfig(BaseModel):
    model_path: str
    dataset: List[List[str]]
    device: str = "cuda"
    n_devices: Optional[int] = None
    activation_layers: List[str] = ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']

class EnhanceConfig(BaseModel):
    layers: Optional[List[int]] = None
    W_O: bool = True
    mlp: bool = True
    strength: float = 1.0

# Global variables to store instances
abliterator = None
reverse_abliterator = None

@app.post("/initialize_abliterator")
async def initialize_abliterator(config: AbliteratorConfig):
    global abliterator
    try:
        abliterator = ModelAbliterator(
            model=config.model_path,
            dataset=config.dataset,
            device=config.device,
            n_devices=config.n_devices,
            activation_layers=config.activation_layers
        )
        return {"message": "Abliterator initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize_reverse_abliterator")
async def initialize_reverse_abliterator(config: AbliteratorConfig):
    global reverse_abliterator
    try:
        reverse_abliterator = ReverseAbliterator(
            model=config.model_path,
            dataset=config.dataset,
            device=config.device,
            n_devices=config.n_devices,
            activation_layers=config.activation_layers
        )
        return {"message": "ReverseAbliterator initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/abliterate")
async def abliterate(config: EnhanceConfig):
    if abliterator is None:
        raise HTTPException(status_code=400, detail="Abliterator not initialized")
    try:
        abliterator.apply_refusal_dirs(
            abliterator.refusal_dirs().values(),
            W_O=config.W_O,
            mlp=config.mlp,
            layers=config.layers
        )
        return {"message": "Abliteration completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhance")
async def enhance(config: EnhanceConfig):
    if reverse_abliterator is None:
        raise HTTPException(status_code=400, detail="ReverseAbliterator not initialized")
    try:
        reverse_abliterator.enhance_model(
            layers=config.layers,
            W_O=config.W_O,
            mlp=config.mlp,
            strength=config.strength
        )
        return {"message": "Enhancement completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_abliterator")
async def test_abliterator(N: int = 16, batch_size: int = 4):
    if abliterator is None:
        raise HTTPException(status_code=400, detail="Abliterator not initialized")
    try:
        results = abliterator.test(N=N, batch_size=batch_size)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_reverse_abliterator")
async def test_reverse_abliterator(N: int = 16, batch_size: int = 4, max_tokens_generated: int = 64):
    if reverse_abliterator is None:
        raise HTTPException(status_code=400, detail="ReverseAbliterator not initialized")
    try:
        results = reverse_abliterator.test_enhancement(N=N, batch_size=batch_size, max_tokens_generated=max_tokens_generated)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_huggingface_dataset")
async def load_huggingface_dataset(request: dict):
    try:
        repo_id = request.get('repo_id')
        if not repo_id:
            raise ValueError("Repository ID is required")

        # Load the dataset from HuggingFace
        dataset = load_dataset(repo_id)
        
        # Handle different dataset structures
        if 'train' in dataset:
            data = dataset['train']
        elif 'test' in dataset:
            data = dataset['test']
        else:
            data = dataset[list(dataset.keys())[0]]

        # Extract target and baseline instructions
        # Modify these based on your dataset's column names
        target_instructions = []
        baseline_instructions = []

        if 'goal' in data.features:
            # For toxic steering dataset
            target_instructions = [example['goal'] for example in data]
            baseline_instructions = target_instructions[:len(target_instructions)//2]
            target_instructions = target_instructions[len(target_instructions)//2:]
        elif 'instruction' in data.features:
            # For instruction datasets
            instructions = [example['instruction'] for example in data]
            mid = len(instructions) // 2
            target_instructions = instructions[:mid]
            baseline_instructions = instructions[mid:]
        else:
            raise ValueError("Dataset format not supported")

        # Initialize or update the abliterator with new data
        if reverse_abliterator is None:
            config = AbliteratorConfig(
                model_path="your_model_path",  # Update with actual model path
                dataset=[target_instructions, baseline_instructions],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            await initialize_reverse_abliterator(config)
        else:
            # Update existing abliterator's dataset
            reverse_abliterator.target_inst_train, reverse_abliterator.target_inst_test = \
                reverse_abliterator.prepare_dataset(target_instructions)
            reverse_abliterator.baseline_inst_train, reverse_abliterator.baseline_inst_test = \
                reverse_abliterator.prepare_dataset(baseline_instructions)
            
            # Recache activations with new data
            reverse_abliterator.cache_activations(N=len(target_instructions), batch_size=8)

        return {
            "message": f"Successfully loaded dataset from {repo_id}",
            "num_examples": len(target_instructions) + len(baseline_instructions),
            "num_target": len(target_instructions),
            "num_baseline": len(baseline_instructions)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

print("FastAPI application for Abliterator and ReverseAbliterator is ready.")