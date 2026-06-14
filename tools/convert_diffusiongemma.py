#!/usr/bin/env python3
"""
Custom GGUF conversion for DiffusionGemma (based on Gemma 4 text backbone).

DiffusionGemma architecture:
- Top-level: DiffusionGemmaForBlockDiffusion (not yet in llama.cpp)
- Text backbone: diffusion_gemma_text -> essentially Gemma 4 26B A4B MoE
- Vision encoder: Gemma 4 vision (27 layers, 1152 hidden, 16-head)
- Canvas length: 256 tokens
- Block autoregressive diffusion with bidirectional attention

Strategy: 
1. Convert text backbone using Gemma 4 conversion logic
2. Handle diffusion-specific parameters (canvas_length, eoi_token_id, etc.)
3. Handle vision encoder if needed (for multimodal)
"""

import sys
import argparse
from pathlib import Path

# Add llama.cpp conversion modules to path
sys.path.insert(0, "/home/wubu/llama.cpp")

import torch
from conversion.gemma import Gemma4Model
from conversion.base import ModelBase, gguf, logger

# Register DiffusionGemma text backbone using Gemma 4 logic
@ModelBase.register("DiffusionGemmaForBlockDiffusion")
class DiffusionGemmaModel(Gemma4Model):
    """DiffusionGemma text backbone conversion - uses Gemma 4 logic with diffusion params"""
    
    # The actual HF architecture name
    model_arch = gguf.MODEL_ARCH.GEMMA4
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DiffusionGemma-specific hparams from top-level config
        self.canvas_length = self.hparams.get("canvas_length", 256)
        self.boi_token_id = self.hparams.get("boi_token_id", 255999)
        self.eoi_token_id = self.hparams.get("eoi_token_id", 258882)
        self.image_token_id = self.hparams.get("image_token_id", 258880)
    
    def set_gguf_parameters(self):
        """Set GGUF parameters including diffusion-specific ones"""
        super().set_gguf_parameters()
        
        # Add diffusion-specific metadata
        # Note: These are custom extensions, may need GGUF spec updates
        self.gguf_writer.add_general_file_type(self.ftype)
        
        # Store diffusion params in general metadata
        meta = self.gguf_writer
        
    def set_vocab(self):
        """Use Gemma 4 vocab handling"""
        super().set_vocab()
    
    def filter_tensors(self, item):
        """Filter tensors - skip vision-related for text-only conversion"""
        name, gen = item
        
        # Skip vision tower tensors for text-only model
        if name.startswith(("vision_tower.", "vision_model.", "multi_modal_projector.", "embed_vision.", "embed_audio.")):
            return None
        
        # Skip audio tower if present
        if "audio_tower" in name or "embed_audio" in name:
            return None
        
        return super().filter_tensors(item)
    
    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None):
        """Modify tensors - handle diffusion-specific tensor names"""
        # Handle any tensor name remapping if needed
        yield from super().modify_tensors(data_torch, name, bid)


def convert_diffusiongemma(model_dir, outfile, ftype="Q4_K_M"):
    """Convert DiffusionGemma safetensors to GGUF"""
    model_dir = Path(model_dir)
    
    print(f"Loading model from {model_dir}")
    print(f"Output: {outfile}")
    print(f"Quantization: {ftype}")
    
    # Run conversion
    model = DiffusionGemmaModel(
        dir_model=model_dir,
        ftype=ftype,
        fname_out=Path(outfile),
        use_temp_file=False,
    )
    
    # Run conversion
    model.convert(outfile)
    print(f"Conversion complete: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DiffusionGemma to GGUF")
    parser.add_argument("--model-dir", required=True, help="Path to DiffusionGemma HF model directory")
    parser.add_argument("--outfile", required=True, help="Output GGUF file path")
    parser.add_argument("--ftype", default="Q4_K_M", help="Quantization type (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)")
    
    args = parser.parse_args()
    
    convert_diffusiongemma(args.model_dir, args.outfile, args.ftype)