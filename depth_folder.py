import torch
from PIL import Image
import argparse
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize, save_raw_16bit
import os
from pathlib import Path

def process_folder(input_dir, output_dir, model_name="zoedepth"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    os.makedirs(output_dir, exist_ok=True)
    #Load model
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    for img_path in Path(input_dir).iterdir():
        if img_path.suffix.lower() not in valid_extensions:
            continue
        print(f"\nProcessing: {img_path.name}")
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            with torch.no_grad():
                depth = model.infer_pil(img)
            colored = Image.fromarray(colorize(depth))
            output_filename = f"{img_path.stem}_depth_colored.png"
            colored.save(os.path.join(output_dir, output_filename))
            #save_raw_16bit(depth, os.path.join(output_dir, f"{img_path.stem}_depth_raw.png"))
            
            print(f"Dimensiones:{depth.shape}")
            print(f"Min: {np.min(depth):.3f} m")
            print(f"Max: {np.max(depth):.3f} m")
            print(f"Promedio: {np.mean(depth):.3f} m")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True)
    parser.add_argument('--output_dir', '-o', default='output')
    parser.add_argument('--model', '-m', default='zoedepth')
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.model)

if __name__ == "__main__":
    main()