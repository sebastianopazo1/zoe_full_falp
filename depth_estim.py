import torch
from PIL import Image
import argparse
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize, save_raw_16bit
import os

def process_image(image_path, output_dir, model_name="zoedepth"):
    # Setup device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Load and process image
    img = Image.open(image_path).convert("RGB")
    
    # Get depth prediction
    with torch.no_grad():
        depth = model.infer_pil(img)
        
    # Save outputs
    colored = Image.fromarray(colorize(depth))
    colored.save(os.path.join(output_dir, "depth_colored.png"))
    save_raw_16bit(depth, os.path.join(output_dir, "depth_raw.png"))
    
    # Imprimir algunos valores de profundidad
    print("\nValores de profundidad:")
    print(f"Dimensiones del mapa de profundidad: {depth.shape}")
    print(f"Valor mínimo de profundidad: {np.min(depth):.3f} metros")
    print(f"Valor máximo de profundidad: {np.max(depth):.3f} metros")
    print(f"Profundidad promedio: {np.mean(depth):.3f} metros")
    
    # Imprimir valores en puntos específicos
    h, w = depth.shape
    print("\nValores en puntos específicos:")
    print(f"Centro de la imagen: {depth[h//2, w//2]:.3f} metros")
    print(f"Esquina superior izquierda: {depth[0, 0]:.3f} metros")
    print(f"Esquina superior derecha: {depth[0, -1]:.3f} metros")
    print(f"Esquina inferior izquierda: {depth[-1, 0]:.3f} metros")
    print(f"Esquina inferior derecha: {depth[-1, -1]:.3f} metros")
    
    print(f"\nResultados guardados en: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Depth Estimation for Single Image')
    parser.add_argument('--image', '-i', 
                        required=True,
                        help='Path to input image')
    parser.add_argument('--output', '-o',
                        default='output',
                        help='Output directory path (default: output)')
    parser.add_argument('--model', '-m',
                        default='zoedepth',
                        help='Model name (default: zoedepth)')

    args = parser.parse_args()
    process_image(args.image, args.output, args.model)

if __name__ == "__main__":
    main()