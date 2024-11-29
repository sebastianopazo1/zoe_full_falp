import torch
from PIL import Image
import numpy as np
import cv2
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize, save_raw_16bit
import os

def create_black_mask(image):
    """Create a mask for non-black areas of the image."""
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convert to grayscale if image is RGB
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Create binary mask (non-black pixels)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return mask

def process_image(image_path, output_dir, model_name="zoedepth"):
    """Process image and generate depth map, ignoring black areas."""
    # Setup device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Load image and create mask
    img = Image.open(image_path).convert("RGB")
    mask = create_black_mask(img)
    
    # Get depth prediction
    with torch.no_grad():
        depth_output = model.infer_pil(img)
        
    # Convert depth to numpy if it's a tensor
    if isinstance(depth_output, torch.Tensor):
        depth_output = depth_output.cpu().numpy()
    
    # Normalize mask to 0-1 range
    mask = mask.astype(float) / 255.0
    
    # Apply mask to depth map
    masked_depth = depth_output * mask
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save raw masked depth
    raw_depth_path = os.path.join(output_dir, f"{base_name}_depth_raw.png")
    save_raw_16bit(masked_depth, raw_depth_path)
    
    # Save colored masked depth with white background for black areas
    colored_depth = colorize(masked_depth, invalid_val=0, background_color=(255, 255, 255, 255))
    colored_path = os.path.join(output_dir, f"{base_name}_depth_colored.png")
    Image.fromarray(colored_depth).save(colored_path)
    
    # Print depth statistics for non-black areas
    valid_depth = masked_depth[mask > 0]
    if len(valid_depth) > 0:
        print("\nDepth Statistics (non-black areas):")
        print(f"Minimum depth: {np.min(valid_depth):.3f} meters")
        print(f"Maximum depth: {np.max(valid_depth):.3f} meters")
        print(f"Average depth: {np.mean(valid_depth):.3f} meters")
    
    print(f"\nResults saved in: {output_dir}")
    return raw_depth_path, colored_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Depth Estimation ignoring black areas')
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