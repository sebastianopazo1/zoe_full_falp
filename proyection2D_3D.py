import torch
from PIL import Image
import argparse
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d

def process_image(image_path, model_name="zoedepth"):
    # Setup device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    #procesar la imagen
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_pil(img)
    points_3d = depth_to_points(depth[None])
    pcd = o3d.geometry.PointCloud()
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0

    pcd.points = o3d.utility.Vector3dVector(points)
    #pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Visualize
    print("\nControles de visualización:")
    print("- Rotar: Click izquierdo + arrastrar")
    print("- Zoom: Scroll del ratón")
    print("- Pan: Click derecho + arrastrar")
    print("- Salir: Q")
    
    o3d.visualization.draw_geometries([pcd, coord_frame])

def main():
    parser = argparse.ArgumentParser(description='Depth to 3D Point Cloud Visualization')
    parser.add_argument('--image', '-i', 
                        required=True,
                        help='Path to input image')
    parser.add_argument('--model', '-m',
                        default='zoedepth',
                        help='Model name (default: zoedepth)')

    args = parser.parse_args()
    process_image(args.image, args.model)

if __name__ == "__main__":
    main()