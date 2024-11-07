import torch
from PIL import Image
import argparse
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
from zoedepth.utils.misc import colorize

def process_image(image_path, output_dir, model_name="zoedepth"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_pil(img)
        
    colored_depth = colorize(depth)
    colored = Image.fromarray(colored_depth)
    colored.save(f"{output_dir}/depth_colored.png")
    
    points_3d = depth_to_points(depth[None])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    pcd = o3d.geometry.PointCloud()
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualizacion de profundidad y nube de puntos 3D')
    parser.add_argument('--image','-i', 
                        required=True,
                        help='Imagen de entrada')
    parser.add_argument('--output','-o',
                        default='output',
                        help='Path salida')
    parser.add_argument('--model', '-m',
                        default='zoedepth',
                        help='MOdelo utilizado')

    args = parser.parse_args()
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    process_image(args.image, args.output, args.model)

if __name__ == "__main__":
    main()