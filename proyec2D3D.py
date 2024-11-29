import torch
from PIL import Image
import argparse
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
from zoedepth.utils.misc import colorize

def filter_points_by_depth(points, colors, max_depth):
    """
    Filter points based on maximum depth threshold
    Args:
        points: Nx3 array of points
        colors: Nx3 array of colors
        max_depth: maximum depth threshold
    Returns:
        filtered points and colors
    """
    # Calculate depths (z-coordinate)
    depths = points[:, 2]
    
    # Create mask for points within threshold
    mask = depths <= max_depth
    
    # Apply mask to points and colors
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    
    return filtered_points, filtered_colors

def process_image(image_path, output_dir, model_name="zoedepth", max_depth=10.0):
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
    
    # Convert depth to 3D points
    points_3d = depth_to_points(depth[None])
    
    # Prepare points and colors
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0
    
    # Filter points based on maximum depth
    filtered_points, filtered_colors = filter_points_by_depth(points, colors, max_depth)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Create point cloud with filtered points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float32))
    
    # Set visualization options
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Print statistics
    print(f"\nPoint cloud statistics:")
    print(f"Original points: {len(points)}")
    print(f"Filtered points: {len(filtered_points)}")
    print(f"Points removed: {len(points) - len(filtered_points)}")
    print(f"Maximum depth: {max_depth:.2f}m")
    
    # Add geometries and run visualizer
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
                        help='Modelo utilizado')
    parser.add_argument('--max-depth',
                        type=float,
                        default=10.0,
                        help='Profundidad máxima en metros (default: 10.0)')

    args = parser.parse_args()
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    process_image(args.image, args.output, args.model, args.max_depth)

if __name__ == "__main__":
    main()