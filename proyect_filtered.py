import torch
from PIL import Image
import argparse
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
from zoedepth.utils.misc import colorize
import os

def filter_points_by_mask_and_depth(points, colors, alpha_channel, max_depth, alpha_threshold=128):
    
    # Calculate depths (z-coordinate)
    depths = points[:, 2]
    mask_flat = alpha_channel.reshape(-1)

    depth_mask = depths <= max_depth
    alpha_mask = mask_flat > alpha_threshold  

    final_mask = np.logical_and(depth_mask, alpha_mask)

    filtered_points = points[final_mask]
    filtered_colors = colors[final_mask]
    """
    print(f"\nFiltrado por mascara:")
    print(f"  - Puntos totales: {len(points)}")
    print(f"  - Puntos con alpha > {alpha_threshold}: {np.count_nonzero(alpha_mask)}")
    print(f"  - Puntos con profundidad <= {max_depth}: {np.count_nonzero(depth_mask)}")
    print(f"  - Puntos finales luego de ambos filtros: {len(filtered_points)}")
    """

    return filtered_points, filtered_colors

def process_image(image_path, mask_path, output_dir, model_name="zoedepth", max_depth=10.0, voxel_size=0.02):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Load original image
    img = Image.open(image_path).convert("RGB")

    mask_img = Image.open(mask_path).convert("RGBA")
    mask_img = mask_img.resize(img.size, Image.LANCZOS)
    mask_np = np.array(mask_img)
    alpha_channel = mask_np[:, :, 3] 

    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_pil(img)
    minim = depth.min()
    maxim = depth.max()

    med_value = (maxim + minim) / 2 + 0.05

    colored_depth = colorize(depth)
    colored = Image.fromarray(colored_depth)
    colored.save(f"{output_dir}/depth_colored.png")

    points_3d = depth_to_points(depth[None])
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0

    filtered_points, filtered_colors = filter_points_by_mask_and_depth(points, colors, alpha_channel, med_value, alpha_threshold=128)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float32))
    
    # Downsampling de la nube de puntos
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Imprimir estadisticas
    print(f"\nEstadísticas finales de la nube de puntos:")
    print(f"Original points: {len(points)}")
    print(f"Filtered points after mask & depth: {len(filtered_points)}")
    print(f"Filtered points after downsampling: {len(np.asarray(pcd.points))}")
    print(f"Maximum depth: {med_value:.2f}m")

    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualización de profundidad y nube de puntos 3D con máscara')
    parser.add_argument('--image','-i', 
                        required=True,
                        help='Imagen de entrada')
    parser.add_argument('--mask','-m',
                        required=True,
                        help='Máscara alfa en formato PNG con canal alfa')
    parser.add_argument('--output','-o',
                        default='output',
                        help='Path salida')
    parser.add_argument('--model', '-md',
                        default='zoedepth',
                        help='Modelo utilizado')
    parser.add_argument('--max-depth',
                        type=float,
                        default=10.0,
                        help='Profundidad máxima en metros (default: 10.0)')
    parser.add_argument('--voxel-size',
                        type=float,
                        default=0.001,
                        help='Tamaño de voxel para downsampling (default: 0.02)')

    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    process_image(args.image, args.mask, args.output, args.model, args.max_depth, args.voxel_size)

if __name__ == "__main__":
    main()
