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

def rotate_points_180(points):
    rotation_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    return np.dot(points, rotation_matrix)

def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, model_name="zoedepth", max_depth=10.0, voxel_size=0.02):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Configurar modelo
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Procesar primera imagen y máscara
    img1 = Image.open(image_path1).convert("RGB")
    mask_img1 = Image.open(mask_path1).convert("RGBA")
    mask_img1 = mask_img1.resize(img1.size, Image.LANCZOS)
    mask_np1 = np.array(mask_img1)
    alpha_channel1 = mask_np1[:, :, 3]
    img_np1 = np.array(img1)

    # Procesar segunda imagen y máscara
    img2 = Image.open(image_path2).convert("RGB")
    mask_img2 = Image.open(mask_path2).convert("RGBA")
    mask_img2 = mask_img2.resize(img2.size, Image.LANCZOS)
    mask_np2 = np.array(mask_img2)
    alpha_channel2 = mask_np2[:, :, 3]
    img_np2 = np.array(img2)

    with torch.no_grad():
        # Obtener profundidad para ambas imágenes
        depth1 = model.infer_pil(img1)
        depth2 = model.infer_pil(img2)

    # Guardar mapas de profundidad coloreados
    colored_depth1 = colorize(depth1)
    colored_depth2 = colorize(depth2)
    Image.fromarray(colored_depth1).save(f"{output_dir}/depth_colored1.png")
    Image.fromarray(colored_depth2).save(f"{output_dir}/depth_colored2.png")

    # Convertir profundidad a puntos 3D para ambas imágenes
    points1_3d = depth_to_points(depth1[None])
    points2_3d = depth_to_points(depth2[None])

    # Preparar puntos y colores
    points1 = points1_3d.reshape(-1, 3)
    colors1 = img_np1.reshape(-1, 3) / 255.0
    
    points2 = points2_3d.reshape(-1, 3)
    colors2 = img_np2.reshape(-1, 3) / 255.0

    # Filtrar puntos usando máscaras y profundidad
    filtered_points1, filtered_colors1 = filter_points_by_mask_and_depth(
        points1, colors1, alpha_channel1, max_depth)
    filtered_points2, filtered_colors2 = filter_points_by_mask_and_depth(
        points2, colors2, alpha_channel2, max_depth)

    # Rotar los puntos de la segunda imagen 180 grados
    rotation_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    filtered_points2 = np.dot(filtered_points2, rotation_matrix)

    # Crear nubes de puntos
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(filtered_points1.astype(np.float32))
    pcd1.colors = o3d.utility.Vector3dVector(filtered_colors1.astype(np.float32))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(filtered_points2.astype(np.float32))
    pcd2.colors = o3d.utility.Vector3dVector(filtered_colors2.astype(np.float32))

    # Aplicar downsampling si es necesario
    if voxel_size > 0:
        pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
        pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    # Visualización
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    # Estadísticas
    print(f"\nEstadísticas de las nubes de puntos:")
    print(f"Imagen 1 - Puntos originales: {len(points1)}")
    print(f"Imagen 1 - Puntos filtrados: {len(filtered_points1)}")
    print(f"Imagen 2 - Puntos originales: {len(points2)}")
    print(f"Imagen 2 - Puntos filtrados: {len(filtered_points2)}")
    print(f"Profundidad máxima: {max_depth:.2f}m")

    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualización de profundidad y nube de puntos 3D con máscaras para dos imágenes')
    parser.add_argument('--image1', '-i1', required=True, help='Primera imagen de entrada')
    parser.add_argument('--mask1', '-m1', required=True, help='Máscara para primera imagen')
    parser.add_argument('--image2', '-i2', required=True, help='Segunda imagen de entrada')
    parser.add_argument('--mask2', '-m2', required=True, help='Máscara para segunda imagen')
    parser.add_argument('--output', '-o', default='output', help='Path salida')
    parser.add_argument('--model', '-md', default='zoedepth', help='Modelo utilizado')
    parser.add_argument('--max-depth', type=float, default=10.0, help='Profundidad máxima en metros (default: 10.0)')
    parser.add_argument('--voxel-size', type=float, default=0.001, help='Tamaño de voxel para downsampling (default: 0.001)')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    process_two_images(args.image1, args.mask1, args.image2, args.mask2, 
                      args.output, args.model, args.max_depth, args.voxel_size)

if __name__ == "__main__":
    main()