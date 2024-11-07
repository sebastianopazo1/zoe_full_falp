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

    # Procesar imagen
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_pil(img)
        
    points_3d = depth_to_points(depth[None])
    
    # Configurar visualizador con soporte GPU
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Crear nube de puntos
    pcd = o3d.geometry.PointCloud()
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0

    # Optimizar datos para GPU 
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    
    # Configurar opciones de renderizado
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    opt.use_gpu = True # Activar renderizado por GPU
    
    # Crear marco de coordenadas
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Agregar geometrías
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    # Ejecutar visualizador
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualizacion de puntos 3D en base a la profundidad')
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