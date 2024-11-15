
import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import cv2

def process_two_cameras(image_path1, image_path2, model_name="zoedepth"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Inicializar modelo
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Parámetros intrínsecos de la cámara (ajustar según tu cámara)
    fx = 525.0  # distancia focal x
    fy = 525.0  # distancia focal y
    cx = 319.5  # centro óptico x
    cy = 239.5  # centro óptico y
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Matriz de transformación para la segunda cámara (38cm abajo)
    T_2to1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0.38],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

    # Procesar imágenes
    with torch.no_grad():
        # Cámara 1 (origen)
        img1 = Image.open(image_path1).convert("RGB")
        depth1 = model.infer_pil(img1)
        img_np1 = np.array(img1)
        
        # Cámara 2 (38cm abajo)
        img2 = Image.open(image_path2).convert("RGB")
        depth2 = model.infer_pil(img2)
        img_np2 = np.array(img2)

    # Crear nubes de puntos
    def depth_to_points_with_K(depth_map, K, color_img):
        h, w = depth_map.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Coordenadas homogéneas
        x_homo = (x - K[0,2]) / K[0,0]
        y_homo = (y - K[1,2]) / K[1,1]
        
        points = np.stack([
            x_homo * depth_map,
            y_homo * depth_map,
            depth_map,
            np.ones_like(depth_map)
        ], axis=-1)
        
        points = points.reshape(-1, 4)
        colors = color_img.reshape(-1, 3) / 255.0
        
        return points, colors

    # Generar nubes de puntos
    points1, colors1 = depth_to_points_with_K(depth1, K, img_np1)
    points2, colors2 = depth_to_points_with_K(depth2, K, img_np2)
    
    # Transformar puntos de cámara 2 al sistema de coordenadas de cámara 1
    #points2 = (T_2to1 @ points2.T).T

    # Visualización
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    # Nube de puntos 1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1[:, :3].astype(np.float32))
    pcd1.colors = o3d.utility.Vector3dVector(colors1.astype(np.float32))
    vis.add_geometry(pcd1)

    # Nube de puntos 2
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2[:, :3].astype(np.float32))
    pcd2.colors = o3d.utility.Vector3dVector(colors2.astype(np.float32))
    vis.add_geometry(pcd2)

    # Sistema de coordenadas
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame)

    # Configuración de visualización
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Procesamiento de dos cámaras')
    parser.add_argument('--image1', required=True, help='Imagen de cámara 1 (origen)')
    parser.add_argument('--image2', required=True, help='Imagen de cámara 2 (38cm abajo)')
    parser.add_argument('--model', default='zoedepth', help='Modelo a utilizar')
    
    args = parser.parse_args()
    process_two_cameras(args.image1, args.image2, args.model)