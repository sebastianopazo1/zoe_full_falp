import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import cv2

def resize_image(image, max_size=(1024, 1024)):
    """
    Redimensiona la imagen manteniendo la proporción hasta que su lado más largo sea max_size.
    """
    image.thumbnail(max_size, Image.ANTIALIAS)
    return image

def process_two_cameras(image_path1, image_path2, model_name="zoedepth"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for model inference: {DEVICE}")

    # Inicializar modelo
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    T_cam1_world = np.eye(4)
    T_cam2_world = np.array([[1, 0, 0, 0],
                             [0, 1, 0, -0.5],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

    with torch.no_grad():
        # Cargar y redimensionar 
        img1 = Image.open(image_path1).convert("RGB")
        img1 = resize_image(img1, max_size=(1024, 1024))  
        depth1 = model.infer_pil(img1)
        img_np1 = np.array(img1)

        img2 = Image.open(image_path2).convert("RGB")
        img2 = resize_image(img2, max_size=(1024, 1024))  
        depth2 = model.infer_pil(img2)
        img_np2 = np.array(img2)

    points1 = depth_to_points(depth1[None])  # [B, H, W, 3]
    points2 = depth_to_points(depth2[None])  # [B, H, W, 3]

    points1_flat = points1.reshape(-1, 3)
    points2_flat = points2.reshape(-1, 3)

    # Transformar puntos al sistema de coordenadas mundial
    points1_homogeneous = np.concatenate([points1_flat, np.ones((points1_flat.shape[0], 1))], axis=1)
    points1_world = (T_cam1_world @ points1_homogeneous.T).T[:, :3]
    points2_homogeneous = np.concatenate([points2_flat, np.ones((points2_flat.shape[0], 1))], axis=1)
    points2_world = (T_cam2_world @ points2_homogeneous.T).T[:, :3]

    # Crear nubes de puntos en CPU
    # Nube de puntos 1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1_world.astype(np.float32))
    colors1 = (img_np1.reshape(-1, 3) / 255.0).astype(np.float32)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)

    # Nube de puntos 2
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2_world.astype(np.float32))
    colors2 = (img_np2.reshape(-1, 3) / 255.0).astype(np.float32)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)


    coord_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame1)

    coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    coord_frame2.transform(T_cam2_world)
    vis.add_geometry(coord_frame2)

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
