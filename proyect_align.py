import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import os
import argparse
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_model
import cv2

def resize_image(image, max_size=(640, 480)):
    """
    Redimensiona la imagen manteniendo la proporción
    """
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def get_3d_keypoints(img_np, depth_map, pose_model):
    """
    Obtiene los keypoints 3D específicos usando mmpose y el mapa de profundidad,
    asegurando información en los tres ejes
    """
    mmpose_results = inference_topdown(pose_model, img_np)
    keypoints_2d = mmpose_results[0].pred_instances.keypoints[0]
    keypoints_scores = mmpose_results[0].pred_instances.keypoint_scores[0]
    
    # Definir los pares de keypoints que queremos usar
    keypoint_pairs = {
        'shoulders': (5, 6),    # left_shoulder, right_shoulder
        'hips': (11, 12),       # left_hip, right_hip
        'knees': (13, 14)       # left_knee, right_knee
    }
    
    keypoints_3d = []
    valid_pairs = {}
    
    # Procesar cada par de keypoints
    for pair_name, (idx1, idx2) in keypoint_pairs.items():
        # Verificar que ambos puntos del par tienen buena confianza (por ejemplo > 0.5)
        if keypoints_scores[idx1] > 0.5 and keypoints_scores[idx2] > 0.5:
            x1, y1 = map(int, keypoints_2d[idx1])
            x2, y2 = map(int, keypoints_2d[idx2])
            
            # Verificar que los puntos están dentro de los límites de la imagen
            if (0 <= x1 < depth_map.shape[1] and 0 <= y1 < depth_map.shape[0] and
                0 <= x2 < depth_map.shape[1] and 0 <= y2 < depth_map.shape[0]):
                
                z1 = depth_map[y1, x1]
                z2 = depth_map[y2, x2]
                
                # Calcular el punto medio del par
                mid_point = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    (z1 + z2) / 2
                ]
                
                keypoints_3d.append(mid_point)
                valid_pairs[pair_name] = mid_point
    
    return np.array(keypoints_3d), valid_pairs

def estimate_transformation(source_points, target_points):
    """
    Estima la transformación rígida entre dos conjuntos de puntos 3D
    asegurando una transformación válida en los tres ejes
    """
    if len(source_points) < 3 or len(target_points) < 3:
        print("No hay suficientes keypoints para estimar la transformación")
        return None, None
    
    # Calcular centroide
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    # Centrar puntos
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    # Calcular matriz de covarianza
    H = source_centered.T @ target_centered
    
    try:
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Calcular matriz de rotación
        R = Vt.T @ U.T
        
        # Asegurar una rotación válida (determinante positivo)
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        
        # Verificar que la matriz de rotación es válida
        if not np.allclose(np.linalg.det(R), 1.0, rtol=1e-4):
            print("Advertencia: La matriz de rotación puede no ser válida")
            return None, None
        
        # Calcular traslación
        t = target_centroid - R @ source_centroid
        
        # Verificar que la transformación es razonable
        max_rotation = np.max(np.abs(R - np.eye(3)))
        max_translation = np.max(np.abs(t))
        
        if max_rotation > 0.9 or max_translation > 5.0:  # Valores umbral ajustables
            print("Advertencia: Transformación posiblemente incorrecta")
            return None, None
        
        return R, t
        
    except np.linalg.LinAlgError:
        print("Error en el cálculo de SVD")
        return None, None
    
def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, 
                      model_name="zoedepth", max_depth=10.0, voxel_size=0.02, target_size=(640, 480)):
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Configurar modelo de profundidad
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Configurar modelo de pose
    pose_config = './td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    pose_checkpoint = './td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=DEVICE)

    # Procesar primera imagen y máscara
    img1 = Image.open(image_path1).convert("RGB")
    img1 = resize_image(img1, target_size)
    img_np1 = np.array(img1)
    
    mask_img1 = Image.open(mask_path1).convert("RGBA")
    mask_img1 = resize_image(mask_img1, target_size)
    mask_np1 = np.array(mask_img1)
    alpha_channel1 = mask_np1[:, :, 3]

    # Procesar segunda imagen y máscara
    img2 = Image.open(image_path2).convert("RGB")
    img2 = resize_image(img2, target_size)
    img_np2 = np.array(img2)
    
    mask_img2 = Image.open(mask_path2).convert("RGBA")
    mask_img2 = resize_image(mask_img2, target_size)
    mask_np2 = np.array(mask_img2)
    alpha_channel2 = mask_np2[:, :, 3]

    with torch.no_grad():
        depth1 = model.infer_pil(img1)
        depth2 = model.infer_pil(img2)

    # Obtener keypoints 3D
    # En process_two_images:
    keypoints3d_1, pairs1 = get_3d_keypoints(img_np1, depth1, pose_model)
    keypoints3d_2, pairs2 = get_3d_keypoints(img_np2, depth2, pose_model)

    R, t = estimate_transformation(keypoints3d_1, keypoints3d_2)

    if R is None or t is None:
        print("No se pudo estimar una transformación válida. Usando identidad.")
        R = np.eye(3)
        t = np.zeros(3)

    # Convertir profundidad a puntos 3D
    points1 = depth_to_points(depth1[None])
    points2 = depth_to_points(depth2[None])

    # Aplanar puntos
    points1_flat = points1.reshape(-1, 3)
    points2_flat = points2.reshape(-1, 3)
    
    # Aplicar transformación a points2
    points2_transformed = (R @ points2_flat.T).T + t

    R_180 = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])
    
    center = np.mean(points2_transformed, axis=0)
    points2_transformed = points2_transformed - center
    points2_transformed = (R_180 @ points2_transformed.T).T
    points2_transformed = points2_transformed + center

    # Aplicar máscaras y filtros de profundidad
    depth_mask1 = points1_flat[:, 2] < max_depth
    depth_mask2 = points2_transformed[:, 2] < max_depth
    alpha_mask1 = alpha_channel1.reshape(-1) > 128
    alpha_mask2 = alpha_channel2.reshape(-1) > 128

    final_mask1 = depth_mask1 & alpha_mask1
    final_mask2 = depth_mask2 & alpha_mask2

    points1_filtered = points1_flat[final_mask1]
    points2_filtered = points2_transformed[final_mask2]

    colors1 = (img_np1.reshape(-1, 3)[final_mask1] / 255.0).astype(np.float32)
    colors2 = (img_np2.reshape(-1, 3)[final_mask2] / 255.0).astype(np.float32)

    # Crear nubes de puntos
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1_filtered)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2_filtered)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)

    # Downsampling
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    points_combined = np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
    center = points_combined.mean(axis=0)
    max_bound = points_combined.max(axis=0)
    min_bound = points_combined.min(axis=0)
    scene_scale = np.linalg.norm(max_bound - min_bound)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])

    # Configurar la cámara
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    
    # Ajustar la distancia de la cámara basada en la escala de la escena
    camera_distance = scene_scale * 1.5
    
    # Calcular nueva posición de la cámara
    camera_pos = center + np.array([0, 0, camera_distance])
    camera_params.extrinsic = np.array([
        [1, 0, 0, camera_pos[0]],
        [0, 1, 0, camera_pos[1]],
        [0, 0, 1, camera_pos[2]],
        [0, 0, 0, 1]
    ])
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    ctr.set_zoom(0.7)

    ctr.set_front([-0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(center)

    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualización de profundidad y nube de puntos 3D con máscaras y alineación por keypoints')
    parser.add_argument('--image1', '-i1', required=True, help='Primera imagen de entrada')
    parser.add_argument('--mask1', '-m1', required=True, help='Máscara para primera imagen')
    parser.add_argument('--image2', '-i2', required=True, help='Segunda imagen de entrada')
    parser.add_argument('--mask2', '-m2', required=True, help='Máscara para segunda imagen')
    parser.add_argument('--output', '-o', default='output', help='Path salida')
    parser.add_argument('--model', '-md', default='zoedepth', help='Modelo utilizado')
    parser.add_argument('--max-depth', type=float, default=10.0, help='Profundidad máxima en metros (default: 10.0)')
    parser.add_argument('--voxel-size', type=float, default=0.02, help='Tamaño de voxel para downsampling (default: 0.02)')
    parser.add_argument('--width', type=int, default=640, help='Ancho de la imagen redimensionada (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Alto de la imagen redimensionada (default: 480)')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    target_size = (args.width, args.height)
    
    process_two_images(args.image1, args.mask1, args.image2, args.mask2, 
                      args.output, args.model, args.max_depth, args.voxel_size,
                      target_size)

if __name__ == "__main__":
    main()