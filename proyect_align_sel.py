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

def init_model(model_name="zoedepth"):
    """Inicializa el modelo de profundidad"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()
    return model

def init_pose_detector(config_file, checkpoint_file, device='cuda'):
    """Inicializa el detector de pose"""
    return init_pose_model(config_file, checkpoint_file, device=device)

def resize_image(image, max_size=(640, 480)):
    """Redimensiona la imagen manteniendo la proporción"""
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def load_and_process_images(image_path1, mask_path1, image_path2, mask_path2, target_size=(640, 480)):
    """Carga y procesa las imágenes y máscaras"""
    # Procesar primera imagen
    img1 = Image.open(image_path1).convert("RGB")
    img1 = resize_image(img1, target_size)
    mask_img1 = Image.open(mask_path1).convert("RGBA")
    mask_img1 = resize_image(mask_img1, target_size)
    
    # Procesar segunda imagen
    img2 = Image.open(image_path2).convert("RGB")
    img2 = resize_image(img2, target_size)
    mask_img2 = Image.open(mask_path2).convert("RGBA")
    mask_img2 = resize_image(mask_img2, target_size)
    
    return img1, mask_img1, img2, mask_img2

def get_depth_maps(model, img1, img2):
    """Obtiene mapas de profundidad para ambas imágenes"""
    with torch.no_grad():
        depth1 = model.infer_pil(img1)
        depth2 = model.infer_pil(img2)
    return depth1, depth2

def erode_alpha_mask(alpha_channel, kernel_size=5, iterations=1):
    """Aplica erosión a la máscara alpha"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(alpha_channel, kernel, iterations=iterations)

def process_masks(mask_img1, mask_img2):
    """Procesa las máscaras alpha"""
    mask_np1 = np.array(mask_img1)
    mask_np2 = np.array(mask_img2)
    
    alpha_channel1 = mask_np1[:, :, 3]
    alpha_channel2 = mask_np2[:, :, 3]
    
    alpha_channel1_eroded = erode_alpha_mask(alpha_channel1, kernel_size=3, iterations=2)
    alpha_channel2_eroded = erode_alpha_mask(alpha_channel2, kernel_size=3, iterations=2)
    
    return alpha_channel1_eroded, alpha_channel2_eroded

def create_point_clouds(points1, points2, colors1, colors2, voxel_size=0.02):
    """Crea y procesa nubes de puntos"""
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    
    # Downsampling
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)
    
    return pcd1, pcd2

def estimate_normals(pcd1, pcd2, voxel_size):
    """Estima normales para las nubes de puntos"""
    pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    return pcd1, pcd2

def register_point_clouds(pcd1, pcd2, voxel_size):
    """Improved point cloud registration with better parameters"""
    # Increase search radius for features
    radius_feature = voxel_size * 10  # Increased from 5
    
    # Compute FPFH features with more neighbors
    pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200)  # Increased max_nn
    )
    pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd2,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200)
    )
    
    # More relaxed distance threshold
    distance_threshold = voxel_size * 2.0  # Increased from 1.5
    
    # Modified RANSAC parameters
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1, pcd2, pcd1_fpfh, pcd2_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000)  # Increased iterations
    )
    
    # Refined ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1,
        distance_threshold,
        result_ransac.transformation,  # Use RANSAC result as initial alignment
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)  # Increased iterations
    )
    
    return result_ransac.transformation, result_icp.transformation

def setup_visualization():
    """Configura la visualización"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])
    
    return vis

def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, 
                      model_name="zoedepth", max_depth=10.0, voxel_size=0.02, target_size=(640, 480)):
    """Función principal que procesa dos imágenes"""
    # Inicializar modelos
    depth_model = init_model(model_name)
    
    # Cargar y procesar imágenes
    img1, mask_img1, img2, mask_img2 = load_and_process_images(
        image_path1, mask_path1, image_path2, mask_path2, target_size)
    
    # Obtener mapas de profundidad
    depth1, depth2 = get_depth_maps(depth_model, img1, img2)
    
    # Procesar máscaras
    alpha_mask1, alpha_mask2 = process_masks(mask_img1, mask_img2)
    
    # Convertir a arrays numpy
    img_np1 = np.array(img1)
    img_np2 = np.array(img2)
    
    # Crear nubes de puntos
    points1 = depth_to_points(depth1[None])
    points2 = depth_to_points(depth2[None])
    
    # Filtrar puntos con máscaras
    alpha_mask1_flat = alpha_mask1.reshape(-1) > 128
    alpha_mask2_flat = alpha_mask2.reshape(-1) > 128
    
    points1_filtered = points1.reshape(-1, 3)[alpha_mask1_flat]
    points2_filtered = points2.reshape(-1, 3)[alpha_mask2_flat]
    
    colors1 = (img_np1.reshape(-1, 3)[alpha_mask1_flat] / 255.0).astype(np.float32)
    colors2 = (img_np2.reshape(-1, 3)[alpha_mask2_flat] / 255.0).astype(np.float32)
    
    # Crear y procesar nubes de puntos
    pcd1, pcd2 = create_point_clouds(points1_filtered, points2_filtered, colors1, colors2, voxel_size)
    
    # Estimar normales
    pcd1, pcd2 = estimate_normals(pcd1, pcd2, voxel_size)
    
    # Registrar nubes de puntos
    T_ransac, T_icp = register_point_clouds(pcd1, pcd2, voxel_size)
    
    # Aplicar transformaciones
    pcd2.transform(T_ransac)
    pcd2.transform(T_icp)
    
    # Ajustes finales de posición
    center2 = pcd2.get_center()
    pcd2.translate(-center2)
    R_180 = pcd2.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
    pcd2.rotate(R_180, center=(0, 0, 0))
    center2[2] = center2[2] + 0.4
    pcd2.translate(center2)
    
    # Guardar resultados
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pcd_combined = pcd1 + pcd2
        o3d.io.write_point_cloud(os.path.join(output_dir, "combined_pt.ply"), pcd_combined)
        o3d.io.write_point_cloud(os.path.join(output_dir, "combined_pcd1.ply"), pcd1)
        o3d.io.write_point_cloud(os.path.join(output_dir, "combined_pcd2.ply"), pcd2)
    
    # Visualización
    vis = setup_visualization()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    
    # Configurar cámara
    points_combined = np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
    center = points_combined.mean(axis=0)
    max_bound = points_combined.max(axis=0)
    min_bound = points_combined.min(axis=0)
    scene_scale = np.linalg.norm(max_bound - min_bound)
    
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    camera_distance = scene_scale * 1.5
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
    parser = argparse.ArgumentParser(description='Visualización de profundidad y nube de puntos 3D con máscaras y alineación')
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
    
    process_two_images(
        args.image1, args.mask1, args.image2, args.mask2,
        args.output, args.model, args.max_depth, args.voxel_size,
        target_size=(args.width, args.height)
    )

if __name__ == "__main__":
    main()