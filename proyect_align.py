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

def get_3d_keypoints(img_np, depth_map, pose_model, score_threshold=0.3, visualize=True):
    """
    Obtiene keypoints 3D específicos usando MMPose y el mapa de profundidad.
    Visualiza los keypoints 2D sobre la imagen si 'visualize' es True.
    Ajusta el umbral de score a 'score_threshold'.
    """
    # Inferencia de pose 2D
    mmpose_results = inference_topdown(pose_model, img_np)
    keypoints_2d = mmpose_results[0].pred_instances.keypoints[0]
    keypoints_scores = mmpose_results[0].pred_instances.keypoint_scores[0]
    
    # Visualizar keypoints en la imagen
    if visualize:
        img_vis = img_np.copy()
        for i, (x, y) in enumerate(keypoints_2d):
            score = keypoints_scores[i]
            if score > score_threshold:
                cv2.circle(img_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(img_vis, f"{i}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 1)
        cv2.imshow("Keypoints Detectados", img_vis)
        print("Presiona cualquier tecla en la ventana para continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Definir pares de keypoints que queremos usar
    keypoint_pairs = {
        'shoulders': (5, 6),    # left_shoulder, right_shoulder
        'hips': (11, 12),       # left_hip, right_hip
        'knees': (13, 14)       # left_knee, right_knee
    }
    
    keypoints_3d = []
    valid_pairs = {}

    for pair_name, (idx1, idx2) in keypoint_pairs.items():
        if keypoints_scores[idx1] > score_threshold and keypoints_scores[idx2] > score_threshold:
            x1, y1 = map(int, keypoints_2d[idx1])
            x2, y2 = map(int, keypoints_2d[idx2])
            
          
            if (0 <= x1 < depth_map.shape[1] and 0 <= y1 < depth_map.shape[0] and
                0 <= x2 < depth_map.shape[1] and 0 <= y2 < depth_map.shape[0]):
                
                z1 = depth_map[y1, x1]
                z2 = depth_map[y2, x2]
                
                # Punto medio 3D
                mid_point = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    (z1 + z2) / 2
                ]
                
                keypoints_3d.append(mid_point)
                valid_pairs[pair_name] = mid_point
    
    return np.array(keypoints_3d), valid_pairs


def erode_alpha_mask(alpha_channel, kernel_size=5, iterations=1):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(alpha_channel, kernel, iterations=iterations)
    return eroded_mask

def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, 
                      model_name="zoedepth", max_depth=10.0, voxel_size=0.02, target_size=(640, 480)):
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Configurar modelo de profundidad
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()
    """
    # Configurar modelo de pose
    pose_config = './td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    pose_checkpoint = './td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=DEVICE)
    """
    
    # Procesar primera imagen
    img1 = Image.open(image_path1).convert("RGB")
    img1 = resize_image(img1, target_size)
    img_np1 = np.array(img1)
    
    mask_img1 = Image.open(mask_path1).convert("RGBA")
    mask_img1 = resize_image(mask_img1, target_size)
    mask_np1 = np.array(mask_img1)
    alpha_channel1 = mask_np1[:, :, 3]

    # CONTRACCIÓN DE LA MÁSCARA
    alpha_channel1_eroded = erode_alpha_mask(alpha_channel1, kernel_size=3, iterations=2)

    # Procesar segunda imagen
    img2 = Image.open(image_path2).convert("RGB")
    img2 = resize_image(img2, target_size)
    img_np2 = np.array(img2)
    
    mask_img2 = Image.open(mask_path2).convert("RGBA")
    mask_img2 = resize_image(mask_img2, target_size)
    mask_np2 = np.array(mask_img2)
    alpha_channel2 = mask_np2[:, :, 3]

    # CONTRACCIÓN DE LA MÁSCARA
    alpha_channel2_eroded = erode_alpha_mask(alpha_channel2, kernel_size=3, iterations=2)

    # Inferir mapas de profundidad
    with torch.no_grad():
        depth1 = model.infer_pil(img1)
        depth2 = model.infer_pil(img2)

    # Filtrado con las máscaras erosionadas
    points1 = depth_to_points(depth1[None])  # (1, H, W, 3)
    points2 = depth_to_points(depth2[None])  # (1, H, W, 3)
    points1_flat = points1.reshape(-1, 3)
    points2_flat = points2.reshape(-1, 3)

    alpha_mask1 = alpha_channel1_eroded.reshape(-1) > 128
    alpha_mask2 = alpha_channel2_eroded.reshape(-1) > 128

    points1_filtered = points1_flat[alpha_mask1]
    points2_filtered = points2_flat[alpha_mask2]

    colors1 = (img_np1.reshape(-1, 3)[alpha_mask1] / 255.0).astype(np.float32)
    colors2 = (img_np2.reshape(-1, 3)[alpha_mask2] / 255.0).astype(np.float32)

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

    # ===========================
    # 2) REGISTRO GLOBAL (FPFH + RANSAC)
    # ===========================
    # a) Estimamos normales (requisito para FPFH)
    pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    #Calcular descriptores FPFH
    radius_feature = voxel_size * 5
    pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd2,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    #Registro global RANSAC
    distance_threshold = voxel_size * 1.5
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
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    T_ransac = result_ransac.transformation
    print("Transformation from RANSAC:\n", T_ransac)

    pcd2.transform(T_ransac)

    # (Opcional) REFINE con ICP
    distance_threshold_icp = voxel_size * 1.0
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1,
        distance_threshold_icp,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T_icp = result_icp.transformation
    print("Transformation refined via ICP:\n", T_icp)

    # Aplicamos la ref. final a pcd2 (acumulando la matriz de RANSAC)
    pcd2.transform(T_icp)

    center2 = pcd2.get_center()
    pcd2.translate(-center2)
    R_180 = pcd2.get_rotation_matrix_from_axis_angle([0, np.pi, 0])
    pcd2.rotate(R_180, center=(0, 0, 0))
    center2[2] = center2[2] + 0.4
    pcd2.translate(center2)

    pcd_comb = pcd1 + pcd2
    output_pcd1 = os.path.join(output_dir, "combined_pcd1.ply")
    output_pcd2 = os.path.join(output_dir, "combined_pcd2.ply")
    output_ply_path = os.path.join(output_dir, "combined_pt.ply")
    o3d.io.write_point_cloud(output_ply_path, pcd_comb)
    o3d.io.write_point_cloud(output_pcd1, pcd1)
    o3d.io.write_point_cloud(output_pcd2, pcd2)
    # ===========================
    # Visualizacion Open3D
    # ===========================
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
