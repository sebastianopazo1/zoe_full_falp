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

def transform_points(points, transformation):
    """
    Aplica una matriz de transformación 4x4 a un conjunto de puntos 3D.
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # Convertir a coordenadas homogéneas
    transformed_points = (transformation @ points_homogeneous.T).T
    return transformed_points[:, :3]  # Convertir de vuelta a coordenadas 3D

def erode_alpha_mask(alpha_channel, kernel_size=5, iterations=1):
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(alpha_channel, kernel, iterations=iterations)
    return eroded_mask

def process_image(image_path, mask_path, model, target_size, voxel_size=0.02):
    """
    Procesa una sola imagen con su máscara, permitiendo seleccionar puntos 2D y obtener nubes de puntos 3D.
    """
    img = Image.open(image_path).convert("RGB")
    img = resize_image(img, target_size)
    img_np = np.array(img)

    mask_img = Image.open(mask_path).convert("RGBA")
    mask_img = resize_image(mask_img, target_size)
    mask_np = np.array(mask_img)
    alpha_channel = mask_np[:, :, 3]

    # Contracción de la máscara
    alpha_channel_eroded = erode_alpha_mask(alpha_channel, kernel_size=5, iterations=2)

    # Inferir mapa de profundidad
    with torch.no_grad():
        depth = model.infer_pil(img)

    # Selección manual de puntos 2D
    points_2d = select_points_2d(img_np.copy(), "Seleccionar puntos en Imagen")
    points_3d_selected = []
    for x, y in points_2d:
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            z = depth[y, x]
            points_3d_selected.append([x, y, z])
            print(f"Punto 3D seleccionado: ({x}, {y}, {z:.4f})")

    points_3d_selected = np.array(points_3d_selected)

    # Crear nube de puntos con filtrado alpha
    points = depth_to_points(depth[None])
    points_flat = points.reshape(-1, 3)
    alpha_mask = alpha_channel_eroded.reshape(-1) > 128
    points_filtered = points_flat[alpha_mask]
    colors = (img_np.reshape(-1, 3)[alpha_mask] / 255.0).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, points_3d_selected

def select_points_2d(image_np, window_name="Seleccionar puntos"):
    """
    Permite seleccionar manualmente puntos 2D con el mouse en una ventana de OpenCV.
    Retorna una lista con las coordenadas (x, y) de los puntos seleccionados.
    """
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Punto seleccionado: ({x}, {y})")
            points.append((x, y))
            cv2.circle(image_np, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, image_np)

    cv2.imshow(window_name, image_np)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Haz clic izquierdo para seleccionar puntos. Presiona cualquier tecla para finalizar...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    return points
def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, 
                      model_name="zoedepth", max_depth=10.0, voxel_size=0.02, target_size=(640, 480)):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Configurar modelo de profundidad
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Procesar imágenes
    pcd1, points_3d_1 = process_image(image_path1, mask_path1, model, target_size, voxel_size)
    pcd2, points_3d_2 = process_image(image_path2, mask_path2, model, target_size, voxel_size)

    # Downsampling
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    # Estimación de normales
    pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

    # Calcular descriptores FPFH
    radius_feature = voxel_size * 5
    pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    # Registro global
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1, pcd2, pcd1_fpfh, pcd2_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    T_ransac = result_ransac.transformation
    print("Transformación RANSAC:\n", T_ransac)

    # Aplicar transformación a puntos seleccionados
    points_3d_2_transformed = transform_points(points_3d_2, T_ransac)

    # Visualización final
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    # Agregar nubes de puntos
    vis.add_geometry(pcd1)
    pcd2.transform(T_ransac)
    vis.add_geometry(pcd2)

    # Agregar puntos seleccionados como rojo
    selected_pcd1 = o3d.geometry.PointCloud()
    selected_pcd1.points = o3d.utility.Vector3dVector(points_3d_1)
    selected_pcd1.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(selected_pcd1)

    selected_pcd2 = o3d.geometry.PointCloud()
    selected_pcd2.points = o3d.utility.Vector3dVector(points_3d_2_transformed)
    selected_pcd2.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(selected_pcd2)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])

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

