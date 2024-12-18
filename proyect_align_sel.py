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

def create_sphere(center, radius=0.02, color=[1, 0, 0]):
    """Crea una esfera en la posición especificada"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere

def visualize_point_clouds(pcd1, pcd2, highlighted_pcd):
    """Visualiza las nubes de puntos usando el renderizador basado en GPU"""
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Crear ventana
    window = app.create_window("Visualización de Nubes de Puntos", 1280, 720)

    # Crear widget de escena
    scene_widget = o3d.visualization.gui.SceneWidget()
    scene_widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # Añadir geometrías a la escena
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0

    scene_widget.scene.add_geometry("PointCloud1", pcd1, material)
    scene_widget.scene.add_geometry("PointCloud2", pcd2, material)
    
    if len(highlighted_pcd.points) > 0:
        # Crear esferas para los puntos destacados
        for i, point in enumerate(highlighted_pcd.points):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.paint_uniform_color([1, 0, 0])  # Color rojo
            sphere.translate(point)
            scene_widget.scene.add_geometry(f"Sphere_{i}", sphere, material)

    # Configurar cámara usando los límites combinados de ambas nubes de puntos
    bounds1 = pcd1.get_axis_aligned_bounding_box()
    bounds2 = pcd2.get_axis_aligned_bounding_box()
    
    # Combinar los límites manualmente
    min_bound = np.minimum(bounds1.min_bound, bounds2.min_bound)
    max_bound = np.maximum(bounds1.max_bound, bounds2.max_bound)
    
    # Crear un nuevo bounding box combinado
    combined_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Configurar la cámara usando los límites combinados
    scene_widget.setup_camera(60, combined_bounds, combined_bounds.get_center())

    app.run()

def process_two_images(image_path1, image_path2, output_dir, 
                      model_name="zoedepth", target_size=(640, 480)):
    # Inicializar modelo
    depth_model = init_model(model_name)
    
    # Seleccionar puntos en la primera imagen y obtener dimensiones originales
    points_2d, img_shape = select_points(image_path1)
    
    # Cargar y procesar imágenes con canal alpha
    img1 = Image.open(image_path1).convert("RGBA")
    img2 = Image.open(image_path2).convert("RGBA")
    
    # Obtener máscaras alpha
    alpha_mask1 = np.array(img1.split()[-1]) > 0
    alpha_mask2 = np.array(img2.split()[-1]) > 0
    
    # Convertir a RGB para el modelo de profundidad
    img1_rgb = img1.convert("RGB")
    img2_rgb = img2.convert("RGB")
    
    # Redimensionar
    img1_rgb = resize_image(img1_rgb, target_size)
    img2_rgb = resize_image(img2_rgb, target_size)
    
    # Redimensionar máscaras alpha
    alpha_mask1 = Image.fromarray(alpha_mask1)
    alpha_mask2 = Image.fromarray(alpha_mask2)
    alpha_mask1 = np.array(alpha_mask1.resize(target_size, Image.NEAREST))
    alpha_mask2 = np.array(alpha_mask2.resize(target_size, Image.NEAREST))
    
    # Convertir imágenes a numpy arrays
    img_np1 = np.array(img1_rgb)
    img_np2 = np.array(img2_rgb)
    
    # Obtener mapas de profundidad
    with torch.no_grad():
        depth1 = depth_model.infer_pil(img1_rgb)
        depth2 = depth_model.infer_pil(img2_rgb)
    
    # Aplicar máscaras a los mapas de profundidad
    depth1[~alpha_mask1] = 0
    depth2[~alpha_mask2] = 0
    
    # Convertir a puntos 3D
    points1 = depth_to_points(depth1[None])
    points2 = depth_to_points(depth2[None])
    
    # Crear nubes de puntos con filtrado por máscara
    valid_points1 = alpha_mask1.reshape(-1)
    valid_points2 = alpha_mask2.reshape(-1)
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1.reshape(-1, 3)[valid_points1])
    pcd1.colors = o3d.utility.Vector3dVector((img_np1.reshape(-1, 3)[valid_points1] / 255.0).astype(np.float32))
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.reshape(-1, 3)[valid_points2])
    pcd2.colors = o3d.utility.Vector3dVector((img_np2.reshape(-1, 3)[valid_points2] / 255.0).astype(np.float32))
    
    # Procesar puntos seleccionados
    highlighted_points = []
    scale_x = target_size[0] / img_shape[1]
    scale_y = target_size[1] / img_shape[0]
    
    for x, y in points_2d:
        # Escalar las coordenadas 2D
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        
        # Verificar si el punto está dentro de la máscara
        if (scaled_x < target_size[0] and scaled_y < target_size[1] and 
            alpha_mask1[scaled_y, scaled_x]):
            idx = scaled_y * target_size[0] + scaled_x
            if idx < len(points1.reshape(-1, 3)):
                point_3d = points1.reshape(-1, 3)[idx]
                highlighted_points.append(point_3d)
    
    # Visualización
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    window = app.create_window("Visualización de Nubes de Puntos", 1280, 720)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)
    
    # Material para las nubes de puntos
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0
    
    # Añadir nubes de puntos filtradas
    widget3d.scene.add_geometry("PointCloud1", pcd1, material)
    widget3d.scene.add_geometry("PointCloud2", pcd2, material)
    
    # Añadir esferas para los puntos seleccionados
    sphere_material = o3d.visualization.rendering.MaterialRecord()
    sphere_material.base_color = [1.0, 0.0, 0.0, 1.0]
    sphere_material.shader = "defaultLit"
    
    for i, point in enumerate(highlighted_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])
        widget3d.scene.add_geometry(f"Sphere_{i}", sphere, sphere_material)
    
    # Configurar cámara
    bounds = pcd1.get_axis_aligned_bounding_box()
    bounds.extend(pcd2.get_axis_aligned_bounding_box())
    center = bounds.get_center()
    
    widget3d.setup_camera(60, bounds, center)
    widget3d.look_at(center, center + [0, 0, 3], [0, -1, 0])
    
    app.run()

def select_points(image_path):
    """Permite al usuario seleccionar puntos en la imagen y devuelve sus coordenadas"""
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"No se encontró la imagen en {image_path}")

    height, width = original_img.shape[:2]
    max_dimension = 800
    scale = min(max_dimension/width, max_dimension/height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    img = cv2.resize(original_img, (new_width, new_height))
    points = []
    scale_factor = (width/new_width, height/new_height)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(x * scale_factor[0])
            original_y = int(y * scale_factor[1])
            points.append((original_x, original_y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Image', img)

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)

    print("Selecciona los puntos en la imagen (Presiona 'q' cuando termines)")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return points, (height, width)

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