import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import os
import argparse
import cv2

def init_model(model_name="zoedepth"):
    """Inicializa el modelo de profundidad"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()
    return model

def resize_image(image, target_size=(640, 480)):
    """Redimensiona la imagen manteniendo la proporción"""
    image.thumbnail(target_size, Image.LANCZOS)
    return image

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
    
    # Redimensionar imágenes RGB
    img1_rgb = resize_image(img1_rgb, target_size)
    img2_rgb = resize_image(img2_rgb, target_size)
    
    # Obtener mapas de profundidad
    with torch.no_grad():
        depth1 = depth_model.infer_pil(img1_rgb)
        depth2 = depth_model.infer_pil(img2_rgb)
    
    # Redimensionar máscaras alpha para que coincidan con las dimensiones del mapa de profundidad
    depth_shape = depth1.shape
    alpha_mask1 = Image.fromarray(alpha_mask1.astype(np.uint8) * 255)
    alpha_mask2 = Image.fromarray(alpha_mask2.astype(np.uint8) * 255)
    alpha_mask1 = np.array(alpha_mask1.resize((depth_shape[1], depth_shape[0]), Image.NEAREST)) > 0
    alpha_mask2 = np.array(alpha_mask2.resize((depth_shape[1], depth_shape[0]), Image.NEAREST)) > 0
    
    # Aplicar máscaras a los mapas de profundidad
    depth1[~alpha_mask1] = 0
    depth2[~alpha_mask2] = 0
    
    # Convertir a puntos 3D
    points1 = depth_to_points(depth1[None])
    points2 = depth_to_points(depth2[None])
    
    # Crear nubes de puntos con filtrado por máscara
    valid_points1 = alpha_mask1.reshape(-1)
    valid_points2 = alpha_mask2.reshape(-1)
    
    # Crear nubes de puntos
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1.reshape(-1, 3)[valid_points1])
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2.reshape(-1, 3)[valid_points2])
    
    # Asignar colores
    img_np1 = np.array(img1_rgb.resize((depth_shape[1], depth_shape[0])))
    img_np2 = np.array(img2_rgb.resize((depth_shape[1], depth_shape[0])))
    
    colors1 = (img_np1.reshape(-1, 3)[valid_points1] / 255.0).astype(np.float32)
    colors2 = (img_np2.reshape(-1, 3)[valid_points2] / 255.0).astype(np.float32)
    
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    
    # Crear nube de puntos destacados
    highlighted_pcd = o3d.geometry.PointCloud()
    highlighted_points = []
    
    # Escalar coordenadas 2D
    scale_x = depth_shape[1] / img_shape[1]
    scale_y = depth_shape[0] / img_shape[0]
    
    for x, y in points_2d:
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        
        if (scaled_x < depth_shape[1] and scaled_y < depth_shape[0] and 
            alpha_mask1[scaled_y, scaled_x]):
            idx = scaled_y * depth_shape[1] + scaled_x
            if idx < len(points1.reshape(-1, 3)):
                point_3d = points1.reshape(-1, 3)[idx]
                highlighted_points.append(point_3d)
    
    if highlighted_points:
        highlighted_pcd.points = o3d.utility.Vector3dVector(highlighted_points)
    
    # Visualización
    visualize_point_clouds(pcd1, pcd2, highlighted_pcd)

def visualize_point_clouds(pcd1, pcd2, highlighted_pcd):
    """Visualiza las nubes de puntos usando el renderizador basado en GPU"""
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Crear ventana
    window = app.create_window("Visualización de Nubes de Puntos", 1280, 720)
    widget3d = o3d.visualization.gui.SceneWidget()
    widget3d.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(widget3d)

    # Material para las nubes de puntos
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0

    # Añadir nubes de puntos
    widget3d.scene.add_geometry("PointCloud1", pcd1, material)
    widget3d.scene.add_geometry("PointCloud2", pcd2, material)

    # Añadir puntos destacados si existen
    if len(highlighted_pcd.points) > 0:
        sphere_material = o3d.visualization.rendering.MaterialRecord()
        sphere_material.base_color = [1.0, 0.0, 0.0, 1.0]
        sphere_material.shader = "defaultLit"

        for i, point in enumerate(highlighted_pcd.points):
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

def main():
    parser = argparse.ArgumentParser(description='Visualización de profundidad y nube de puntos 3D')
    parser.add_argument('--image1', '-i1', required=True, help='Primera imagen de entrada')
    parser.add_argument('--image2', '-i2', required=True, help='Segunda imagen de entrada')
    parser.add_argument('--mask1', '-m1', required=True, help='Máscara para primera imagen')
    parser.add_argument('--mask2', '-m2', required=True, help='Máscara para segunda imagen')
    parser.add_argument('--output', '-o', default='output', help='Directorio de salida')
    parser.add_argument('--model', '-m', default='zoedepth', help='Nombre del modelo')
    parser.add_argument('--width', type=int, default=640, help='Ancho de imagen')
    parser.add_argument('--height', type=int, default=480, help='Alto de imagen')
    parser.add_argument('--voxel-size', type=float, default=0.02, help='Tamaño de voxel para downsampling')
    
    args = parser.parse_args()
    
    target_size = (args.width, args.height)
    
    process_two_images(
        args.image1,
        args.image2,
        args.output,
        model_name=args.model,
        target_size=target_size,
        mask1_path=args.mask1,
        mask2_path=args.mask2,
        voxel_size=args.voxel_size
    )

if __name__ == "__main__":
    main()