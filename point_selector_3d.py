import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import cv2

def select_points(image_path):
    """Allow user to select points on the image and return their coordinates"""
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"No se encontro la imagen en {image_path}.")
    
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
    
    print("Selecciona los puntos en la imagen (Apretar 'q' cuando termine)")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return points, original_img.shape[:2]

def process_image_with_points(image_path, model_name="zoedepth"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    points_2d, img_shape = select_points(image_path)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_pil(img)
    
    points_3d = depth_to_points(depth[None])

    pcd = o3d.geometry.PointCloud()
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
  
    highlighted_points = []
    for x, y in points_2d:
        idx = y * img_shape[1] + x
        point_3d = points_3d.reshape(-1, 3)[idx]
        highlighted_points.append(point_3d)
    
    if highlighted_points:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=720)
        
        # Añadir la nube de puntos principal
        vis.add_geometry(pcd)
        
        # Añadir los puntos destacados
        highlighted_pcd = o3d.geometry.PointCloud()
        highlighted_pcd.points = o3d.utility.Vector3dVector(np.array(highlighted_points))
        highlighted_colors = np.array([[1, 0, 0] for _ in highlighted_points])
        highlighted_pcd.colors = o3d.utility.Vector3dVector(highlighted_colors)
        vis.add_geometry(highlighted_pcd)
        
        # Añadir esferas rojas para los puntos seleccionados
        for i in range(len(highlighted_points)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(highlighted_points[i])
            sphere.paint_uniform_color([1, 0, 0])
            vis.add_geometry(sphere)
        
        # Añadir visualización de la cámara
        # 1. Sistema de coordenadas de la cámara
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        vis.add_geometry(camera_frame)
        
        # 2. Esfera para marcar posición de la cámara
        camera_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        camera_pos.paint_uniform_color([0, 1, 0])  # Verde para la cámara
        vis.add_geometry(camera_pos)
        
        # 3. Crear un cono para representar la dirección de la cámara
        camera_direction = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.1)
        # Rotar el cono para que apunte en la dirección Z
        R = camera_direction.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
        camera_direction.rotate(R, center=(0, 0, 0))
        camera_direction.paint_uniform_color([0, 0, 1])  # Azul para la dirección
        vis.add_geometry(camera_direction)
        
        # Configurar opciones de renderizado
        opt = vis.get_render_option()
        opt.point_size = 1.0
        opt.background_color = np.asarray([0, 0, 0])
        
        # Configurar la vista inicial
        ctr = vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        
        vis.run()
        vis.destroy_window()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D point selection and visualization')
    parser.add_argument('--image','-i', required=True, help='Input image path')
    parser.add_argument('--model', '-m', default='zoedepth', help='Model to use')
    
    args = parser.parse_args()
    process_image_with_points(args.image, args.model)

if __name__ == "__main__":
    main()