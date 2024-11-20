import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import cv2

def select_points(image_path):
    """Permite al usuario seleccionar puntos en la imagen y devuelve sus coordenadas."""
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"No se encontró la imagen en {image_path}.")

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
    print(f"Usando dispositivo: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    points_2d, img_shape = select_points(image_path)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    with torch.no_grad():
        depth = model.infer_pil(img)
    
    points_3d = depth_to_points(depth[None])

    # Crear nube de puntos
    points = points_3d.reshape(-1, 3)
    colors = img_np.reshape(-1, 3) / 255.0

    highlighted_points = []
    for x, y in points_2d:
        idx = y * img_shape[1] + x
        point_3d = points_3d.reshape(-1, 3)[idx]
        highlighted_points.append(point_3d)
    
    # Visualización con Open3D utilizando el nuevo renderizador basado en GPU
    visualize_point_cloud(points, colors, highlighted_points)

def visualize_point_cloud(points, colors, highlighted_points):
    # Crear la nube de puntos principal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

    # Crear la nube de puntos destacada
    highlighted_pcd = o3d.geometry.PointCloud()
    highlighted_pcd.points = o3d.utility.Vector3dVector(np.array(highlighted_points))
    highlighted_colors = np.array([[1, 0, 0] for _ in highlighted_points], dtype=np.float32)
    highlighted_pcd.colors = o3d.utility.Vector3dVector(highlighted_colors)

    # Iniciar la aplicación de visualización de Open3D
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Crear una ventana
    window = app.create_window("Visualización de Nube de Puntos", 1280, 720)

    # Crear un widget de escena
    scene_widget = o3d.visualization.gui.SceneWidget()
    scene_widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(scene_widget)

    # Añadir geometrías a la escena
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    scene_widget.scene.add_geometry("PointCloud", pcd, material)
    scene_widget.scene.add_geometry("HighlightedPoints", highlighted_pcd, material)

    # Configurar cámara
    bbox = pcd.get_axis_aligned_bounding_box()
    scene_widget.setup_camera(60, bbox, bbox.get_center())

    # Ejecutar la aplicación
    app.run()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Selección y visualización de puntos en 3D')
    parser.add_argument('--image','-i', required=True, help='Ruta de la imagen de entrada')
    parser.add_argument('--model', '-m', default='zoedepth', help='Modelo a utilizar')
    
    args = parser.parse_args()
    process_image_with_points(args.image, args.model)

if __name__ == "__main__":
    main()
