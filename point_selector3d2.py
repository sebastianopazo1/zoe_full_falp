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
    return points

def process_image_with_points(image_path, model_name="zoedepth", mask_path=None):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Seleccionar puntos 2D
    points_2d = select_points(image_path)

    # Cargar imagen con PIL
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    # Cargar y preparar la máscara (si se proporciona)
    if mask_path is not None:
        # Convertir a RGBA para extraer el canal alfa
        mask_img = Image.open(mask_path).convert("RGBA")
        if mask_img.size != img.size:
            # Redimensionar la máscara si no coincide
            mask_img = mask_img.resize(img.size, Image.NEAREST)
        # Extraer canal alfa
        _, _, _, a = mask_img.split()
        mask_np = np.array(a)
        # Crear máscara booleana
        mask = (mask_np > 0)
    else:
        mask = np.ones((H, W), dtype=bool)

    print(f"Imagen: {H}x{W}")
    print(f"Máscara: {mask.shape}, puntos válidos: {np.sum(mask)}")

    with torch.no_grad():
        depth = model.infer_pil(img)
    
    # Convertir profundidad a puntos 3D (H, W, 3)
    points_3d = depth_to_points(depth[None])

    points_all = points_3d.reshape(-1, 3)
    colors_all = img_np.reshape(-1, 3) / 255.0
    mask_flat = mask.flatten()

    # Filtrado de puntos por máscara
    points_filtered = points_all[mask_flat]
    colors_filtered = colors_all[mask_flat]

    print(f"Puntos totales: {points_all.shape[0]}")
    print(f"Puntos filtrados: {points_filtered.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered.astype(np.float32))
    #pcd = pcd.voxel_down_sample(voxel_size=0.000001)
    
    # Filtrar puntos destacados según la máscara
    highlighted_points = []
    for x, y in points_2d:
        if 0 <= x < W and 0 <= y < H:
            idx = y * W + x
            if mask_flat[idx]:
                highlighted_points.append(points_all[idx])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    # Añadir la nube de puntos filtrada
    vis.add_geometry(pcd)
    
    if highlighted_points:
        highlighted_pcd = o3d.geometry.PointCloud()
        highlighted_pcd.points = o3d.utility.Vector3dVector(np.array(highlighted_points))
        highlighted_colors = np.array([[1, 0, 0] for _ in highlighted_points])
        highlighted_pcd.colors = o3d.utility.Vector3dVector(highlighted_colors)
        vis.add_geometry(highlighted_pcd)
        
        # Añadir esferas rojas para los puntos seleccionados
        for hp in highlighted_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(hp)
            sphere.paint_uniform_color([1, 0, 0])
            vis.add_geometry(sphere)
    
    # Añadir visualización de la cámara
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(camera_frame)
    
    camera_pos = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    camera_pos.paint_uniform_color([0, 1, 0])
    vis.add_geometry(camera_pos)
    
    camera_direction = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.1)
    R = camera_direction.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
    camera_direction.rotate(R, center=(0, 0, 0))
    camera_direction.paint_uniform_color([0, 0, 1])
    vis.add_geometry(camera_direction)
    
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.3)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    
    vis.run()
    vis.destroy_window()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D point selection and visualization with alpha mask')
    parser.add_argument('--image','-i', required=True, help='Input image path')
    parser.add_argument('--model', '-m', default='zoedepth', help='Model to use')
    parser.add_argument('--mask', '-k', help='Path to the mask image (PNG with alpha)')
    args = parser.parse_args()
    process_image_with_points(args.image, args.model, args.mask)

if __name__ == "__main__":
    main()
