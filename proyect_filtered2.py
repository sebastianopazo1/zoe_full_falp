import torch
from PIL import Image
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d
import os
import argparse

def resize_image(image, max_size=(640, 480)):
    """
    Redimensiona la imagen manteniendo la proporción hasta que su lado más largo sea max_size
    """
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, 
                      model_name="zoedepth", max_depth=10.0, voxel_size=0.02, target_size=(640, 480)):
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Configurar modelo
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Procesar primera imagen y máscara
    img1 = Image.open(image_path1).convert("RGB")
    img1 = resize_image(img1, target_size)
    print(f"Tamaño de imagen 1 después de resize: {img1.size}")
    
    mask_img1 = Image.open(mask_path1).convert("RGBA")
    mask_img1 = resize_image(mask_img1, target_size)
    mask_np1 = np.array(mask_img1)
    alpha_channel1 = mask_np1[:, :, 3]
    img_np1 = np.array(img1)

    img2 = Image.open(image_path2).convert("RGB")
    img2 = resize_image(img2, target_size)
    print(f"Tamaño de imagen 2 después de resize: {img2.size}")
    
    mask_img2 = Image.open(mask_path2).convert("RGBA")
    mask_img2 = resize_image(mask_img2, target_size)
    mask_np2 = np.array(mask_img2)
    alpha_channel2 = mask_np2[:, :, 3]
    img_np2 = np.array(img2)

    with torch.no_grad():
        #prediccion profundidad
        depth1 = model.infer_pil(img1)
        depth2 = model.infer_pil(img2)

    R = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    print(R.shape)
    # Convertir profundidad a puntos 3D
    points1 = depth_to_points(depth1[None])
    points2 = depth_to_points(depth2[None], R = R)

    # Aplanar puntos
    points1_flat = points1.reshape(-1, 3)
    points2_flat = points2.reshape(-1, 3)
    depth_mask1 = points1_flat[:, 2] < max_depth
    depth_mask2 = points2_flat[:, 2] < max_depth

    alpha_mask1 = alpha_channel1.reshape(-1) > 128
    alpha_mask2 = alpha_channel2.reshape(-1) > 128

    final_mask1 = depth_mask1 & alpha_mask1
    final_mask2 = depth_mask2 & alpha_mask2

    #aplicacion de mascras
    points1_filtered = points1_flat[final_mask1]
    points2_filtered = points2_flat[final_mask2]

    #obtencion colores
    colors1 = (img_np1.reshape(-1, 3)[final_mask1] / 255.0).astype(np.float32)
    colors2 = (img_np2.reshape(-1, 3)[final_mask2] / 255.0).astype(np.float32)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1_filtered)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2_filtered)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)

    #voxel downsampling
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)

    #COnfiguracion renderizado
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualización de profundidad y nube de puntos 3D con máscaras para dos imágenes')
    parser.add_argument('--image1', '-i1', required=True, help='Primera imagen de entrada')
    parser.add_argument('--mask1', '-m1', required=True, help='Máscara para primera imagen')
    parser.add_argument('--image2', '-i2', required=True, help='Segunda imagen de entrada')
    parser.add_argument('--mask2', '-m2', required=True, help='Máscara para segunda imagen')
    parser.add_argument('--output', '-o', default='output', help='Path salida')
    parser.add_argument('--model', '-md', default='zoedepth', help='Modelo utilizado')
    parser.add_argument('--max-depth', type=float, default=10.0, help='Profundidad máxima en metros (default: 10.0)')
    parser.add_argument('--voxel-size', type=float, default=0.02, help='Tamaño de voxel para downsampling (default: 0.02)')
    parser.add_argument('--width', type=int, default=1024, help='Ancho de la imagen redimensionada (default: 640)')
    parser.add_argument('--height', type=int, default=1280, help='Alto de la imagen redimensionada (default: 480)')

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    target_size = (args.width, args.height)
    
    process_two_images(args.image1, args.mask1, args.image2, args.mask2, 
                      args.output, args.model, args.max_depth, args.voxel_size,
                      target_size)

if __name__ == "__main__":
    main()