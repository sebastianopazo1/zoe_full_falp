import numpy as np
import cv2
import sys
import os
import torch
import gc
from scipy.ndimage import binary_fill_holes
from PIL import Image
from matplotlib import pyplot as plt
from os import path, makedirs
from importlib import reload
from stitching import AffineStitcher
from ultralytics import YOLOWorld, SAM
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points

import open3d as o3d

def load_and_resize_images(folder, new_width):
    images = []
    filenames = ["0002.jpg", "0003.jpg", "0004.jpg"]
    #filenames = ["0001.jpg", "0002.jpg", "0003.jpg"]
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Error al cargar la imagen: {filename}. Saltando.")
            continue
        new_size = (new_width, int(img.shape[0] * new_width / img.shape[1]))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    return images

def segmentacion_pose(img_array: np.ndarray) -> np.ndarray:
    torch.cuda.empty_cache()

    world_model = YOLOWorld('yolov8m-world.pt') #Modelo de detección
    world_model.set_classes(["person", "clothes", "underwear"])
    world_model.to('cuda')
    with torch.cuda.amp.autocast():
        world_results = world_model(img_array, imgsz=640)
    del world_model
    gc.collect()
    torch.cuda.empty_cache()

    sam_model = SAM('sam2.1_b.pt') #Modelo de segmentación
    sam_model.to('cuda')
    mask_total = np.zeros(img_array.shape[:2], dtype=np.uint8)

    for r in world_results:
        if r.boxes is not None:
            for box in r.boxes: # Con las bounding boxes de input se aplica el algoritmo de segmentación
                bbox = box.xyxy[0].cpu().numpy()
                with torch.cuda.amp.autocast():
                    sam_results = sam_model(
                        source=img_array,
                        bboxes=[bbox.tolist()],
                        save=False,
                        conf=0.4
                    )
                mask = sam_results[0].masks.data.cpu().numpy()[0]
                mask_bool = mask.astype(bool)
                filled_mask = binary_fill_holes(mask_bool) # Se aplica el llenado de agujeros para obtener una máscara completa
                filled_mask_uint8 = filled_mask.astype(np.uint8) * 255
                mask_total = np.maximum(mask_total, filled_mask_uint8)

    sam_model.cpu()
    del sam_model
    gc.collect()
    torch.cuda.empty_cache()

    alpha_mask = mask_total
    img_with_alpha = np.dstack((img_array, alpha_mask))

    return img_with_alpha


def load_points_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        points = [tuple(map(int, line.split())) for line in f]
    return points

def process_image_with_points(image_array, txt_path, model_name="zoedepth", mask_image=None, output_ply_path="output.ply"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()

    # Cargar puntos 2D desde archivo
    points_2d = load_points_from_txt(txt_path)

    # Convertir imagen a formato PIL
    img = Image.fromarray(image_array).convert("RGB")
    img_np = np.array(img)[:, :, ::-1]  # Convertir de BGR (OpenCV) a RGB
    H, W = img_np.shape[:2]

    # Preparar máscara
    if mask_image is not None:
        mask_img = mask_image.convert("RGBA")
        if mask_img.size != img.size:
            mask_img = mask_img.resize(img.size, Image.NEAREST)
        _, _, _, a = mask_img.split()
        mask_np = np.array(a)
        mask = (mask_np > 0)
    else:
        mask = np.ones((H, W), dtype=bool)

    print(f"Imagen: {H}x{W}")
    print(f"Máscara: {mask.shape}, puntos válidos: {np.sum(mask)}")

    with torch.no_grad():
        depth = model.infer_pil(img)

    ### Conversión de profundidad a puntos 3D
    points_3d = depth_to_points(depth[None])

    points_all = points_3d.reshape(-1, 3)
    colors_all = img_np.reshape(-1, 3) / 255.0
    mask_flat = mask.flatten()

    ### Filtrado de puntos por máscara
    points_filtered = points_all[mask_flat]
    colors_filtered = colors_all[mask_flat]

    print(f"Puntos totales: {points_all.shape[0]}")
    print(f"Puntos filtrados: {points_filtered.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_filtered.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered.astype(np.float32))

    # Guardar nube de puntos como archivo PLY
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"Nube de puntos guardada en {output_ply_path}")

    highlighted_points = []
    for x, y in points_2d:
        if 0 <= x < W and 0 <= y < H:
            idx = y * W + x
            if mask_flat[idx]:
                highlighted_points.append(points_all[idx])

    if highlighted_points:
        highlighted_pcd = o3d.geometry.PointCloud()
        highlighted_pcd.points = o3d.utility.Vector3dVector(np.array(highlighted_points))
        highlighted_colors = np.array([[1, 0, 0] for _ in highlighted_points])
        highlighted_pcd.colors = o3d.utility.Vector3dVector(highlighted_colors)
        highlighted_ply_path = output_ply_path.replace(".ply", "_highlighted.ply")
        o3d.io.write_point_cloud(highlighted_ply_path, highlighted_pcd)
        print(f"Puntos destacados guardados en {highlighted_ply_path}")
    return pcd, highlighted_pcd


folder        = "./input/138/1/original"
new_width     = 6000
images_affine = load_and_resize_images(folder, new_width)
print(images_affine[0].shape)

##Stitcher con metodo Affine
settings = {"confidence_threshold": 0.4}
stitcher_affine = AffineStitcher(**settings)

images_rotAff = []
for image in images_affine:
    rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) #Se rotan las imágenes para obtener stitching vertical
    images_rotAff.append(rotated)
panorama_affine = stitcher_affine.stitch(images_rotAff)

panorama_affine = cv2.rotate(panorama_affine, cv2.ROTATE_90_CLOCKWISE ) #Se devuelve la imagen completa a la posición vertical
print(panorama_affine.shape)

img_with_alpha = segmentacion_pose(panorama_affine)
img_with_alpha_pil = Image.fromarray(img_with_alpha, mode='RGBA')

pcd, highlighted_pcd = process_image_with_points(panorama_affine,"./puntos_seleccionados.txt", "zoedepth", img_with_alpha_pil) #

