from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import os
import torch
import gc
from PIL import Image
from stitching import AffineStitcher
from ultralytics import YOLOWorld, SAM
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    world_model = YOLOWorld('yolov8m-world.pt')
    world_model.set_classes(["person"])
    world_model.to(DEVICE).eval()
    sam_model = SAM('sam2.1_b.pt')
    sam_model.to(DEVICE).eval()
    depth_model = build_model(get_config("zoedepth", "infer")).to(DEVICE).eval()
except Exception as e:
    exit(1)

def load_and_resize_images(folder, new_width):
    images = []
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        filepath = os.path.join(folder, filename)
        img = cv2.imread(filepath)
        if img is None:
            continue
        new_size = (new_width, int(img.shape[0] * new_width / img.shape[1]))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    return images

def pose_seg(img_array):
    torch.cuda.empty_cache()
    with torch.no_grad():
        world_results = world_model(img_array, imgsz=640)
    mask_total = np.zeros(img_array.shape[:2], dtype=np.uint8)
    for r in world_results:
        if r.boxes is not None:
            for box in r.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                with torch.no_grad():
                    sam_results = sam_model(source=img_array, bboxes=[bbox.tolist()], save=False, conf=0.4)
                mask = sam_results[0].masks.data.cpu().numpy()[0]
                mask_total = np.maximum(mask_total, (mask > 0).astype(np.uint8) * 255)
    return mask_total

@app.route('/process', methods=['POST'])
def process_images():
    try:
        data = request.json
        folder = data['folder']
        new_width = data.get('new_width', 3000)
        point_density = data.get('point_density', 1.0)
        images_affine = load_and_resize_images(folder, new_width)

        if not images_affine:
            return jsonify({"error": "No se encontraron imagenes en la carpeta"}), 400
        settings = {"confidence_threshold": 0.4}
        stitcher_affine = AffineStitcher(**settings)
        images_rotAff = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in images_affine]
        panorama_affine = stitcher_affine.stitch(images_rotAff)

        if panorama_affine is None:
            return jsonify({"error": "No se pudo generar la imagen panorámica"}), 400
        panorama_affine = cv2.rotate(panorama_affine, cv2.ROTATE_90_CLOCKWISE)
        mask_alpha = pose_seg(panorama_affine)
        with torch.no_grad():
            depth = depth_model.infer_pil(Image.fromarray(panorama_affine))

        points_3d = depth_to_points(depth[None]).reshape(-1, 3)
        depth_values = points_3d[:, 2]

        depth_threshold = np.percentile(depth_values, 20)
        valid_depth_mask = depth_values <= depth_threshold
        mask_flat = (mask_alpha.flatten() > 0) & valid_depth_mask
        points_3d = points_3d[mask_flat]

        if point_density < 1.0:
            indices = np.random.choice(len(points_3d), int(len(points_3d) * point_density), replace=False)
            points_3d = points_3d[indices]

        img_rgb = cv2.cvtColor(panorama_affine, cv2.COLOR_BGR2RGB)
        img_rgb_flat = img_rgb.reshape(-1, 3)
        colors = img_rgb_flat[mask_flat] / 255.0
        if point_density < 1.0:
            colors = colors[indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        output_ply_path = "output.ply"

        o3d.io.write_point_cloud(output_ply_path, pcd)
        return send_file(output_ply_path, as_attachment=True)
    
    except FileNotFoundError as e:
        return jsonify({"error": f"Archivo no encontrado: {str(e)}"}), 404
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return jsonify({"error": "Falta de memoria en GPU"}), 500
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
