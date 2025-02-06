from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import torch
import gc
from PIL import Image
from io import BytesIO
from minio import Minio
from stitching import AffineStitcher
from ultralytics import YOLOWorld, SAM
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
import open3d as o3d

app = Flask(__name__)

# Configuración de MinIO
MINIO_ENDPOINT = "localhost:9000"
ACCESS_KEY = "qELvkUD36q7oEQZI7Jv9"
SECRET_KEY = "OAHzCwRnwWBmX22rycb8CC8AcHmoZbYlWnJF2vaS"
BUCKET_NAME = "api-test"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    world_model = YOLOWorld('yolov8m-world.pt')
    world_model.set_classes(["person"])
    world_model.to(DEVICE).eval()
    
    sam_model = SAM('sam2.1_b.pt')
    sam_model.to(DEVICE).eval()
    
    depth_model = build_model(get_config("zoedepth", "infer")).to(DEVICE).eval()
    print("Listo para recibir solicitudes :)")
except Exception as e:
    exit(1)

def load_images_from_minio(folder_prefix, new_width):
    images = []
    
    objects = minio_client.list_objects(BUCKET_NAME, prefix=folder_prefix, recursive=True)
    filenames = sorted([obj.object_name for obj in objects if obj.object_name.endswith(('.jpg', '.png', '.jpeg'))])

    for filename in filenames:
        response = minio_client.get_object(BUCKET_NAME, filename)
        img_array = np.frombuffer(response.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            continue
        new_size = (new_width, int(img.shape[0] * new_width / img.shape[1]))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    
    return images

def upload_to_minio(file_path, minio_path):
    minio_client.fput_object(BUCKET_NAME, minio_path, file_path)

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

def save_obj_with_mtl(obj_path, pcd):
    vertices = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    """
    with open(mtl_path, 'w') as mtl_file:
        mtl_file.write("newmtl material_0\n")
        mtl_file.write("Ka 0.2 0.2 0.2\n")
        mtl_file.write("Kd 1.0 1.0 1.0\n")
        mtl_file.write("Ks 0.0 0.0 0.0\n")
        mtl_file.write("illum 1\n")
    """
    luminosity = 2
    with open(obj_path, 'w') as obj_file:
        #obj_file.write("mtllib " + os.path.basename(mtl_path) + "\n")
        #obj_file.write("usemtl material_0\n")
        for v, c in zip(vertices, colors):
            obj_file.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {min(c[0]*luminosity,1):.3f} {min(c[1]*luminosity,1):.3f} {min(c[2]*luminosity,1):.3f}\n")

@app.route('/process', methods=['POST'])
def process_images():
    try:
        data = request.json
        folder_prefix = data['folder_prefix']
        new_width = data.get('new_width', 3000)
        point_density = data.get('point_density', 1.0)

        images_affine = load_images_from_minio(folder_prefix, new_width)
        if not images_affine:
            return jsonify({"error": "No se encontraron imagenes en el bucket"}), 400
        
        settings = {"confidence_threshold": 0.4}
        stitcher_affine = AffineStitcher(**settings)
        images_rotAff = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in images_affine]
        panorama_affine = stitcher_affine.stitch(images_rotAff)
        
        if panorama_affine is None:
            return jsonify({"error": "No se pudo generar la imagen panoramica"}), 400
        
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
        
        output_obj_path = "/tmp/output.obj"
        
        save_obj_with_mtl(output_obj_path, pcd)
        
        upload_to_minio(output_obj_path, f"{folder_prefix}/output.obj")
        #upload_to_minio(output_mtl_path, f"{folder_prefix}/output.mtl")

        return jsonify({"message": "Archivos subidos con exito", "file_url_obj": f"{folder_prefix}/output.obj"})
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500
    finally:
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
