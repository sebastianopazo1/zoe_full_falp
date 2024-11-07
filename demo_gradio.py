import gradio as gr
import numpy as np
import trimesh
from zoedepth.utils.geometry import depth_to_points, create_triangles
import tempfile
import cv2
from PIL import Image

def depth_edges_mask(depth):
    depth_dx, depth_dy = np.gradient(depth)
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    mask = depth_grad > 0.05
    return mask

def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def get_mesh_with_points(model, image, coords, keep_edges=False):
    image.thumbnail((1024,1024))
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)
    verts = pts3d.reshape(-1, 3)
    image_np = np.array(image)
    
    if keep_edges:
        triangles = create_triangles(image_np.shape[0], image_np.shape[1])
    else:
        triangles = create_triangles(image_np.shape[0], image_np.shape[1], mask=~depth_edges_mask(depth))
    
    colors = image_np.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)

    if coords is not None:
        x, y = coords
        idx = y * image_np.shape[1] + x
        point_3d = verts[idx]
        sphere = trimesh.primitives.Sphere(radius=0.05, center=point_3d)
        sphere.visual.face_colors = [255, 0, 0, 255]  # Rojo
        mesh = trimesh.util.concatenate([mesh, sphere])

    glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    glb_path = glb_file.name
    mesh.export(glb_path)
    return glb_path

def create_demo(model):
    def on_click(image, evt: gr.SelectData, keep_edges):
        coords = (evt.index[0], evt.index[1])
        # Dibujar punto en la imagen
        img_array = np.array(image)
        cv2.circle(img_array, (evt.index[0], evt.index[1]), 5, (255, 0, 0), -1)
        img_pil = Image.fromarray(img_array)
        
        # Generar mesh 3D con el punto seleccionado
        mesh_path = get_mesh_with_points(model, image, coords, keep_edges)
        return img_pil, mesh_path
    
    with gr.Blocks() as demo:
        with gr.Row():
            image = gr.Image(label="Input Image", type='pil')
            result = gr.Model3D(label="3d mesh reconstruction", clear_color=[1.0, 1.0, 1.0, 1.0])
        
        checkbox = gr.Checkbox(label="Keep occlusion edges", value=False)
        image.select(
            on_click,
            inputs=[image, checkbox],
            outputs=[image, result]
        )
        
        return demo

if __name__ == "__main__":
    import torch
    
    model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    demo = create_demo(model)
    demo.launch()