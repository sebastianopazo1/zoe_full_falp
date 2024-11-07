import gradio as gr
import numpy as np
import trimesh
from zoedepth.utils.geometry import depth_to_points, create_triangles
from functools import partial
import tempfile

def depth_edges_mask(depth):
    depth_dx, depth_dy = np.gradient(depth)
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    mask = depth_grad > 0.05
    return mask

def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def get_mesh(model, image, keep_edges=False):
    image.thumbnail((1024,1024))
    depth = predict_depth(model, image)
    pts3d = depth_to_points(depth[None])
    pts3d = pts3d.reshape(-1, 3)
    verts = pts3d.reshape(-1, 3)
    image = np.array(image)
    if keep_edges:
        triangles = create_triangles(image.shape[0], image.shape[1])
    else:
        triangles = create_triangles(image.shape[0], image.shape[1], mask=~depth_edges_mask(depth))
    colors = image.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=verts, faces=triangles, vertex_colors=colors)
    glb_file = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
    glb_path = glb_file.name
    mesh.export(glb_path)
    return glb_path

def create_demo(model):
    with gr.Blocks() as demo:  
        with gr.Row():
            image = gr.Image(label="Input Image", type='pil')
            result = gr.Model3D(label="3d mesh reconstruction", clear_color=[1.0, 1.0, 1.0, 1.0])
        checkbox = gr.Checkbox(label="Keep occlusion edges", value=False)
        submit = gr.Button("Submit")
        submit.click(partial(get_mesh, model), inputs=[image, checkbox], outputs=[result])
        return demo

if __name__ == "__main__":
    import torch
    
    model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    demo = create_demo(model)
    demo.launch()