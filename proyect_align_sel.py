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
    """Initialize depth estimation model"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    conf = get_config(model_name, "infer")
    model = build_model(conf).to(DEVICE)
    model.eval()
    return model

def resize_image(image, max_size=(640, 480)):
    """Resize image maintaining aspect ratio"""
    image.thumbnail(max_size, Image.LANCZOS)
    return image

def load_and_process_images(image_path1, mask_path1, image_path2, mask_path2, target_size=(640, 480)):
    """Load and process images and masks"""
    
    img1 = Image.open(image_path1).convert("RGB")
    img1 = resize_image(img1, target_size)
    mask_img1 = Image.open(mask_path1).convert("RGBA")
    mask_img1 = resize_image(mask_img1, target_size)
  
    img2 = Image.open(image_path2).convert("RGB")
    img2 = resize_image(img2, target_size)
    mask_img2 = Image.open(mask_path2).convert("RGBA")
    mask_img2 = resize_image(mask_img2, target_size)
    
    return img1, mask_img1, img2, mask_img2

def get_depth_maps(model, img1, img2):
    """Get depth maps for both images"""
    with torch.no_grad():
        depth1 = model.infer_pil(img1)
        depth2 = model.infer_pil(img2)
    return depth1, depth2

def erode_alpha_mask(alpha_channel, kernel_size=5, iterations=1):
    """Apply erosion to alpha mask"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(alpha_channel, kernel, iterations=iterations)

def process_masks(mask_img1, mask_img2):

    mask_np1 = np.array(mask_img1)
    mask_np2 = np.array(mask_img2)
    
    alpha_channel1 = mask_np1[:, :, 3]
    alpha_channel2 = mask_np2[:, :, 3]
    
    alpha_channel1_eroded = erode_alpha_mask(alpha_channel1, kernel_size=3, iterations=2)
    alpha_channel2_eroded = erode_alpha_mask(alpha_channel2, kernel_size=3, iterations=2)
    
    return alpha_channel1_eroded, alpha_channel2_eroded

def create_point_clouds(points1, points2, colors1, colors2, voxel_size=0.02):
    """Create and process point clouds"""
   
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    
    pcd1 = pcd1.voxel_down_sample(voxel_size=voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size=voxel_size)
    
    return pcd1, pcd2

def register_point_clouds(pcd1, pcd2, voxel_size):
    """Register point clouds using RANSAC and ICP"""
    
    pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    radius_feature = voxel_size * 10
    pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
   
    distance_threshold = voxel_size * 2
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1, pcd2, pcd1_fpfh, pcd2_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    
    return result_ransac.transformation, result_icp.transformation

def setup_visualization():
    """Setup visualization window"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])
    
    return vis

def process_two_images(image_path1, mask_path1, image_path2, mask_path2, output_dir, 
                      model_name="zoedepth", max_depth=10.0, voxel_size=0.02, target_size=(640, 480)):
  
    model = init_model(model_name)
  
    img1, mask_img1, img2, mask_img2 = load_and_process_images(
        image_path1, mask_path1, image_path2, mask_path2, target_size)
  
    depth1, depth2 = get_depth_maps(model, img1, img2)
    
    alpha_mask1, alpha_mask2 = process_masks(mask_img1, mask_img2)
 
    img_np1 = np.array(img1)
    img_np2 = np.array(img2)
    
    points1 = depth_to_points(depth1[None])
    points2 = depth_to_points(depth2[None])
    
    alpha_mask1_flat = alpha_mask1.reshape(-1) > 128
    alpha_mask2_flat = alpha_mask2.reshape(-1) > 128
    
    points1_filtered = points1.reshape(-1, 3)[alpha_mask1_flat]
    points2_filtered = points2.reshape(-1, 3)[alpha_mask2_flat]
    
    colors1 = (img_np1.reshape(-1, 3)[alpha_mask1_flat] / 255.0).astype(np.float32)
    colors2 = (img_np2.reshape(-1, 3)[alpha_mask2_flat] / 255.0).astype(np.float32)
    
   
    pcd1, pcd2 = create_point_clouds(points1_filtered, points2_filtered, colors1, colors2, voxel_size)
  
    T_ransac, T_icp = register_point_clouds(pcd1, pcd2, voxel_size)
    
    pcd2.transform(T_ransac)
    pcd2.transform(T_icp)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cloud1.ply"), pcd1)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cloud2.ply"), pcd2)
        pcd_combined = pcd1 + pcd2
        o3d.io.write_point_cloud(os.path.join(output_dir, "combined.ply"), pcd_combined)
    
    vis = setup_visualization()
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
  
    vis.run()
    vis.destroy_window()
    
    return T_ransac, T_icp

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Processing and Alignment')
    parser.add_argument('--image1', '-i1', required=True, help='First input image')
    parser.add_argument('--mask1', '-m1', required=True, help='First image mask')
    parser.add_argument('--image2', '-i2', required=True, help='Second input image')
    parser.add_argument('--mask2', '-m2', required=True, help='Second image mask')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--model', '-m', default='zoedepth', help='Model name')
    parser.add_argument('--voxel-size', type=float, default=0.02, help='Voxel size for downsampling')
    parser.add_argument('--width', type=int, default=640, help='Target image width')
    parser.add_argument('--height', type=int, default=480, help='Target image height')
    
    args = parser.parse_args()
    
    target_size = (args.width, args.height)
    
    T_ransac, T_icp = process_two_images(
        args.image1, args.mask1, args.image2, args.mask2,
        args.output, args.model, voxel_size=args.voxel_size,
        target_size=target_size
    )
    
    print("RANSAC Transformation:")
    print(T_ransac)
    print("\nICP Refinement:")
    print(T_icp)

if __name__ == "__main__":
    main()