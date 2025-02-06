import open3d as o3d
import numpy as np

def visualize_point_cloud_with_highlights(ply_file_path, highlighted_ply_path=None):
    
    # Cargar la nube de puntos principal
    pcd = o3d.io.read_point_cloud(ply_file_path)

    if pcd.is_empty():
        print(f"La nube de puntos en {ply_file_path} está vacía o no se pudo cargar.")
        return

    geometries = [pcd]
    """
    # Cargar los puntos destacados 
    if highlighted_ply_path:
        highlighted_pcd = o3d.io.read_point_cloud(highlighted_ply_path)
        if highlighted_pcd.is_empty():
            print(f"El archivo de puntos destacados en {highlighted_ply_path} está vacío o no se pudo cargar.")
        else:
            # Visualización puntos destacados
            for point in np.asarray(highlighted_pcd.points):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.translate(point)
                sphere.paint_uniform_color([1, 0, 0])
                geometries.append(sphere)
    """
    print("Presiona 'q' para salir de la visualización.")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    ply_file_path = "./output_139.ply"  # Cambiar al path donde se encuentran los archivos .ply
    highlighted_ply_path = "../../Descargas/output_highlighted.ply"

    visualize_point_cloud_with_highlights(ply_file_path, highlighted_ply_path)