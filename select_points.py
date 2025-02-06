import cv2
import os

def select_points(image_path, output_txt_path):
    """
    Permite seleccionar puntos en una imagen y guarda las coordenadas en un archivo de texto.

    Args:
        image_path (str): Ruta de la imagen local.
        output_txt_path (str): Ruta donde se guardarán los puntos seleccionados.

    Returns:
        None
    """
    # Cargar imagen
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"No se encontró la imagen en {image_path}.")

    height, width = original_img.shape[:2]
    max_dimension = 800
    scale = min(max_dimension / width, max_dimension / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Redimensionamiento de la imagen
    img = cv2.resize(original_img, (new_width, new_height))
    points = []
    scale_factor = (width / new_width, height / new_height)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(x * scale_factor[0])
            original_y = int(y * scale_factor[1])
            points.append((original_x, original_y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Image', img)

    # Crear una ventana para mostrar la imagen
    cv2.namedWindow('Image')
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mouse_callback)

    print("Selecciona los puntos en la imagen. Presiona 'q' para finalizar.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Guardar los puntos en un archivo de texto
    with open(output_txt_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]}\n")

    print(f"Puntos guardados en {output_txt_path}")


if __name__ == "__main__":
    # Ruta de la imagen local
    image_path = "./resultado_opencv/138/138_stitch_affine.jpg"
    # Ruta para guardar el archivo de texto con los puntos
    output_txt_path = "puntos_seleccionados.txt"

    if os.path.exists(image_path):
        select_points(image_path, output_txt_path)
    else:
        print(f"La imagen no existe en la ruta especificada: {image_path}")