import cv2
import os
import argparse
from pathlib import Path

def main(input_dir, output_dir, rotate_angle):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cv2.ocl.setUseOpenCL(False)
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    images = []
    images_rotated = []
    target_names = ["0002", "0003", "0004"]

    for img_file in sorted(Path(input_dir).glob("*.jpg")):
        if not any(target in img_file.name for target in target_names):
            continue
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen {img_file}")
            continue
        images.append(img)

    # for idx, img in enumerate(images):
    #     rotated = cv2.rotate(img, rotate_angle)
    #     images_rotated.append(rotated)
    #     output_path = os.path.join(output_dir, f"rotated_{idx + 1}.jpg")
    #     cv2.imwrite(output_path, rotated)

    print("Imágenes rotadas guardadas en:", output_dir)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        #pano = cv2.rotate(pano, cv2.ROTATE_90_CLOCKWISE)
        pano_output_path = os.path.join(output_dir, "panorama.jpg")
        cv2.imwrite(pano_output_path, pano)
        print(f"Imagen panoramica guardada en: {pano_output_path}")
    else:
        print(f"No se pudieron coser las imagenes, codigo de error: {status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar imagenes para hacer stitching")
    parser.add_argument("--input_dir", type=str, required=True, help="Directorio de entrada con las imágenes.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directorio de salida para guardar las imágenes procesadas.")
    parser.add_argument("--angle", type=str, default="ccw", choices=["cw", "ccw"], help="Ángulo de rotación: 'cw' (90° horario) o 'ccw' (90° antihorario).")
    args = parser.parse_args()
    rotate_map = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}
    rotate_angle = rotate_map[args.angle]
    main(args.input_dir, args.output_dir, rotate_angle)
