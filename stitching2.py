import cv2
import os
import imutils
import numpy as np

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)


    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False, minKeypoints=4):
        (imageB, imageA) = images

        # Detectar y describir puntos clave
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # Obtener dimensiones de las imágenes
        heightA, widthA = imageA.shape[:2]
        heightB, widthB = imageB.shape[:2]

        # Calcular los límites para cada imagen
        centerA_x = widthA // 2
        centerB_x = widthB // 2
        
        # Filtrar puntos clave y características dentro de los límites (2000 píxeles desde el centro)
        valid_indices_A = []
        valid_indices_B = []
        
        # Filtrar puntos de imageA
        for i, kp in enumerate(kpsA):
            x = kp[0]
            if centerA_x - 2000 <= x <= centerA_x + 2000:
                valid_indices_A.append(i)
                
        # Filtrar puntos de imageB
        for i, kp in enumerate(kpsB):
            x = kp[0]
            if centerB_x - 2000 <= x <= centerB_x + 2000:
                valid_indices_B.append(i)
        
        # Actualizar los puntos clave y características
        kpsA = kpsA[valid_indices_A]
        featuresA = featuresA[valid_indices_A]
        kpsB = kpsB[valid_indices_B]
        featuresB = featuresB[valid_indices_B]

        if len(kpsA) < minKeypoints or len(kpsB) < minKeypoints:
            print("No se detectaron suficientes puntos clave.")
            return None

        # Emparejar puntos clave
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            print("No se encontraron suficientes coincidencias para unir las imágenes.")
            return None

        (matches, H, status) = M

        # Transformar perspectiva
        result_width = max(widthA + widthB, widthA * 2)
        result_height = max(heightA + heightB, heightA * 2)

        result = cv2.warpPerspective(imageA, H, (result_width, result_height))

        # Superponer imagenB en el resultado
        y_offset, x_offset = heightB, widthB
        result[0:y_offset, 0:x_offset] = imageB

        # Recorte automático para eliminar áreas negras
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        result = result[y:y+h, x:x+w]

        # Mostrar coincidencias opcionalmente
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        return result


    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)

        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis
    
def load_and_resize_images(folder, new_width):
    images = []
    filenames = ["0002.jpg", "0003.jpg", "0004.jpg"]
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

def crop_black_borders(image):
    # Convertir la imagen a escala de grises y crear una mascara binaria
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # Si no hay contornos, devuelve la imagen original

    x, y, w, h = cv2.boundingRect(contours[0])
    for contour in contours:
        x_, y_, w_, h_ = cv2.boundingRect(contour)
        x = min(x, x_)
        y = min(y, y_)
        w = max(w, x_ + w_) - x
        h = max(h, y_ + h_) - y

    cropped = image[y:y+h, x:x+w]
    return cropped

def main():
    folder = "./input/147/1/original"
    new_width = 1024
    images = load_and_resize_images(folder, new_width)
    if len(images) < 2:
        print("No hay suficientes imágenes para unir. Saliendo.")
        return
    stitcher = Stitcher()
    (result12, vis12) = stitcher.stitch([images[1], images[2]], showMatches=True)
    cv2.imshow('vis first match', vis12)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (result123, vis123) = stitcher.stitch([images[0], result12 ], showMatches=True)
    cv2.imshow('vis second match', vis123)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if result123 is not None:
        result_cropped = crop_black_borders(result123)
        result_cropped2 = result_cropped[:, :images[0].shape[1]]
        cv2.imwrite("./stitched_zoe/147_1_ori.jpg", result_cropped2)

if __name__ == "__main__":
    main()
