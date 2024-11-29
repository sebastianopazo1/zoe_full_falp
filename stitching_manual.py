import cv2
import numpy as np

# Configurar la superposición (en píxeles)
superposicion = 2900  # Puedes modificar este valor para probar diferentes superposiciones

# Leer las imágenes
img1 = cv2.imread('./input/147/1/original/0002.jpg')
img2 = cv2.imread('./input/147/1/original/0003.jpg')
img3 = cv2.imread('./input/147/1/original/0004.jpg')

def unir_imagenes_superior(img_superior, img_inferior, superposicion):
    #asegurar que las imagenes tienen el mismo alto
    if img_superior.shape[1] != img_inferior.shape[1]:
        ancho = min(img_superior.shape[1], img_inferior.shape[1])
        img_superior = cv2.resize(img_superior, (ancho, img_superior.shape[0]))
        img_inferior = cv2.resize(img_inferior, (ancho, img_inferior.shape[0]))

    #se calcula la dimension de la nueva imagen
    altura_total = img_superior.shape[0] + img_inferior.shape[0] - superposicion
    ancho_total = img_superior.shape[1]
    #imagen en negro con las nuevas  dimensiones
    imagen_unida = np.zeros((altura_total, ancho_total, 3), dtype=np.uint8)

    imagen_unida[0:img_superior.shape[0] - superposicion, 0:ancho_total] = img_superior[0:img_superior.shape[0] - superposicion, 0:ancho_total]
    imagen_unida[img_superior.shape[0] - superposicion:img_superior.shape[0], 0:ancho_total] = img_superior[img_superior.shape[0] - superposicion:, 0:ancho_total]
    imagen_unida[img_superior.shape[0]:altura_total, 0:ancho_total] = img_inferior[superposicion:, 0:ancho_total]

    return imagen_unida

imagen_intermedia = unir_imagenes_superior(img2, img3, superposicion)
imagen_final = unir_imagenes_superior(img1, imagen_intermedia, superposicion)
cv2.imwrite('imagen_unida.jpg', imagen_final)

print("La imagen ha sido unida exitosamente y guardada como 'imagen_unida.jpg'.")
