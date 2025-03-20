import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image_path = "pieces_convert/puzzle_contour.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Appliquer un seuillage pour binariser l'image
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Trouver les contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer une image de même taille remplie de noir
filled_image = np.zeros_like(binary)

# Remplir les contours détectés
cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

# Afficher le résultat
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Contour détecté")
plt.imshow(binary, cmap='gray')

plt.subplot(1,2,2)
plt.title("Contour rempli")
plt.imshow(filled_image, cmap='gray')
plt.show()

cv2.imwrite("pieces_convert/puzzle_contour_remplie.jpg", filled_image)
