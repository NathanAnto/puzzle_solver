import cv2
import numpy as np
from matplotlib import pyplot as plt

# Charger l'image
image_path = "detected_pieces/piece_6.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)


# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
#gray = cv2.dilate(gray, kernel, iterations=1)
#gray = cv2.erode(gray, kernel, iterations=1)

# Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Détection des contours avec Canny
edges = cv2.Canny(blurred, 50, 150)

# Trouver les contours externes
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Créer une image blanche de la même taille
mask = np.ones_like(gray) * 255  # Fond blanc

# Dessiner les contours en noir
cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=5)


# Afficher l'image résultante
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()

# Sauvegarder l'image résultante
cv2.imwrite("pieces_convert/puzzle_contour.jpg", mask)
