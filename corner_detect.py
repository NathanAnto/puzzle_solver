import cv2
import numpy as np

# Charger l'image
image_path = "pieces_convert/puzzle_contour_remplie.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Trouver les contours
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_puzzle = max(contours, key=cv2.contourArea)  # Prendre le plus grand contour

# Convertir en format couleur pour annotation
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Appliquer la détection des coins de Harris
dst = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)

# Dilater pour mieux voir les coins détectés
dst = cv2.dilate(dst, None)

# Seuil pour conserver les meilleurs coins
image_color[dst > 0.01 * dst.max()] = [0, 0, 0]  # Rouge pour les coins

# Afficher le résultat
'''
cv2.imshow("Coins détectés", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Détection des contours pour la transformée de Hough
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Trouver les lignes avec HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=10, maxLineGap=500000)

# Dessiner les lignes détectées

image_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert pour les lignes
'''
# Afficher l’image avec les lignes détectées
cv2.imshow("Lignes détectées (Hough)", image_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Filtrer les lignes pour obtenir les 4 meilleures
horizontal_lines = []
vertical_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < abs(x2 - x1):  # Ligne horizontale
        horizontal_lines.append(line[0])
    else:  # Ligne verticale
        vertical_lines.append(line[0])

# Trier les lignes pour prendre les plus extrêmes
horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])  # Haut vers Bas
vertical_lines = sorted(vertical_lines, key=lambda x: x[0])  # Gauche vers Droite

# Sélectionner les 2 meilleures lignes horizontales et verticales
top_line = horizontal_lines[0]
bottom_line = horizontal_lines[-1]
left_line = vertical_lines[0]
right_line = vertical_lines[-1]

# Dessiner sur l’image
cv2.line(image_color, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (255, 0, 255), 2)  # Bleu
cv2.line(image_color, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (255, 0, 0), 2)  # Bleu
cv2.line(image_color, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 2)  # Vert
cv2.line(image_color, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 2)  # Vert

cv2.imshow("Côtés détectés", image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
