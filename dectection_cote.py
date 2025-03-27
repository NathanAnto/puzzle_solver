import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Créer le dossier de sortie s'il n'existe pas
input_folder = "detected_pieces"
output_folder = "pieces_contour"
os.makedirs(output_folder, exist_ok=True)

# Lister tous les fichiers d'image dans le dossier d'entrée
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Impossible de lire l'image : {filename}")
            continue

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5), np.uint8)

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

        # Sauvegarder l'image résultante
        output_path = os.path.join(output_folder, f"contour_{filename}")
        cv2.imwrite(output_path, mask)

        print(f"Image traitée et sauvegardée : {output_path}")
