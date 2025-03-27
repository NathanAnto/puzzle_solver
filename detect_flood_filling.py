import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Dossiers
input_folder = "pieces_contour"
output_folder = "pieces_remplie"
os.makedirs(output_folder, exist_ok=True)

# Lister tous les fichiers d'image dans le dossier d'entrée
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Impossible de lire l'image : {filename}")
            continue

        # Appliquer un seuillage pour binariser l'image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # Trouver les contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Créer une image de même taille remplie de noir
        filled_image = np.zeros_like(binary)

        # Remplir les contours détectés
        cv2.drawContours(filled_image, contours, -1, (255), thickness=cv2.FILLED)

        # Sauvegarder l'image résultante
        output_path = os.path.join(output_folder, f"remplie_{filename}")
        cv2.imwrite(output_path, filled_image)

        print(f"Image remplie sauvegardée : {output_path}")
