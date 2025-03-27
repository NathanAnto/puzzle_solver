import cv2
import numpy as np
import os

input_folder = "cotes_extraits"
output_folder = "cotes_extraits"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    # On cherche les silhouettes (celles qui ne contiennent pas '_color' dans leur nom)
    if filename.lower().endswith('.png') and '_color' not in filename:
        silhouette_path = os.path.join(input_folder, filename)
        color_path = os.path.join(input_folder, filename.replace('.png', '_color.png'))

        # Vérifier que le fichier couleur correspondant existe
        if not os.path.exists(color_path):
            print(f"Fichier couleur introuvable pour {filename}")
            continue

        # Lecture de la silhouette (niveaux de gris)
        sil_gray = cv2.imread(silhouette_path, cv2.IMREAD_GRAYSCALE)
        if sil_gray is None:
            print(f"Impossible de lire la silhouette : {silhouette_path}")
            continue

        # Lecture de l'image couleur
        color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if color_img is None:
            print(f"Impossible de lire l'image couleur : {color_path}")
            continue

        # Création du masque binaire à partir de la silhouette
        # (Tout ce qui est blanc > 127 => 255, sinon 0)
        _, mask = cv2.threshold(sil_gray, 127, 255, cv2.THRESH_BINARY)

        # On applique le masque pour enlever la zone noire
        # => Les pixels où mask=0 deviendront noirs (ou transparents si on gère un canal alpha)
        color_masked = cv2.bitwise_and(color_img, color_img, mask=mask)

        # Enregistrement
        # Exemple : "remplie_contour_piece_0_bottom_male.png" -> "remplie_contour_piece_0_bottom_male_masked.png"
        out_name = filename.replace('.png', '_color.png')
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, color_masked)
        print(f"Export : {out_path}")
