import cv2
import numpy as np
import os

# === 1. Chemins des fichiers ===
mask_path = "puzzle_masque_lisse.png"
transparent_img_path = "puzzle_transparent.png"

# === 2. Charger les images ===
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread(transparent_img_path, cv2.IMREAD_UNCHANGED)

if mask is None or original_image is None:
    raise ValueError("Le masque ou l'image d'origine n'a pas été trouvé.")

# === 3. Créer les dossiers de sortie ===
output_dir_rgba = "piece"
output_dir_bw = "piece_noir_blanc"
os.makedirs(output_dir_rgba, exist_ok=True)
os.makedirs(output_dir_bw, exist_ok=True)

# === 4. Détecter les contours des pièces ===
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === 5. Extraire et sauvegarder chaque pièce ===
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    # Filtrer les petits bruits
    if w * h < 100:
        continue

    # Extraire la sous-image RGBA
    piece = original_image[y:y + h, x:x + w]

    # Créer un masque local pour cette pièce
    local_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(local_mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)

    # ➤ Version 1 : Sauvegarde en RGBA avec transparence
    piece_rgba = piece.copy()
    piece_rgba[:, :, 3] = local_mask  # Remplace le canal alpha
    rgba_path = os.path.join(output_dir_rgba, f"piece_{i}.png")
    cv2.imwrite(rgba_path, piece_rgba)

    # ➤ Version 2 : Sauvegarde en noir et blanc (255 = pièce, 0 = fond)
    bw_path = os.path.join(output_dir_bw, f"piece_{i}.png")
    cv2.imwrite(bw_path, local_mask)

    print(f"✅ Sauvegardé : {rgba_path} et {bw_path}")
