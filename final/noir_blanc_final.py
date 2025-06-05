import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Charger l'image RGBA ===
img_rgba = cv2.imread("puzzle_transparent.png", cv2.IMREAD_UNCHANGED)

if img_rgba is None or img_rgba.shape[2] != 4:
    raise ValueError("L'image doit avoir un canal alpha (RGBA).")

# === 2. Extraire le canal alpha ===
alpha = img_rgba[:, :, 3]

# === 3. Binariser le masque : pièces = 255, fond = 0 ===
_, binary_mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)

# === 4. Méthode 1 : Flou + seuillage (ACTIVE PAR DÉFAUT) ===
blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)
_, smoothed_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# === 5. Méthode 2 : Morphologie (décommente pour tester) ===
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
#smoothed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)

# === 6. Méthode 3 : Contours et remplissage (décommente pour tester) ===
#contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#smoothed_mask = np.zeros_like(binary_mask)
#cv2.drawContours(smoothed_mask, contours, -1, 255, thickness=cv2.FILLED)

# === 7. Affichage ===
plt.figure(figsize=(8, 6))
plt.imshow(smoothed_mask, cmap='gray')
plt.axis("off")
plt.title("Masque binaire avec bords lissés")
plt.show()

# === 8. Sauvegarde ===
cv2.imwrite("puzzle_masque_lisse.png", smoothed_mask)
print("✅ Masque lissé enregistré sous : puzzle_masque_lisse.png")
