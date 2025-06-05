import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Charger l'image ===
image_path = "20250508_124648870_iOS.png"  # Remplace par ton image
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Image non trouvée. Vérifie le chemin.")

# === 2. Redimensionner (optionnel) ===
max_dim = 1000
h, w = img.shape[:2]
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# === 3. GrabCut pour extraire le premier plan ===
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)

# iterCount pour modifier le rendu
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

# === 4. Ajouter un canal alpha basé sur le masque ===
b, g, r = cv2.split(img)
alpha = (mask2 * 255).astype("uint8")
rgba = cv2.merge((b, g, r, alpha))

# === 5. Sauvegarder avec fond transparent ===
output_path = "puzzle_transparent.png"
cv2.imwrite(output_path, rgba)

# === 6. Afficher le résultat avec transparence sur fond blanc pour visualisation ===
background = np.ones_like(img, dtype=np.uint8) * 255  # fond blanc
img_with_alpha = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)

# Remplir zones transparentes avec blanc (pour affichage uniquement)
alpha_mask = alpha.astype(bool)
preview = img.copy()
preview[~alpha_mask] = [255, 255, 255]

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Prévisualisation avec fond blanc")
plt.show()

print(f"✅ Image PNG avec transparence enregistrée sous : {output_path}")
