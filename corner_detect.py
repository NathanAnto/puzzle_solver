import cv2
import numpy as np
import os


# ---------------------------
# Fonctions utilitaires
# ---------------------------
def line_intersection(line1, line2):
    """
    Calcule l'intersection de deux lignes (chacune définie par [x1, y1, x2, y2]).
    Retourne (x, y) ou None en cas de parallélisme.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (int(px), int(py))


def reorder_points_clockwise(pts):
    """
    Réordonne une liste de points (x,y) en sens horaire autour du centre.
    """
    pts = np.array(pts, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    idx_sorted = np.argsort(angles)
    return pts[idx_sorted]


def sample_patch(image, center, normal, distance, window_size=5):
    """
    Échantillonne une petite fenêtre centrée en center + distance*normal.
    Retourne la moyenne d'intensité du patch.
    """
    sp = np.array(center) + distance * np.array(normal)
    sp = sp.astype(int)
    h, w = image.shape[:2]
    half = window_size // 2
    x, y = sp[0], sp[1]
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half + 1)
    y2 = min(h, y + half + 1)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return 0
    return np.mean(patch)


def extract_edge_patch(image, pt1, pt2, patch_width, patch_height, direction):
    """
    Extrait un patch rectifié le long du segment [pt1, pt2].
    - patch_width est la largeur (souvent égale à la longueur du segment).
    - patch_height est la hauteur (base + extension si male/femelle).
    - direction (1 ou -1) détermine de quel côté de la droite extraire le patch.
    """
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None
    dir_x, dir_y = dx / length, dy / length
    # Normal de base (la fonction utilisera -dy, dx multiplié par direction)
    nx, ny = -dir_y * direction, dir_x * direction
    cx, cy = (pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0
    src_pts = np.float32([
        [cx - dir_x * patch_width / 2 - nx * patch_height / 2, cy - dir_y * patch_width / 2 - ny * patch_height / 2],
        [cx + dir_x * patch_width / 2 - nx * patch_height / 2, cy + dir_y * patch_width / 2 - ny * patch_height / 2],
        [cx - dir_x * patch_width / 2 + nx * patch_height / 2, cy - dir_y * patch_width / 2 + ny * patch_height / 2]
    ])
    dst_pts = np.float32([[0, 0], [patch_width, 0], [0, patch_height]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    patch = cv2.warpAffine(image, M, (patch_width, patch_height))
    return patch


# ---------------------------
# Paramètres et dossiers
# ---------------------------
input_folder = "pieces_remplie"
output_folder = "cotes_extraits"
os.makedirs(output_folder, exist_ok=True)

# Paramètres d'échantillonnage pour la classification
sample_distance = 10  # distance en pixels pour échantillonner le profil
window_size = 5  # taille de la fenêtre pour l'échantillonnage
white_thresh = 200  # seuil pour considérer qu'un patch est blanc (male)
black_thresh = 50  # seuil pour considérer qu'un patch est noir (femelle)

# Paramètres pour l'extraction du patch
base_patch_height = 60  # hauteur de base
extension_value = 20  # extension à ajouter en cas de male ou femelle

# ---------------------------
# Traitement de chaque image
# ---------------------------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(input_folder, filename)
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print(f"Erreur de lecture : {filename}")
        continue

    print(f"Traitement de : {filename}")
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    # --- Détection des lignes ---
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=10, maxLineGap=500000)
    if lines is None or len(lines) < 4:
        print("Pas de lignes détectées ou insuffisantes pour 4 côtés.")
        continue

    # Séparation en lignes horizontales et verticales
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < abs(x2 - x1):
            horizontal_lines.append(line[0])
        else:
            vertical_lines.append(line[0])
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        print("Pas assez de lignes horizontales ou verticales.")
        continue

    horizontal_lines = sorted(horizontal_lines, key=lambda l: l[1])
    vertical_lines = sorted(vertical_lines, key=lambda l: l[0])
    top_line = horizontal_lines[0]
    bottom_line = horizontal_lines[-1]
    left_line = vertical_lines[0]
    right_line = vertical_lines[-1]

    # Calcul des intersections pour obtenir les coins
    top_left = line_intersection(top_line, left_line)
    top_right = line_intersection(top_line, right_line)
    bottom_left = line_intersection(bottom_line, left_line)
    bottom_right = line_intersection(bottom_line, right_line)
    if None in [top_left, top_right, bottom_left, bottom_right]:
        print("Erreur lors du calcul des intersections.")
        continue
    corners = [top_left, top_right, bottom_right, bottom_left]
    corners = reorder_points_clockwise(corners)
    center = np.mean(corners, axis=0)
    center = (int(center[0]), int(center[1]))

    # (Optionnel) Visualisation des coins et des lignes
    vis = image_color.copy()
    cv2.circle(vis, top_left, 5, (0, 0, 255), -1)
    cv2.circle(vis, top_right, 5, (0, 0, 255), -1)
    cv2.circle(vis, bottom_right, 5, (0, 0, 255), -1)
    cv2.circle(vis, bottom_left, 5, (0, 0, 255), -1)
    cv2.line(vis, top_line[:2], top_line[2:], (255, 0, 255), 2)
    cv2.line(vis, bottom_line[:2], bottom_line[2:], (255, 0, 0), 2)
    cv2.line(vis, left_line[:2], left_line[2:], (0, 255, 255), 2)
    cv2.line(vis, right_line[:2], right_line[2:], (0, 255, 0), 2)
    cv2.imshow("Coins et lignes", vis)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Association des côtés du quadrilatère
    edges_dict = {
        "top": (corners[0], corners[1]),
        "right": (corners[1], corners[2]),
        "bottom": (corners[2], corners[3]),
        "left": (corners[3], corners[0])
    }

    # Pour chaque côté, échantillonne un profil et détermine la classification
    for side_name, (pt1, pt2) in edges_dict.items():
        # Calcul du segment et de son milieu
        mid = ((pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0)
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            continue
        # Calcul de la normale de base (avec (-dy, dx)) et normalisation
        base_normal = np.array([-dy, dx]) / length
        # Calcul du vecteur du milieu vers le centre
        vec_center = np.array(center) - np.array(mid)
        vec_center_norm = vec_center / (np.linalg.norm(vec_center) + 1e-6)
        # Détermination des normales intérieure et extérieure
        if np.dot(base_normal, vec_center_norm) > 0:
            interior_normal = base_normal
            exterior_normal = -base_normal
        else:
            interior_normal = -base_normal
            exterior_normal = base_normal

        # Échantillonnage de petits patches sur chaque côté de la ligne
        ext_mean = sample_patch(image_gray, mid, exterior_normal, sample_distance, window_size)
        int_mean = sample_patch(image_gray, mid, interior_normal, sample_distance, window_size)

        # Classification selon les moyennes (ajustez les seuils selon vos images)
        if ext_mean > white_thresh:
            classification = "male"
        elif int_mean < black_thresh:
            classification = "femelle"
        else:
            classification = "neutre"

        print(f"Côté {side_name}: ext_mean={ext_mean:.1f}, int_mean={int_mean:.1f} => {classification}")

        # Détermination de la hauteur du patch à extraire
        extra = extension_value if classification in ["male", "femelle"] else 0
        final_patch_height = base_patch_height + extra
        seg_length = int(length)  # largeur du patch égale à la longueur du segment

        # Choix de la direction d'extraction :
        # Utilisons la fonction get_patch_direction issue du code précédent.
        # Ici, on veut extraire vers l'extérieur pour "male" et vers l'intérieur pour "femelle".
        base_dir = 1 if np.dot(np.array([-dy, dx]), vec_center_norm) >= 0 else -1
        if classification == "male":
            extraction_direction = -base_dir
        elif classification == "femelle":
            extraction_direction = base_dir
        else:
            extraction_direction = base_dir

        patch = extract_edge_patch(image_color, pt1, pt2, seg_length, final_patch_height, extraction_direction)
        if patch is not None:
            outname = f"{os.path.splitext(filename)[0]}_{side_name}_{classification}.png"
            out_path = os.path.join(output_folder, outname)
            cv2.imwrite(out_path, patch)
            print(f"Patch '{side_name}' ({classification}) sauvegardé : {out_path}")
