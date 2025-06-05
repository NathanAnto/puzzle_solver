import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import json


class PuzzlePieceAnalyzer:
    def __init__(self, input_dir="piece_noir_blanc", corners_json_path="corners_output/puzzle_corners.json"):
        self.input_dir = input_dir
        self.corners_json_path = corners_json_path
        self.piece_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
        self.pieces_data = []

        # Charger les coins depuis le fichier JSON
        self.corners_data = self.load_corners_from_json()

        if not self.piece_files:
            raise ValueError("Aucune image trouvée dans le dossier 'piece_noir_blanc'.")

    def load_corners_from_json(self):
        """Charge les coins depuis le fichier JSON"""
        try:
            with open(self.corners_json_path, 'r') as f:
                corners_json = json.load(f)

            corners_data = {}
            for piece_key, piece_info in corners_json.items():
                filename = piece_info['original_filename']
                corners = []

                # Convertir les coins en format numpy array
                for corner in piece_info['corners']:
                    corners.append([corner['x'], corner['y']])

                corners_data[filename] = np.array(corners, dtype=np.float32)

            print(f"Coins chargés pour {len(corners_data)} pièces depuis {self.corners_json_path}")
            return corners_data

        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier de coins non trouvé : {self.corners_json_path}")
        except Exception as e:
            raise Exception(f"Erreur lors du chargement des coins : {e}")

    def get_corners_for_piece(self, filename):
        """Récupère les coins pour une pièce donnée"""
        if filename in self.corners_data:
            return self.corners_data[filename]
        else:
            raise ValueError(f"Coins non trouvés pour la pièce {filename}")

    def detect_corners(self, contour):
        """Méthode obsolète - remplacée par get_corners_for_piece"""
        # Cette méthode est conservée pour compatibilité mais ne sera plus utilisée
        pts = contour.reshape(-1, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def extract_side_contour(self, contour, corner1, corner2, side_name):
        """Extrait la portion du contour entre deux coins pour un côté spécifique"""
        pts = contour.reshape(-1, 2)

        # Trouver les indices des coins dans le contour
        idx1 = np.argmin(np.linalg.norm(pts - corner1, axis=1))
        idx2 = np.argmin(np.linalg.norm(pts - corner2, axis=1))

        # Assurer que idx1 < idx2 pour simplifier
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
            corner1, corner2 = corner2, corner1

        # Les deux chemins possibles
        path1 = pts[idx1:idx2 + 1]  # Chemin direct (sens horaire)
        path2 = np.concatenate([pts[idx2:], pts[:idx1 + 1]])  # Chemin long (sens anti-horaire)

        # Déterminer quel chemin prendre selon le côté
        # Calculer le centre de masse de la pièce pour référence
        center = np.mean(pts, axis=0)

        # Point milieu de chaque chemin
        if len(path1) > 1:
            mid1 = path1[len(path1) // 2]
        else:
            mid1 = (corner1 + corner2) / 2

        if len(path2) > 1:
            mid2 = path2[len(path2) // 2]
        else:
            mid2 = (corner1 + corner2) / 2

        # Vecteur du centre vers chaque point milieu
        vec1 = mid1 - center
        vec2 = mid2 - center

        # Choisir le bon chemin selon le côté demandé
        if side_name == "haut":
            # Pour le côté haut, on veut le chemin dont le milieu est le plus haut (y minimal)
            if mid1[1] < mid2[1]:
                return path1
            else:
                return path2[::-1]
        elif side_name == "bas":
            # Pour le côté bas, on veut le chemin dont le milieu est le plus bas (y maximal)
            if mid1[1] > mid2[1]:
                return path1
            else:
                return path2[::-1]
        elif side_name == "gauche":
            # Pour le côté gauche, on veut le chemin dont le milieu est le plus à gauche (x minimal)
            if mid1[0] < mid2[0]:
                return path1
            else:
                return path2[::-1]
        elif side_name == "droite":
            # Pour le côté droite, on veut le chemin dont le milieu est le plus à droite (x maximal)
            if mid1[0] > mid2[0]:
                return path1
            else:
                return path2[::-1]

        # Par défaut, prendre le chemin le plus court
        return path1

    def rotate_points(self, points, angle):
        """Fait une rotation des points autour de leur centre"""
        if len(points) == 0:
            return points

        # Centre des points
        center = np.mean(points, axis=0)

        # Matrice de rotation
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Appliquer la rotation
        centered_points = points - center
        rotated_points = np.dot(centered_points, rotation_matrix.T)

        return rotated_points + center

    def normalize_side_with_rotation(self, side_points, num_points=17):
        """Normalise un côté avec rotation pour aligner les extrémités horizontalement et orienter les tenons vers le haut"""
        if len(side_points) < 2:
            return np.zeros((num_points, 2)), 0

        # Calculer l'angle pour aligner les extrémités
        start_point = side_points[0]
        end_point = side_points[-1]

        # Vecteur entre début et fin
        direction_vector = end_point - start_point

        # Angle pour rendre ce vecteur horizontal
        angle = -np.arctan2(direction_vector[1], direction_vector[0])

        # Appliquer la rotation à tous les points
        rotated_points = self.rotate_points(side_points, angle)

        # Translater pour que le premier point soit à l'origine
        translated_points = rotated_points - rotated_points[0]

        # Déterminer si nous devons faire une rotation de 180° pour orienter les tenons vers le haut
        # Calculer la déviation moyenne par rapport à la ligne droite (y=0)
        y_coords = translated_points[:, 1]
        mean_y = np.mean(y_coords)

        # Si la moyenne des Y est négative, cela signifie que le côté est principalement en dessous
        # de la ligne droite, donc nous devons le retourner
        flip_needed = mean_y < 0
        total_angle = angle

        if flip_needed:
            # Rotation de 180° supplémentaire
            translated_points = self.rotate_points(translated_points, np.pi)
            total_angle += np.pi
            # Réajuster la translation après la rotation
            translated_points = translated_points - translated_points[0]

        # Calculer les distances cumulées le long du côté
        if len(translated_points) > 1:
            distances = np.cumsum(np.sqrt(np.sum(np.diff(translated_points, axis=0) ** 2, axis=1)))
            distances = np.insert(distances, 0, 0)
        else:
            distances = np.array([0])

        # Normaliser entre 0 et 1
        if distances[-1] > 0:
            distances = distances / distances[-1]

        # Interpoler pour obtenir un nombre fixe de points
        interp_distances = np.linspace(0, 1, num_points)

        try:
            if len(np.unique(distances)) > 1:  # Vérifier qu'on a des distances différentes
                interp_x = interpolate.interp1d(distances, translated_points[:, 0], kind='linear', bounds_error=False,
                                                fill_value='extrapolate')(interp_distances)
                interp_y = interpolate.interp1d(distances, translated_points[:, 1], kind='linear', bounds_error=False,
                                                fill_value='extrapolate')(interp_distances)
                normalized_side = np.column_stack([interp_x, interp_y])
            else:
                # Si tous les points sont au même endroit, créer une ligne droite
                normalized_side = np.linspace(translated_points[0], translated_points[-1], num_points)
        except (ValueError, IndexError):
            # En cas d'erreur, retourner une ligne droite
            normalized_side = np.linspace(translated_points[0], translated_points[-1], num_points)

        return normalized_side, total_angle

    def normalize_side(self, side_points, num_points=100):
        """Normalise un côté pour avoir un nombre fixe de points (méthode originale conservée)"""
        if len(side_points) < 2:
            return np.zeros((num_points, 2))

        # Calculer les distances cumulées le long du côté
        distances = np.cumsum(np.sqrt(np.sum(np.diff(side_points, axis=0) ** 2, axis=1)))
        distances = np.insert(distances, 0, 0)

        # Normaliser entre 0 et 1
        if distances[-1] > 0:
            distances = distances / distances[-1]

        # Interpoler pour obtenir un nombre fixe de points
        interp_distances = np.linspace(0, 1, num_points)

        try:
            interp_x = interpolate.interp1d(distances, side_points[:, 0], kind='linear')(interp_distances)
            interp_y = interpolate.interp1d(distances, side_points[:, 1], kind='linear')(interp_distances)
            normalized_side = np.column_stack([interp_x, interp_y])
        except ValueError:
            # En cas d'erreur, retourner une ligne droite
            normalized_side = np.linspace(side_points[0], side_points[-1], num_points)

        return normalized_side

    def classify_side_type(self, side_points):
        """Placeholder pour classification future"""
        return "à_classifier"

    def analyze_piece(self, filename):
        """Analyse une pièce complète"""
        filepath = os.path.join(self.input_dir, filename)

        # Charger l'image
        mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Trouver le contour principal
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        # Récupérer les coins depuis le JSON au lieu de les détecter
        try:
            corners = self.get_corners_for_piece(filename)
            print(f"Coins chargés pour {filename}: {corners}")
        except ValueError as e:
            print(f"Erreur pour {filename}: {e}")
            # Fallback vers la détection automatique si nécessaire
            corners = self.detect_corners(contour)
            print(f"Utilisation de la détection automatique pour {filename}")

        # Extraire les 4 côtés
        sides_data = {}
        side_names = ["haut", "droite", "bas", "gauche"]

        for i in range(4):
            corner1 = corners[i]
            corner2 = corners[(i + 1) % 4]

            # Extraire les points du côté
            side_points = self.extract_side_contour(contour, corner1, corner2, side_names[i])

            # Normaliser le côté (méthode originale)
            normalized_side = self.normalize_side(side_points)

            # Normaliser avec rotation et 17 points (15 points intermédiaires + 2 extrémités)
            normalized_side_rotated, rotation_angle = self.normalize_side_with_rotation(side_points, 17)

            # Classifier le type de côté
            side_type = self.classify_side_type(side_points)

            # Calculer la longueur
            length = np.linalg.norm(corner2 - corner1)

            sides_data[side_names[i]] = {
                'points_bruts': side_points,
                'points_normalises': normalized_side,
                'points_normalises_rotated': normalized_side_rotated,
                'angle_rotation': rotation_angle,
                'tenon_vers_haut': np.mean(normalized_side_rotated[:, 1]) >= 0,
                'type': side_type,
                'longueur': length,
                'coins': (corner1, corner2)
            }

        piece_data = {
            'filename': filename,
            'contour': contour,
            'corners': corners,
            'sides': sides_data,
            'mask': mask
        }

        return piece_data

    def visualize_piece_analysis(self, piece_data):
        """Visualise l'analyse d'une pièce avec les côtés normalisés et alignés"""
        if piece_data is None:
            return

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f"Analyse de {piece_data['filename']}", fontsize=16)

        # Image originale avec coins et contours
        img_color = cv2.cvtColor(piece_data['mask'], cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [piece_data['contour']], -1, (0, 0, 255), 2)

        # Marquer les coins (avec des couleurs différentes pour voir l'ordre)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGRA
        for idx, corner in enumerate(piece_data['corners']):
            cv2.circle(img_color, tuple(corner.astype(int)), 8, colors[idx], -1)
            # Ajouter le numéro du coin
            cv2.putText(img_color, str(idx), tuple((corner + 10).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        axes[0, 0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Contour et coins (JSON)")
        axes[0, 0].axis('off')

        # Afficher les 4 côtés originaux
        side_names = ["haut", "droite", "bas", "gauche"]
        positions_orig = [(0, 1), (0, 2), (1, 0), (1, 1)]
        positions_normalized = [(1, 2), (2, 0), (2, 1), (2, 2)]

        for i, side_name in enumerate(side_names):
            side_data = piece_data['sides'][side_name]

            # Côté original
            ax_orig = axes[positions_orig[i][0], positions_orig[i][1]]
            points = side_data['points_bruts']
            if len(points) > 0:
                ax_orig.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
                ax_orig.scatter([points[0, 0], points[-1, 0]], [points[0, 1], points[-1, 1]],
                                c='red', s=50, zorder=5)
                ax_orig.invert_yaxis()
            ax_orig.set_title(f"Côté {side_name} (original)\nLongueur: {side_data['longueur']:.1f}px")
            ax_orig.set_aspect('equal')
            ax_orig.grid(True, alpha=0.3)

            # Côté normalisé et aligné avec 17 points
            ax_norm = axes[positions_normalized[i][0], positions_normalized[i][1]]
            normalized_points = side_data['points_normalises_rotated']
            if len(normalized_points) > 0:
                ax_norm.plot(normalized_points[:, 0], normalized_points[:, 1], 'g-', linewidth=2, label='Courbe')
                # Marquer les 17 points
                ax_norm.scatter(normalized_points[:, 0], normalized_points[:, 1],
                                c='red', s=30, zorder=5, label='17 points')
                # Marquer spécialement le début et la fin
                ax_norm.scatter([normalized_points[0, 0], normalized_points[-1, 0]],
                                [normalized_points[0, 1], normalized_points[-1, 1]],
                                c='blue', s=60, zorder=6, marker='s', label='Extrémités')

            ax_norm.set_title(
                f"Côté {side_name} (normalisé - tenons vers le haut)\nAngle: {np.degrees(side_data['angle_rotation']):.1f}°")
            ax_norm.set_aspect('equal')
            ax_norm.grid(True, alpha=0.3)
            ax_norm.legend(fontsize=8)

        # Résumé textuel
        axes[1, 1].text(0.1, 0.9, "Résumé des côtés:", fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.85, f"Coins depuis: {self.corners_json_path}", fontsize=10, color='blue')
        y_pos = 0.75
        for side_name in side_names:
            side_data = piece_data['sides'][side_name]
            text = f"{side_name}: {side_data['longueur']:.1f}px, rot: {np.degrees(side_data['angle_rotation']):.1f}°"
            axes[1, 1].text(0.1, y_pos, text, fontsize=9)
            y_pos -= 0.12

        axes[1, 1].text(0.1, y_pos - 0.1, "Points par côté normalisé: 17", fontsize=10, weight='bold', color='red')
        axes[1, 1].text(0.1, y_pos - 0.2, "Tenons orientés vers le haut", fontsize=10, weight='bold', color='green')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def analyze_all_pieces(self, visualize=True):
        """Analyse toutes les pièces"""
        print(f"Analyse de {len(self.piece_files)} pièces...")

        for filename in self.piece_files:
            print(f"\nTraitement de {filename}...")
            piece_data = self.analyze_piece(filename)

            if piece_data is not None:
                self.pieces_data.append(piece_data)

                if visualize:
                    self.visualize_piece_analysis(piece_data)

                # Afficher le résumé textuel
                print(f"  Côtés détectés:")
                for side_name, side_data in piece_data['sides'].items():
                    print(
                        f"    - {side_name}: {side_data['longueur']:.1f}px, rotation: {np.degrees(side_data['angle_rotation']):.1f}°")

        print(f"\nAnalyse terminée. {len(self.pieces_data)} pièces traitées avec succès.")
        return self.pieces_data

    def get_sides_for_comparison(self):
        """Retourne les côtés normalisés pour comparaison"""
        comparison_data = []

        for piece_data in self.pieces_data:
            piece_sides = {
                'filename': piece_data['filename'],
                'sides': {}
            }

            for side_name, side_data in piece_data['sides'].items():
                piece_sides['sides'][side_name] = {
                    'points_bruts': side_data['points_bruts'],
                    'points_normalises': side_data['points_normalises'],
                    'points_normalises_rotated': side_data['points_normalises_rotated'],
                    'angle_rotation': side_data['angle_rotation'],
                    'tenon_vers_haut': side_data['tenon_vers_haut'],
                    'longueur': side_data['longueur']
                }

            comparison_data.append(piece_sides)

        return comparison_data

    def extract_features_from_normalized_sides(self):
        """Extrait les caractéristiques des côtés normalisés pour l'analyse"""
        features_data = []

        for piece_data in self.pieces_data:
            piece_features = {
                'filename': piece_data['filename'],
                'features': {}
            }

            for side_name, side_data in piece_data['sides'].items():
                normalized_points = side_data['points_normalises_rotated']

                if len(normalized_points) > 0:
                    # Extraire les coordonnées Y (déviation par rapport à la ligne droite)
                    y_coords = normalized_points[:, 1]
                    x_coords = normalized_points[:, 0]

                    # Caractéristiques géométriques
                    features = {
                        'y_coordinates': y_coords,  # Les 17 coordonnées Y (maintenant toujours >= 0)
                        'max_deviation': np.max(y_coords),  # Déviation maximale (maintenant toujours positive)
                        'mean_deviation': np.mean(y_coords),  # Déviation moyenne (maintenant toujours positive)
                        'area_under_curve': np.trapz(y_coords, x_coords),
                        # Aire sous la courbe (maintenant toujours positive)
                        'variation': np.var(y_coords),  # Variance
                        'length': side_data['longueur'],  # Longueur originale
                        'rotation_angle': side_data['angle_rotation'],  # Angle de rotation appliqué
                        'is_tenon': np.max(y_coords) > 0.1 * (np.max(x_coords) - np.min(x_coords)),
                        # Détection de tenon
                        'tenon_height': np.max(y_coords) if np.max(y_coords) > 0 else 0  # Hauteur du tenon
                    }

                    piece_features['features'][side_name] = features

            features_data.append(piece_features)

        return features_data

    def save_all_sides_features_to_json(self, output_file="puzzle_all_sides_features.json"):
        """Sauvegarde l'aire sous la courbe et les points normalisés pour les 4 côtés"""
        save_data = []

        for piece_data in self.pieces_data:
            piece_save = {
                'filename': piece_data['filename'],
                'sides': {}
            }

            side_names = ["haut", "droite", "bas", "gauche"]

            for side_name in side_names:
                if side_name in piece_data['sides']:
                    side_data = piece_data['sides'][side_name]
                    normalized_points = side_data['points_normalises_rotated']

                    if len(normalized_points) > 0:
                        # Extraire les coordonnées Y et X
                        y_coords = normalized_points[:, 1]
                        x_coords = normalized_points[:, 0]

                        # Calculer l'aire sous la courbe
                        area_under_curve = np.trapz(y_coords, x_coords)

                        piece_save['sides'][side_name] = {
                            'aire_sous_courbe': float(area_under_curve),
                            'points_normalises': normalized_points.tolist(),
                            'longueur': float(side_data['longueur']),
                            'angle_rotation': float(side_data['angle_rotation'])
                        }

            save_data.append(piece_save)

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Caractéristiques des 4 côtés sauvegardées dans {output_file}")
        print(f"Nombre de pièces sauvegardées: {len(save_data)}")

    def save_top_side_features_to_json(self, output_file="puzzle_top_sides_features.json"):
        """Sauvegarde l'aire sous la courbe et les points normalisés du côté haut seulement"""
        save_data = []

        for piece_data in self.pieces_data:
            if 'haut' in piece_data['sides']:
                side_data = piece_data['sides']['haut']
                normalized_points = side_data['points_normalises_rotated']

                if len(normalized_points) > 0:
                    # Extraire les coordonnées Y et X
                    y_coords = normalized_points[:, 1]
                    x_coords = normalized_points[:, 0]

                    # Calculer l'aire sous la courbe
                    area_under_curve = np.trapz(y_coords, x_coords)

                    piece_save = {
                        'filename': piece_data['filename'],
                        'side': 'haut',
                        'aire_sous_courbe': float(area_under_curve),
                        'points_normalises': normalized_points.tolist()
                    }

                    save_data.append(piece_save)

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Caractéristiques du côté haut sauvegardées dans {output_file}")
        print(f"Nombre de pièces sauvegardées: {len(save_data)}")


# === Utilisation ===
if __name__ == "__main__":
    # Créer l'analyseur avec le chemin vers le fichier JSON des coins
    analyzer = PuzzlePieceAnalyzer(
        input_dir="piece_noir_blanc",
        corners_json_path="corners_output/puzzle_corners.json"
    )

    # Analyser toutes les pièces
    pieces_data = analyzer.analyze_all_pieces(visualize=True)

    # Obtenir les données pour comparaison
    comparison_data = analyzer.get_sides_for_comparison()

    # Extraire les caractéristiques des côtés normalisés
    features_data = analyzer.extract_features_from_normalized_sides()

    # Sauvegarder les caractéristiques des 4 côtés en JSON
    analyzer.save_all_sides_features_to_json("puzzle_all_sides_features.json")

    # Optionnel: sauvegarder seulement le côté haut (version précédente)
    # analyzer.save_top_side_features_to_json("puzzle_top_sides_features.json")

    # Exemple d'accès aux données d'un côté spécifique
    if comparison_data:
        piece1 = comparison_data[0]
        print(f"\nExemple d'accès aux données:")
        print(f"Pièce: {piece1['filename']}")
        print(f"Longueur côté haut: {piece1['sides']['haut']['longueur']:.1f}px")
        print(f"Angle de rotation côté haut: {np.degrees(piece1['sides']['haut']['angle_rotation']):.1f}°")
        print(f"Points normalisés du côté haut: {piece1['sides']['haut']['points_normalises_rotated'].shape}")

        # Afficher les caractéristiques extraites
        if features_data:
            features1 = features_data[0]
            print(f"\nCaractéristiques du côté haut:")
            haut_features = features1['features']['haut']
            print(f"  - Coordonnées Y: {haut_features['y_coordinates']}")
            print(f"  - Déviation max: {haut_features['max_deviation']:.3f}")
            print(f"  - Aire sous courbe: {haut_features['area_under_curve']:.3f}")
            print(f"  - Est un tenon: {haut_features['is_tenon']}")
            print(f"  - Hauteur tenon: {haut_features['tenon_height']:.3f}")