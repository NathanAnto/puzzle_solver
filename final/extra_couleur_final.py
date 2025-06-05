import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


class PuzzleColorExtractor:
    def __init__(self, color_images_dir="piece", points_json_file="puzzle_points_bruts.json"):
        self.color_images_dir = color_images_dir
        self.points_json_file = points_json_file
        self.pieces_colors_data = []

        # Charger les données de points depuis le JSON
        with open(points_json_file, 'r') as f:
            self.points_data = json.load(f)

        print(f"Chargé {len(self.points_data)} pièces depuis {points_json_file}")

    def get_perpendicular_points(self, point1, point2, offset=2):
        """
        Calcule des points perpendiculaires vers l'intérieur de la pièce
        pour échantillonner la couleur du côté plutôt que le contour
        """
        # Vecteur du côté
        side_vector = np.array(point2) - np.array(point1)
        side_length = np.linalg.norm(side_vector)

        if side_length == 0:
            return [point1, point2]

        # Vecteur unitaire du côté
        side_unit = side_vector / side_length

        # Vecteur perpendiculaire (rotation de 90°)
        perpendicular = np.array([-side_unit[1], side_unit[0]])

        # Points décalés vers l'intérieur (on teste les deux directions)
        offset_points_1 = []
        offset_points_2 = []

        # Point milieu du segment
        mid_point = (np.array(point1) + np.array(point2)) / 2

        # Décalage dans les deux directions perpendiculaires
        offset_1 = mid_point + perpendicular * offset
        offset_2 = mid_point - perpendicular * offset

        return [offset_1.astype(int), offset_2.astype(int), mid_point.astype(int)]

    def sample_colors_between_points(self, image, point1, point2, num_samples=5):
        """
        Échantillonne les couleurs entre deux points en incluant des points
        légèrement décalés vers l'intérieur pour éviter les contours
        """
        colors = []

        # Points d'échantillonnage le long du segment
        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            sample_point = np.array(point1) * (1 - t) + np.array(point2) * t
            sample_point = sample_point.astype(int)

            # Vérifier que le point est dans l'image
            if (0 <= sample_point[0] < image.shape[1] and
                    0 <= sample_point[1] < image.shape[0]):

                # Échantillonner plusieurs points autour pour avoir une couleur représentative
                sample_colors = []

                # Point principal
                color = image[sample_point[1], sample_point[0]]
                if not np.array_equal(color, [0, 0, 0]):  # Éviter le noir (contour)
                    sample_colors.append(color)

                # Points décalés vers l'intérieur
                perpendicular_points = self.get_perpendicular_points(point1, point2, offset=3)
                for perp_point in perpendicular_points:
                    if (0 <= perp_point[0] < image.shape[1] and
                            0 <= perp_point[1] < image.shape[0]):
                        perp_color = image[perp_point[1], perp_point[0]]
                        if not np.array_equal(perp_color, [0, 0, 0]):
                            sample_colors.append(perp_color)

                # Moyenne des couleurs échantillonnées
                if sample_colors:
                    avg_color = np.mean(sample_colors, axis=0).astype(int)
                    colors.append(avg_color.tolist())

        return colors

    def extract_average_color_between_points(self, image, point1, point2):
        """
        Extrait la couleur moyenne entre deux points en évitant les contours
        """
        # Échantillonner plusieurs couleurs le long du segment
        sampled_colors = self.sample_colors_between_points(image, point1, point2, num_samples=7)

        if not sampled_colors:
            # Si aucune couleur valide trouvée, essayer avec des décalages plus importants
            perpendicular_points = self.get_perpendicular_points(point1, point2, offset=5)
            for perp_point in perpendicular_points:
                if (0 <= perp_point[0] < image.shape[1] and
                        0 <= perp_point[1] < image.shape[0]):
                    color = image[perp_point[1], perp_point[0]]
                    if not np.array_equal(color, [0, 0, 0]):
                        return color.tolist()

            # Dernier recours : couleur du point milieu
            mid_point = ((np.array(point1) + np.array(point2)) / 2).astype(int)
            if (0 <= mid_point[0] < image.shape[1] and
                    0 <= mid_point[1] < image.shape[0]):
                return image[mid_point[1], mid_point[0]].tolist()
            else:
                return [128, 128, 128]  # Gris par défaut

        # Calculer la couleur moyenne en excluant les valeurs aberrantes
        sampled_colors = np.array(sampled_colors)

        # Éliminer les couleurs trop sombres (probablement des contours)
        brightness = np.mean(sampled_colors, axis=1)
        valid_colors = sampled_colors[brightness > 50]

        if len(valid_colors) > 0:
            return np.mean(valid_colors, axis=0).astype(int).tolist()
        else:
            return np.mean(sampled_colors, axis=0).astype(int).tolist()

    def extract_colors_for_piece(self, piece_data):
        """
        Extrait les couleurs pour tous les côtés d'une pièce
        """
        filename = piece_data['filename']
        image_path = os.path.join(self.color_images_dir, filename)

        # Charger l'image couleur
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur: impossible de charger l'image {image_path}")
            return None

        # Convertir BGR vers RGB pour matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        piece_colors = {
            'filename': filename,
            'sides': {}
        }

        side_names = ["haut", "droite", "bas", "gauche"]

        for side_name in side_names:
            if side_name not in piece_data['sides']:
                continue

            points_17 = piece_data['sides'][side_name]['points_17']
            side_colors = []

            # Pour chaque paire de points consécutifs (16 segments)
            for i in range(len(points_17) - 1):
                point1 = points_17[i]
                point2 = points_17[i + 1]

                # Extraire la couleur moyenne entre ces deux points
                avg_color = self.extract_average_color_between_points(image_rgb, point1, point2)

                side_colors.append({
                    'segment': f"point_{i}_to_{i + 1}",
                    'point1': point1,
                    'point2': point2,
                    'color_rgb': avg_color,
                    'color_hex': f"#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}"
                })

            piece_colors['sides'][side_name] = {
                'segments': side_colors,
                'total_segments': len(side_colors)
            }

        return piece_colors

    def extract_all_colors(self):
        """
        Extrait les couleurs pour toutes les pièces
        """
        print(f"Extraction des couleurs pour {len(self.points_data)} pièces...")

        for i, piece_data in enumerate(self.points_data):
            print(f"Traitement de {piece_data['filename']} ({i + 1}/{len(self.points_data)})...")

            piece_colors = self.extract_colors_for_piece(piece_data)
            if piece_colors:
                self.pieces_colors_data.append(piece_colors)

        print(f"Extraction terminée. {len(self.pieces_colors_data)} pièces traitées.")
        return self.pieces_colors_data

    def save_colors_to_json(self, output_file="puzzle_colors.json"):
        """
        Sauvegarde les couleurs dans un fichier JSON
        """
        with open(output_file, 'w') as f:
            json.dump(self.pieces_colors_data, f, indent=2)

        print(f"Couleurs sauvegardées dans {output_file}")

    def visualize_piece_colors(self, piece_filename, show_segments=True):
        """
        Visualise les couleurs extraites pour une pièce donnée
        """
        # Trouver les données de la pièce
        piece_colors = None
        piece_points = None

        for pc in self.pieces_colors_data:
            if pc['filename'] == piece_filename:
                piece_colors = pc
                break

        for pp in self.points_data:
            if pp['filename'] == piece_filename:
                piece_points = pp
                break

        if not piece_colors or not piece_points:
            print(f"Pièce {piece_filename} non trouvée")
            return

        # Charger l'image originale
        image_path = os.path.join(self.color_images_dir, piece_filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Couleurs extraites - {piece_filename}", fontsize=16)

        side_names = ["haut", "droite", "bas", "gauche"]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for i, side_name in enumerate(side_names):
            ax = axes[positions[i][0], positions[i][1]]

            if side_name not in piece_colors['sides']:
                ax.set_title(f"Côté {side_name} - Non trouvé")
                ax.axis('off')
                continue

            # Afficher l'image de base
            ax.imshow(image_rgb)

            # Dessiner les points et segments avec leurs couleurs
            side_data = piece_colors['sides'][side_name]
            points_17 = piece_points['sides'][side_name]['points_17']

            # Marquer tous les points
            for j, point in enumerate(points_17):
                ax.scatter(point[0], point[1], c='white', s=20, edgecolors='black', linewidth=1)
                if j % 4 == 0:  # Numéroter quelques points
                    ax.annotate(str(j), (point[0], point[1]), xytext=(5, 5),
                                textcoords='offset points', fontsize=8, color='white')

            # Dessiner les segments avec leurs couleurs extraites
            if show_segments:
                for segment in side_data['segments']:
                    point1 = segment['point1']
                    point2 = segment['point2']
                    color_rgb = np.array(segment['color_rgb']) / 255.0

                    # Ligne colorée pour le segment
                    ax.plot([point1[0], point2[0]], [point1[1], point2[1]],
                            color=color_rgb, linewidth=4, alpha=0.8)

            ax.set_title(f"Côté {side_name} - {side_data['total_segments']} segments")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def compare_side_colors(self, piece1_name, side1_name, piece2_name, side2_name):
        """
        Compare les couleurs entre deux côtés de pièces différentes
        """
        # Trouver les données des deux pièces
        piece1_colors = None
        piece2_colors = None

        for pc in self.pieces_colors_data:
            if pc['filename'] == piece1_name:
                piece1_colors = pc
            elif pc['filename'] == piece2_name:
                piece2_colors = pc

        if not piece1_colors or not piece2_colors:
            print("Une ou les deux pièces non trouvées")
            return None

        if (side1_name not in piece1_colors['sides'] or
                side2_name not in piece2_colors['sides']):
            print("Un ou les deux côtés non trouvés")
            return None

        side1_segments = piece1_colors['sides'][side1_name]['segments']
        side2_segments = piece2_colors['sides'][side2_name]['segments']

        print(f"\nComparaison: {piece1_name} {side1_name} vs {piece2_name} {side2_name}")
        print("=" * 70)

        # Comparer segment par segment
        min_segments = min(len(side1_segments), len(side2_segments))
        total_difference = 0

        for i in range(min_segments):
            color1 = np.array(side1_segments[i]['color_rgb'])
            color2 = np.array(side2_segments[i]['color_rgb'])

            # Distance euclidienne dans l'espace RGB
            color_diff = euclidean(color1, color2)
            total_difference += color_diff

            print(f"Segment {i}: {side1_segments[i]['color_hex']} vs {side2_segments[i]['color_hex']} "
                  f"(diff: {color_diff:.2f})")

        average_difference = total_difference / min_segments if min_segments > 0 else 0
        print(f"\nDifférence moyenne: {average_difference:.2f}")
        print(
            f"Compatibilité: {'Bonne' if average_difference < 50 else 'Moyenne' if average_difference < 100 else 'Faible'}")

        return {
            'piece1': piece1_name,
            'side1': side1_name,
            'piece2': piece2_name,
            'side2': side2_name,
            'average_difference': average_difference,
            'segments_compared': min_segments
        }

    def find_best_color_matches(self, target_piece, target_side, max_results=5):
        """
        Trouve les meilleurs correspondances de couleur pour un côté donné
        """
        if not self.pieces_colors_data:
            print("Aucune donnée de couleur disponible")
            return []

        target_colors = None
        for pc in self.pieces_colors_data:
            if pc['filename'] == target_piece:
                if target_side in pc['sides']:
                    target_colors = pc['sides'][target_side]['segments']
                break

        if not target_colors:
            print(f"Côté {target_side} de {target_piece} non trouvé")
            return []

        matches = []

        # Comparer avec tous les autres côtés
        for piece_colors in self.pieces_colors_data:
            if piece_colors['filename'] == target_piece:
                continue  # Skip la même pièce

            for side_name, side_data in piece_colors['sides'].items():
                comparison = self.compare_side_colors(
                    target_piece, target_side,
                    piece_colors['filename'], side_name
                )
                if comparison:
                    matches.append(comparison)

        # Trier par différence moyenne (plus faible = meilleure correspondance)
        matches.sort(key=lambda x: x['average_difference'])

        return matches[:max_results]


# === Utilisation ===
if __name__ == "__main__":
    # Créer l'extracteur de couleurs
    extractor = PuzzleColorExtractor(
        color_images_dir="piece",
        points_json_file="puzzle_points_bruts.json"
    )

    # Extraire les couleurs pour toutes les pièces
    colors_data = extractor.extract_all_colors()

    # Sauvegarder les résultats
    extractor.save_colors_to_json("puzzle_colors.json")

    # Exemples d'utilisation
    if colors_data:
        print("\n" + "=" * 50)
        print("EXEMPLES D'UTILISATION")
        print("=" * 50)

        # Visualiser une pièce
        first_piece = colors_data[0]['filename']
        print(f"\nVisualisation de {first_piece}:")
        extractor.visualize_piece_colors(first_piece)

        # Comparer deux côtés
        if len(colors_data) >= 2:
            piece1 = colors_data[0]['filename']
            piece2 = colors_data[1]['filename']

            print(f"\nComparaison: {piece1} haut vs {piece2} bas")
            comparison = extractor.compare_side_colors(piece1, "haut", piece2, "bas")

            # Trouver les meilleures correspondances
            print(f"\nMeilleures correspondances pour {piece1} côté haut:")
            matches = extractor.find_best_color_matches(piece1, "haut", max_results=3)

            for i, match in enumerate(matches, 1):
                print(f"{i}. {match['piece2']} {match['side2']} "
                      f"(différence: {match['average_difference']:.2f})")

        # Afficher un résumé
        print(f"\n" + "=" * 50)
        print("RÉSUMÉ")
        print("=" * 50)
        print(f"Pièces traitées: {len(colors_data)}")

        total_segments = 0
        for piece in colors_data:
            for side in piece['sides'].values():
                total_segments += side['total_segments']

        print(f"Segments de couleur extraits: {total_segments}")
        print(f"Données sauvegardées dans: puzzle_colors.json")