import cv2
import numpy as np
import os

# Dossier contenant les pièces remplies
input_folder = "pieces_remplie"

# Lister toutes les images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Erreur de lecture : {filename}")
            continue

        print(f"Traitement de : {filename}")

        # --- Détection des coins de Harris ---
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        dst = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)
        image_color[dst > 0.01 * dst.max()] = [0, 0, 0]  # Rouge pour les coins

        #cv2.imshow("Coins détectés", image_color)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # --- Détection des lignes (Hough) ---
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=10, maxLineGap=500000)

        image_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #cv2.imshow("Lignes détectées (Hough)", image_lines)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # --- Sélection des 4 côtés extrêmes ---
        if lines is not None and len(lines) >= 4:
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < abs(x2 - x1):  # horizontale
                    horizontal_lines.append(line[0])
                else:
                    vertical_lines.append(line[0])

            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                horizontal_lines = sorted(horizontal_lines, key=lambda x: x[1])
                vertical_lines = sorted(vertical_lines, key=lambda x: x[0])

                top_line = horizontal_lines[0]
                bottom_line = horizontal_lines[-1]
                left_line = vertical_lines[0]
                right_line = vertical_lines[-1]

                # Redessiner sur une copie
                final_image = image_color.copy()
                cv2.line(final_image, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (255, 0, 255), 2)
                cv2.line(final_image, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (255, 0, 0), 2)
                cv2.line(final_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 255), 2)
                cv2.line(final_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 2)

                cv2.imshow("Côtés détectés", final_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Pas assez de lignes pour détecter les 4 côtés.")
        else:
            print("Pas de lignes détectées.")
