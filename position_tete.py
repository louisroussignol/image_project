import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os

# Initialisation de Mediapipe et OpenCV
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Variables globales pour la calibration
calibration_data = {'left_eye': [], 'right_eye': []}  # Contiendra les coordonnees des yeux pour les moyennes
remaining_left_captures = 3  # Nombre de captures restantes pour l'œil gauche
remaining_right_captures = 3  # Nombre de captures restantes pour l'œil droit
# Variables pour les moyennes de distance
avg_left_eye_distance = None
avg_right_eye_distance = None
pyautogui.FAILSAFE = False
# Fonction pour calculer l'ecart vertical des yeux (avec inversion des yeux)
def calculate_eye_height_distance(eye_landmarks, eye_side):
    if eye_side == 'left_eye':  # oeil droit
        top_point = eye_landmarks[386].y  # pupille superieur de l'œil droit
        bottom_point = eye_landmarks[374].y  # pupille inferieur de l'œil droit
    elif eye_side == 'right_eye':  #oeil gauche
        top_point = eye_landmarks[159].y  # Coin superieur de l'œil gauche
        bottom_point = eye_landmarks[145].y  # Coin inferieur de l'œil gauche
    eye_distance = abs(top_point - bottom_point)
    return eye_distance

# Fonction pour afficher les instructions sur la video
def display_instructions(frame, stage, remaining_left, remaining_right):
    if remaining_left == 0 and remaining_right == 0:
        text = "Calibration terminee. Appuyez sur 'q' pour passer a l'etape de test."
    else:
        if stage == 'left_eye':
            text = f"Capture de l'oeil gauche. Il reste {remaining_left} capture(s)."
        elif stage == 'right_eye':
            text = f"Capture de l'oeil droit. Il reste {remaining_right} capture(s)."
        else:
            text = "Veuillez attendre le debut de la calibration."

    cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def calibrate_eye(eye_side, cap):
    global remaining_left_captures, remaining_right_captures
    stage = eye_side
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_frame)

            # Si des points clés sont détectés
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculer l'ecart vertical des yeux
                    left_eye_distance = calculate_eye_height_distance(face_landmarks.landmark, 'left_eye')
                    right_eye_distance = calculate_eye_height_distance(face_landmarks.landmark, 'right_eye')

                    # Affichage des distances a l'ecran (sans les filtres de dessin)
                    cv2.putText(frame, f"Gauche: {left_eye_distance:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Droit: {right_eye_distance:.3f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            display_instructions(frame, stage, remaining_left_captures, remaining_right_captures)
            cv2.imshow('Calibration de la souris', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quitter en appuyant sur 'q'
                break
            if key == ord('c'):  # Appuyer sur 'c' pour capturer
                if stage == 'left_eye' and remaining_left_captures > 0:
                    print("Capture de l'oeil gauche demandee.")
                    calibration_data['left_eye'].append(left_eye_distance)
                    remaining_left_captures -= 1
                elif stage == 'right_eye' and remaining_right_captures > 0:
                    print("Capture de l'oeil droit demandee.")
                    calibration_data['right_eye'].append(right_eye_distance)
                    remaining_right_captures -= 1

                # Passer a l'oeil droit apres avoir termine l'oeil gauche
                if remaining_left_captures == 0 and remaining_right_captures > 0:
                    print("Passage a l'oeil droit...")
                    stage = 'right_eye'

    global avg_left_eye_distance, avg_right_eye_distance
    avg_left_eye_distance = np.mean(calibration_data['left_eye']) if calibration_data['left_eye'] else None
    avg_right_eye_distance = np.mean(calibration_data['right_eye']) if calibration_data['right_eye'] else None
    cv2.destroyAllWindows()

# fonction pour savoir si les yeux sont fermés
def is_eye_closed(eye_side, eye_distance):
    if eye_side == 'left_eye':
        return eye_distance < avg_right_eye_distance * 1  # Si la distance est plus petite que la moyenne
    elif eye_side == 'right_eye':
        return eye_distance < avg_left_eye_distance * 1  # Si la distance est plus petite la moyenne
    return False
# Fonction pour demarrer la calibration des yeux
def start_calibration():
    global remaining_left_captures, remaining_right_captures
    global avg_left_eye_distance, avg_right_eye_distance
    cap = cv2.VideoCapture(0)  # Ouvrir la camera une seule fois
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return
    # Modification de la résolution (selon la camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Démarrage de la calibration
    calibrate_eye('left_eye', cap)

    # Après la calibration, afficher les moyennes sur le retour camera
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : impossible de lire le flux de la camera.")
            break

        # Affichage des moyennes
        text_left = f"Moyenne oeil gauche : {avg_left_eye_distance:.3f}" if avg_left_eye_distance is not None else "Moyenne oeil gauche : N/A"
        text_right = f"Moyenne oeil droit : {avg_right_eye_distance:.3f}" if avg_right_eye_distance is not None else "Moyenne oeil droit : N/A"

        # Dessiner le texte sur l'image
        cv2.putText(frame, text_left, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, text_right, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour continuer.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Afficher la frame avec les textes
        cv2.imshow("Calibration terminee", frame)

        # Attendre que l'utilisateur appuie sur 'q' pour continuer
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()  # Fermer toutes les fenetres
            break
    cap.release()
def control_mouse_with_head():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return
    # Définir la résolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    init_time=0
    previous_time=0
    screen_width, screen_height = pyautogui.size()
    center_width,center_height=screen_width//2,screen_height//2
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur : impossible de lire le flux vidéo.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # On prend ici le premier visage détecté (index 0)
                face_landmarks = results.multi_face_landmarks[0]

                # Obtenir les coordonnées du nez pour déplacer la souris
                nose_tip = face_landmarks.landmark[1]
                frame_height, frame_width, _ = frame.shape
                nose_x = int(nose_tip.x * frame_width)
                nose_y = int(nose_tip.y * frame_height)
                screen_x = int(nose_x / frame_width * screen_width)
                screen_y = int(nose_y / frame_height * screen_height)
                left_eye_distance = calculate_eye_height_distance(face_landmarks.landmark, 'left_eye')
                right_eye_distance = calculate_eye_height_distance(face_landmarks.landmark, 'right_eye')
                # Vérifier si les yeux sont fermés
                left_eye_closed = is_eye_closed('left_eye', left_eye_distance)
                right_eye_closed = is_eye_closed('right_eye', right_eye_distance)

                if left_eye_closed and right_eye_closed:
                    if init_time==0:
                        previous_time=time.time()
                        init_time=1
                    elif time.time()-previous_time>=3:
                        break
                elif left_eye_closed:
                    pyautogui.rightClick()
                    init_time=0
                elif right_eye_closed:
                    pyautogui.leftClick()
                    init_time=0
                else:
                    init_time=0
                    move_x = screen_x - center_width
                    move_y = screen_y - center_height

                    # Déplacer la souris de manière relative
                    pyautogui.moveRel(-move_x*0.8, move_y*0.8)#0.8 pour limiter les mouvements brusques et -0.8 car la camera est inversé horizontalement
            # Afficher la vidéo
            cv2.imshow("Controle de la souris avec la tete", frame)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
def clear_terminal():
    # Vérifie le système d'exploitation
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # macOS et Linux
        os.system('clear')

start_calibration()
if avg_left_eye_distance!=None and avg_right_eye_distance!=None:
    control_mouse_with_head()
#clear_terminal()