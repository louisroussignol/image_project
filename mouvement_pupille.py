import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os
# Initialisation de Mediapipe et pyautogui
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
pyautogui.FAILSAFE = False
# Variables globales pour la calibration
calibration_data = {'left_eye': [], 'right_eye': []}  # Contiendra les coordonnees des yeux pour les moyennes
remaining_left_captures = 3  # Nombre de captures restantes
remaining_right_captures = 3  # Nombre de captures restantes
# Variables pour les moyennes de distance
avg_left_eye_distance = None
avg_right_eye_distance = None
left_pupil = None
right_pupil = None
nose = None
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

        cv2.putText(frame, text_left, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, text_right, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour continuer.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Calibration terminee", frame)

        # Attendre que l'utilisateur appuie sur 'q' pour continuer
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()  # Fermer toutes les fenetres
            break
    cap.release()
def capture_positions():
    global left_pupil, right_pupil, nose
    screen_width,screen_height=pyautogui.size()
    center_width,center_height=screen_width//2,screen_height//2
    # Initialisation de la capture vidéo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return
    # Définir la résolution de la caméra à 1080p (1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Dimensions de l'image
        h, w, _ = frame.shape
        pyautogui.moveTo(center_width,center_height)
        if results.multi_face_landmarks:
            # Récupérer les landmarks du premier visage détecté
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Indices des pupilles et du nez
            left_pupil_index = 468  # Pupille gauche
            right_pupil_index = 473  # Pupille droite
            nose_index = 1  # Nez
            left_pupil_landmark = face_landmarks[left_pupil_index]
            right_pupil_landmark = face_landmarks[right_pupil_index]
            nose_landmark = face_landmarks[nose_index]
            # Convertir les coordonnées normalisées en pixels pour l'affichage
            left_pupil_coords = (int(left_pupil_landmark.x * w), int(left_pupil_landmark.y * h))
            right_pupil_coords = (int(right_pupil_landmark.x * w), int(right_pupil_landmark.y * h))
            nose_coords = (int(nose_landmark.x * w), int(nose_landmark.y * h))

            # Dessiner les points sur l'image
            cv2.circle(frame, left_pupil_coords, 5, (0, 255, 0), -1)  # Point vert : pupille gauche
            cv2.circle(frame, right_pupil_coords, 5, (255, 0, 0), -1)  # Point bleu : pupille droite
            cv2.circle(frame, nose_coords, 5, (0, 0, 255), -1)  # Point rouge : nez

            cv2.putText(frame, "Fixer la souris et appuyez sur 'c'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.imshow("Capture des positions", frame)

        # Appuie sur c pour capturer
        if cv2.waitKey(1) & 0xFF == ord('c') and results.multi_face_landmarks:
            h, w, _ = frame.shape
            # Stocker les positions dans les variables globales
            left_pupil = (left_pupil_landmark.x*w, left_pupil_landmark.y*h)
            right_pupil = (right_pupil_landmark.x*w, right_pupil_landmark.y*h)
            nose = (nose_landmark.x*w, nose_landmark.y*h)
            print("Positions capturées :")
            print(f"Pupille gauche : {left_pupil}")
            print(f"Pupille droite : {right_pupil}")
            print(f"Nez : {nose}")
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def follow_pupils():
    global left_pupil, right_pupil, nose
    distance_left_pupil_nose=(nose[0]-left_pupil[0],-left_pupil[1]+nose[1])
    distance_right_pupil_nose=(right_pupil[0]-nose[0],-right_pupil[1]+nose[1])
    print(distance_left_pupil_nose,distance_right_pupil_nose)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la caméra.")
        return
    #Modif résolution camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    init_time = 0
    previous_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Récupérer les landmarks du premier visage détecté
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark

            # Dimensions de l'image
            h, w, _ = frame.shape

            try:
                left_pupil =landmarks[468]  # Pupille gauche
                right_pupil = landmarks[473]  # Pupille droite
                noze=landmarks[1]
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
                # Si la position précédente est définie
                    if noze.z!=-1:
            # Convertir les coordonnées normalisées en pixels
                        left_pupil_coords = (left_pupil.x * w, left_pupil.y * h)
                        right_pupil_coords = (right_pupil.x * w, right_pupil.y * h)
                        noze_coords=(noze.x*w,noze.y*h,noze.z)
                        distance_left=(noze_coords[0]-left_pupil_coords[0],noze_coords[1]-left_pupil_coords[1])
                        distance_right=(right_pupil_coords[0]-noze_coords[0],noze_coords[1]-right_pupil_coords[1])
                        diff_distance_left=(distance_left[0]-distance_left_pupil_nose[0],distance_left[1]-distance_left_pupil_nose[1])
                        diff_distance_right=(distance_right[0]-distance_right_pupil_nose[0],distance_right[1]-distance_right_pupil_nose[1])
                        #print(diff_distance_left,diff_distance_right)
                        moy_distance=((diff_distance_left[0]-diff_distance_right[0])/(2*(1+noze_coords[2])),(diff_distance_left[1]+diff_distance_right[1])/(2*(1+noze_coords[2])))
                        if abs(moy_distance[0])>=2 and abs(moy_distance[1])>=2:
                            pyautogui.moveRel(moy_distance[0]*5,-moy_distance[1]*5)
                        elif abs(moy_distance[0])>=2:
                            pyautogui.moveRel(moy_distance[0]*5,0)
                        elif abs(moy_distance[1])>=2:
                            pyautogui.moveRel(0,-moy_distance[1]*5)
            except IndexError:
                print("Pupille non détectée : les indices de Mediapipe ne sont pas valides.")

        cv2.imshow("Pupil Detection", frame)

        # Quitter avec la touche "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def clear_terminal():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # macOS et Linux
        os.system('clear')

start_calibration()
capture_positions()
if avg_left_eye_distance!=None and avg_right_eye_distance!=None and left_pupil != None and right_pupil != None and nose != None:
    follow_pupils()
clear_terminal()