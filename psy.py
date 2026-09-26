import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. Pengaturan Partikel
NUM_PARTICLES = 1500
particles_pos = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
particles_target = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
particles_color = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)

# --- Generator Bentuk-Bentuk Partikel ---

# Bentuk 1: Medan Bintang (Starfield) - 5 Jari
def generate_starfield():
    pos = np.random.uniform(-4, 4, (NUM_PARTICLES, 3))
    colors = np.tile([0.3, 0.7, 1.0], (NUM_PARTICLES, 1)) + np.random.uniform(-0.1, 0.1, (NUM_PARTICLES, 3))
    return pos.astype(np.float32), np.clip(colors, 0, 1).astype(np.float32)

# Bentuk 2: Planet Saturnus - 1 Jari
def generate_saturn():
    pos = []
    colors = []
    # Bola Planet
    num_sphere = int(NUM_PARTICLES * 0.4)
    for _ in range(num_sphere):
        u = random.random()
        v = random.random()
        theta = u * 2.0 * math.pi
        phi = math.acos(2.0 * v - 1.0)
        r = 1.0
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        pos.append([x, y, z])
        colors.append([1.0, 0.5, 0.1])  # Oranye

    # Cincin Saturnus
    num_ring = NUM_PARTICLES - num_sphere
    for _ in range(num_ring):
        theta = random.uniform(0, 2 * math.pi)
        r = random.uniform(1.5, 2.5)
        x = r * math.cos(theta)
        y = random.uniform(-0.05, 0.05)
        z = r * math.sin(theta)
        pos.append([x, y, z])
        colors.append([0.9, 0.8, 0.4])  # Kuning Emas

    return np.array(pos, dtype=np.float32), np.array(colors, dtype=np.float32)

# Bentuk 3: Hati / Heart 3D - Kepalan Tangan (0 Jari)
def generate_heart():
    pos = []
    colors = []
    for _ in range(NUM_PARTICLES):
        t = random.uniform(0, 2 * math.pi)
        x = 16 * (math.sin(t) ** 3) / 10
        y = (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 10
        z = random.uniform(-0.3, 0.3)
        pos.append([x, y, z])
        colors.append([1.0, 0.1, 0.3])  # Merah/Pink
    return np.array(pos, dtype=np.float32), np.array(colors, dtype=np.float32)

# Bentuk 4: Teks "I LOVE YOU" - 2 Jari
def generate_text_ilu():
    pos = []
    colors = []
    text_pts = []
    
    # "I"
    for y in np.linspace(-1, 1, 30):
        text_pts.append([-2.5, y, 0])
    
    # Simbol Hati (Love)
    for t in np.linspace(0, 2*math.pi, 50):
        hx = (16 * math.sin(t)**3) / 25
        hy = (13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t)) / 25
        text_pts.append([hx, hy, 0])

    # "U"
    for y in np.linspace(0, 1, 20):
        text_pts.append([1.8, y, 0])
        text_pts.append([2.8, y, 0])
    for t in np.linspace(math.pi, 2*math.pi, 20):
        text_pts.append([2.3 + 0.5*math.cos(t), 0 + 0.5*math.sin(t), 0])

    base_count = len(text_pts)
    for i in range(NUM_PARTICLES):
        pt = text_pts[i % base_count]
        jitter = np.random.normal(0, 0.05, 3)
        pos.append(np.array(pt) + jitter)
        colors.append([0.2, 0.9, 1.0])  # Biru Muda / Cyan

    return np.array(pos, dtype=np.float32), np.array(colors, dtype=np.float32)

# Bentuk 5: Silinder - 3 Jari
def generate_cylinder():
    pos = []
    colors = []
    for _ in range(NUM_PARTICLES):
        theta = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.8, 1.2)
        x = r * math.cos(theta)
        z = r * math.sin(theta)
        y = random.uniform(-1.5, 1.5)
        pos.append([x, y, z])
        colors.append([0.1, 0.8, 0.6])  # Hijau Toska
    return np.array(pos, dtype=np.float32), np.array(colors, dtype=np.float32)


# Posisi Awal Partikel
particles_pos, particles_color = generate_starfield()
particles_target = particles_pos.copy()

# Fungsi Menghitung Jari Terbuka
def count_fingers(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    cnt = 0
    # Jempol
    if landmarks[4].x < landmarks[3].x:
        cnt += 1
    # 4 Jari lainnya
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            cnt += 1
    return cnt


# --- Program Utama ---
def main():
    global particles_pos, particles_target, particles_color

    # Inisialisasi Window Pygame OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Hand Gesture 3D Particle System")

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -7)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glPointSize(3.0)

    # Inisialisasi Kamera / Webcam
    cap = cv2.VideoCapture(0)
    current_gesture = -1
    angle = 0

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Ambil frame dari kamera
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_fingers = -1
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected_fingers = count_fingers(hand_landmarks.landmark)

        # Ubah Bentuk Partikel Berdasarkan Jumlah Jari
        if detected_fingers != current_gesture and detected_fingers != -1:
            current_gesture = detected_fingers
            if current_gesture == 5 or current_gesture == 4:
                particles_target, particles_color = generate_starfield()
            elif current_gesture == 2:
                particles_target, particles_color = generate_text_ilu()
            elif current_gesture == 1:
                particles_target, particles_color = generate_saturn()
            elif current_gesture == 0:
                particles_target, particles_color = generate_heart()
            elif current_gesture == 3:
                particles_target, particles_color = generate_cylinder()

        # Efek Transisi Mulus (Animasi Gerak Partikel)
        particles_pos += (particles_target - particles_pos) * 0.1

        # Tampilkan Jendela Webcam
        cv2.imshow("Hand Tracking WebCam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Render Partikel 3D OpenGL
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        
        # Rotasi Otomatis
        angle += 1.0
        glRotatef(angle, 0, 1, 0)

        glBegin(GL_POINTS)
        for i in range(NUM_PARTICLES):
            glColor3fv(particles_color[i])
            glVertex3fv(particles_pos[i])
        glEnd()

        glPopMatrix()
        pygame.display.flip()
        clock.tick(60)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == '__main__':
    main()