import pygame
import cv2
import numpy as np
import mediapipe as mp
import random  # Added for random y variation

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 1000
HEIGHT = 600
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Hand-Controlled Air Hockey")

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(2)

# Paddle properties
PADDLE_WIDTH = 90
PADDLE_HEIGHT = 90
left_paddle_x = 100
left_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
right_paddle_x = WIDTH - 100
right_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2

# Ball properties
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_size = 45
ball_vel_x = 0.2
ball_vel_y = 2.5
friction = 0.995

# Game variables
left_score = 10
right_score = 10
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Previous positions
prev_ball_x = ball_x
prev_ball_y = ball_y
prev_paddleL_x = left_paddle_x
prev_paddleL_y = left_paddle_y
prev_paddleR_x = right_paddle_x
prev_paddleR_y = right_paddle_y

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Webcam
    success, image = cap.read()
    if not success:
        continue
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y
            screen_x = int(index_x * WIDTH)
            screen_y = int(index_y * HEIGHT)

            if i == 0:
                if screen_x < WIDTH // 2:
                    left_paddle_x = screen_x - PADDLE_WIDTH // 2
                    left_paddle_y = screen_y - PADDLE_HEIGHT // 2
                else:
                    right_paddle_x = screen_x - PADDLE_WIDTH // 2
                    right_paddle_y = screen_y - PADDLE_HEIGHT // 2
            else:
                if screen_x < WIDTH // 2:
                    left_paddle_x = screen_x - PADDLE_WIDTH // 2
                    left_paddle_y = screen_y - PADDLE_HEIGHT // 2
                else:
                    right_paddle_x = screen_x - PADDLE_WIDTH // 2
                    right_paddle_y = screen_y - PADDLE_HEIGHT // 2

        # Clamp paddles
        left_paddle_x = max(0, min(left_paddle_x, WIDTH // 2 - PADDLE_WIDTH))
        left_paddle_y = max(0, min(left_paddle_y, HEIGHT - PADDLE_HEIGHT))
        right_paddle_x = max(WIDTH // 2, min(right_paddle_x, WIDTH - PADDLE_WIDTH))
        right_paddle_y = max(0, min(right_paddle_y, HEIGHT - PADDLE_HEIGHT))

        # Paddle speeds
        dxl = left_paddle_x - prev_paddleL_x
        dyl = left_paddle_y - prev_paddleL_y
        dxr = right_paddle_x - prev_paddleR_x
        dyr = right_paddle_y - prev_paddleR_y

        prev_paddleL_x, prev_paddleL_y = left_paddle_x, left_paddle_y
        prev_paddleR_x, prev_paddleR_y = right_paddle_x, right_paddle_y

    # Update ball
    ball_x += ball_vel_x
    ball_y += ball_vel_y

    ball_vel_x *= friction
    ball_vel_y *= friction

    if ball_y <= 0:
        ball_y = 0
        ball_vel_y *= -1
    elif ball_y >= HEIGHT - ball_size:
        ball_y = HEIGHT - ball_size
        ball_vel_y *= -1

    ball_vel_x = max(-20, min(ball_vel_x, 20))
    ball_vel_y = max(-20, min(ball_vel_y, 20))

    left_paddle_rect = pygame.Rect(left_paddle_x, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    right_paddle_rect = pygame.Rect(right_paddle_x, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball_rect = pygame.Rect(ball_x, ball_y, ball_size, ball_size)

    # Left paddle collision
    if left_paddle_rect.colliderect(ball_rect):
        overlap_x = min(ball_x + ball_size - left_paddle_x, left_paddle_x + PADDLE_WIDTH - ball_x)
        overlap_y = min(ball_y + ball_size - left_paddle_y, left_paddle_y + PADDLE_HEIGHT - ball_y)

        if overlap_x < overlap_y:
            # Side collision
            if ball_x < left_paddle_x:
                ball_x = left_paddle_x - ball_size
                ball_vel_x = -abs(dxl)
            else:
                ball_x = left_paddle_x + PADDLE_WIDTH
                ball_vel_x = abs(ball_vel_x)
            ball_vel_x += dxl * 0.2
            ball_vel_y += dyl * 0.4  # 🔥 Stronger Y influence even on side hit
        else:
            # Top/bottom collision
            if ball_y < left_paddle_y:
                ball_y = left_paddle_y - ball_size
                ball_vel_y = -abs(ball_vel_y)
            else:
                ball_y = left_paddle_y + PADDLE_HEIGHT
                ball_vel_y = abs(ball_vel_y)
            ball_vel_y += dyl * 0.6  # 🔥 Even stronger Y influence

        ball_vel_y += random.uniform(-0.5, 0.5)  # Add slight random bounce


    # Right paddle collision
    if right_paddle_rect.colliderect(ball_rect):
        overlap_x = min(ball_x + ball_size - right_paddle_x, right_paddle_x + PADDLE_WIDTH - ball_x)
        overlap_y = min(ball_y + ball_size - right_paddle_y, right_paddle_y + PADDLE_HEIGHT - ball_y)

        if overlap_x < overlap_y:
            # Side collision
            if ball_x < right_paddle_x:
                ball_x = right_paddle_x - ball_size
                ball_vel_x = -abs(ball_vel_x)
            else:
                ball_x = right_paddle_x + PADDLE_WIDTH
                ball_vel_x = abs(ball_vel_x)
            ball_vel_x += dxr * 0.2
            ball_vel_y += dyr * 0.4  # 🔥 Stronger Y influence even on side hit
        else:
            # Top/bottom collision
            if ball_y < right_paddle_y:
                ball_y = right_paddle_y - ball_size
                ball_vel_y = -abs(ball_vel_y)
            else:
                ball_y = right_paddle_y + PADDLE_HEIGHT
                ball_vel_y = abs(ball_vel_y)
            ball_vel_y += dyr * 0.6  # 🔥 Even stronger Y influence

        ball_vel_y += random.uniform(-0.5, 0.5)  # Add slight random bounce

    # Ball out of bounds
    if ball_x < 0:
        left_score -= 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_vel_x, ball_vel_y = -0.2, -2.5
    elif ball_x > WIDTH - ball_size:
        right_score -= 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_vel_x, ball_vel_y = 0.2, 2.5

    # Draw everything
    screen.fill((0, 0, 0))
    for y in range(0, HEIGHT, 20):
        pygame.draw.rect(screen, (50, 50, 50), (WIDTH // 2 - 2, y, 4, 10))
    pygame.draw.rect(screen, (255, 100, 100), left_paddle_rect)
    pygame.draw.rect(screen, (100, 100, 255), right_paddle_rect)
    pygame.draw.ellipse(screen, (255, 255, 255), ball_rect)

    # Score & speed
    ball_speed = np.hypot(ball_vel_x, ball_vel_y) * 60
    screen.blit(font.render(f"Player 1: {left_score}", True, (255, 100, 100)), (WIDTH // 4 - 70, 20))
    screen.blit(font.render(f"Player 2: {right_score}", True, (100, 100, 255)), (3 * WIDTH // 4 - 70, 20))
    screen.blit(font.render(f"Speed: {ball_speed:.2f} px/s", True, (255, 255, 255)), (WIDTH // 2 - 80, HEIGHT - 40))

    pygame.display.flip()

    # Webcam
    cv2.imshow("Hand Tracking", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
