import pygame
import cv2
import numpy as np
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Hand-Controlled Pong")

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define paddle properties
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
left_paddle_y = HEIGHT//2 - PADDLE_HEIGHT//2
right_paddle_y = HEIGHT//2 - PADDLE_HEIGHT//2
paddle_speed = 7

# Define ball properties
ball_x = WIDTH//2
ball_y = HEIGHT//2
ball_size = 15
ball_speed_x = 7
ball_speed_y = 7

# Game variables
left_score = 0
right_score = 0
running = True

while running:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Read webcam
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)
    
    # If hands are detected, update paddle positions
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the y position of the index finger tip
            index_y = hand_landmarks.landmark[8].y
            
            # Determine which hand controls which paddle based on x position
            index_x = hand_landmarks.landmark[8].x
            if index_x < 0.5:  # Left side of screen
                left_paddle_y = int(index_y * HEIGHT)
                left_paddle_y = max(0, min(left_paddle_y, HEIGHT - PADDLE_HEIGHT))
            else:  # Right side of screen
                right_paddle_y = int(index_y * HEIGHT)
                right_paddle_y = max(0, min(right_paddle_y, HEIGHT - PADDLE_HEIGHT))

    # Update ball position
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Ball collision with top and bottom
    if ball_y <= 0 or ball_y >= HEIGHT - ball_size:
        ball_speed_y *= -1

    # Ball collision with paddles
    left_paddle_rect = pygame.Rect(50, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    right_paddle_rect = pygame.Rect(WIDTH - 50 - PADDLE_WIDTH, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball_rect = pygame.Rect(ball_x, ball_y, ball_size, ball_size)
    
    if left_paddle_rect.colliderect(ball_rect) or right_paddle_rect.colliderect(ball_rect):
        ball_speed_x *= -1
        if left_paddle_rect.colliderect(ball_rect):
            left_score += 1
        else:
            right_score += 1

    # Ball out of bounds
    if ball_x >= WIDTH - ball_size:  # Right player missed
        ball_x = WIDTH//2
        ball_y = HEIGHT//2
        left_score += 1
    elif ball_x <= 0:  # Left player missed
        ball_x = WIDTH//2
        ball_y = HEIGHT//2
        right_score += 1

    # Fill the background with black
    screen.fill((0, 0, 0))

    # Draw paddles (white)
    pygame.draw.rect(screen, (255, 255, 255), left_paddle_rect)
    pygame.draw.rect(screen, (255, 255, 255), right_paddle_rect)

    # Draw ball (white)
    pygame.draw.rect(screen, (255, 255, 255), ball_rect)

    # Display scores
    font = pygame.font.Font(None, 36)
    left_score_text = font.render(f"Player 1: {left_score}", True, (255, 255, 255))
    right_score_text = font.render(f"Player 2: {right_score}", True, (255, 255, 255))
    screen.blit(left_score_text, (WIDTH//4 - 70, 20))
    screen.blit(right_score_text, (3*WIDTH//4 - 70, 20))

    pygame.display.flip()

    # Display the webcam feed
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

    # Control game speed
    pygame.time.Clock().tick(60)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
