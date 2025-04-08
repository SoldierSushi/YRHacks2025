# air_hockey.py (Modified for Menu Integration - NameError and Font Fixes)

import pygame
import sys
import cv2 # Make sure OpenCV is installed
import mediapipe as mp # Make sure Mediapipe is installed
import numpy as np # Make sure Numpy is installed
import random
import math
import time

# --- Constants ---
# Defaults used for calculations or standalone testing
DEFAULT_GAME_WIDTH, DEFAULT_GAME_HEIGHT = 700, 500
DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT = 360, 270

# Colors
BG_COLOR_HOCKEY = (15, 25, 40)
CENTER_LINE_COLOR = (50, 70, 100)
PADDLE_L_COLOR = (255, 80, 80)
PADDLE_R_COLOR = (80, 80, 255)
BALL_COLOR = (240, 240, 240)
SCORE_TEXT_L_COLOR = PADDLE_L_COLOR
SCORE_TEXT_R_COLOR = PADDLE_R_COLOR
INFO_TEXT_COLOR = (200, 200, 200)
CAM_BG = (5, 10, 20)
CAM_BORDER_COLOR = (40, 60, 90)
LANDMARK_COLOR = (80, 220, 80)
CONNECTION_COLOR = (60, 150, 60)
ERROR_COLOR = (255, 60, 60)
HIGHLIGHT_COLOR = (255, 255, 100)
TEXT_COLOR = (200, 210, 230) # General text color for game over message

# Game variables / Tuning parameters
PADDLE_WIDTH = 90
PADDLE_HEIGHT = 90
BALL_SIZE = 45
INITIAL_BALL_VEL_X = 0.2
INITIAL_BALL_VEL_Y = 4.0
FRICTION = 0.995
PADDLE_SPEED_INFLUENCE_X = 0.2
PADDLE_SPEED_INFLUENCE_Y = 0.6
RANDOM_BOUNCE_FACTOR = 0.5
MAX_BALL_VEL = 20
WINNING_SCORE = 0 # Score counts down

# --- Helper Functions ---
# (None needed for this game currently)

# --- Drawing Functions ---
def draw_hockey_table(surface, game_width, game_height):
    """Draws the air hockey table background and center line."""
    # surface.fill(BG_COLOR_HOCKEY) # Filling handled in main loop now
    center_line_width = 4; center_line_gap = 10
    num_dashes = game_height // (center_line_width + center_line_gap)
    for i in range(num_dashes):
        y_pos = i * (center_line_width + center_line_gap) + center_line_gap // 2
        pygame.draw.rect(surface, CENTER_LINE_COLOR, (game_width // 2 - center_line_width // 2, y_pos, center_line_width, center_line_width))
    goal_width = 6; goal_height = 150 # Simple visual goals
    pygame.draw.rect(surface, BALL_COLOR, (0, game_height//2 - goal_height//2, goal_width, goal_height))
    pygame.draw.rect(surface, BALL_COLOR, (game_width - goal_width, game_height//2 - goal_height//2, goal_width, goal_height))

def display_hockey_score(surface, left_score, right_score, game_width, score_font):
    """Displays the scores at the top."""
    try:
        left_text = score_font.render(f"{left_score}", True, SCORE_TEXT_L_COLOR)
        right_text = score_font.render(f"{right_score}", True, SCORE_TEXT_R_COLOR)
        surface.blit(left_text, (game_width // 4 - left_text.get_width() // 2, 20))
        surface.blit(right_text, (3 * game_width // 4 - right_text.get_width() // 2, 20))
    except pygame.error as e: print(f"Error displaying score: {e}")
    except AttributeError: print("Error: Score font not loaded for display_hockey_score")


def display_ball_speed(surface, vel_x, vel_y, game_width, game_height, info_font):
    """Displays the ball speed at the bottom center."""
    try:
        speed_pps = np.hypot(vel_x, vel_y) * 60 # Approx speed if running at 60fps target
        speed_text = info_font.render(f"Speed: {speed_pps:.1f} px/s", True, INFO_TEXT_COLOR)
        text_rect = speed_text.get_rect(center=(game_width // 2, game_height - 30))
        surface.blit(speed_text, text_rect)
    except pygame.error as e: print(f"Error displaying speed: {e}")
    except AttributeError: print("Error: Info font not loaded for display_ball_speed")


def hockey_game_over_message(surface, winner, game_width, game_height, go_font, restart_msg_font):
    """Draws the game over message."""
    overlay = pygame.Surface((game_width, game_height), pygame.SRCALPHA); overlay.fill((*BG_COLOR_HOCKEY, 220)); surface.blit(overlay, (0, 0))
    msg_width = 500; msg_height = 200
    msg_box_rect = pygame.Rect((game_width - msg_width) // 2, (game_height - msg_height) // 2, msg_width, msg_height)
    winner_color = PADDLE_L_COLOR if winner == "Player 1" else PADDLE_R_COLOR
    pygame.draw.rect(surface, CAM_BG, msg_box_rect, border_radius=5); pygame.draw.rect(surface, winner_color, msg_box_rect, width=3, border_radius=5)
    try:
        message_text = go_font.render(f":: {winner} WINS! ::", True, winner_color)
        restart_text = restart_msg_font.render("OPEN [REMATCH] | FIST [MENU]", True, TEXT_COLOR) # FIST goes to menu
        msg_rect = message_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 30))
        restart_rect = restart_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 30))
        surface.blit(message_text, msg_rect); surface.blit(restart_text, restart_rect);
    except pygame.error as e: print(f"Error drawing game over message: {e}")
    except AttributeError: print("Error: Game over fonts not loaded")


# --- Hand Tracking Functions ---
def get_hand_gesture(frame, results, mp_hands_instance):
    """Determines if the first detected hand is OPEN or FIST."""
    gesture = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Check first hand for gesture
        try:
            fingertips_ids=[mp_hands_instance.HandLandmark.INDEX_FINGER_TIP, mp_hands_instance.HandLandmark.MIDDLE_FINGER_TIP, mp_hands_instance.HandLandmark.RING_FINGER_TIP, mp_hands_instance.HandLandmark.PINKY_TIP]
            palm_center_approx_id=mp_hands_instance.HandLandmark.MIDDLE_FINGER_MCP; palm_center_pt = hand_landmarks.landmark[palm_center_approx_id]
            fingers_folded = 0; tip_threshold = 0.12
            for tip_id in fingertips_ids:
                tip_pt = hand_landmarks.landmark[tip_id]; distance = math.hypot(tip_pt.x - palm_center_pt.x, tip_pt.y - palm_center_pt.y)
                if distance < tip_threshold: fingers_folded += 1
            if fingers_folded >= 3: gesture = "FIST"
            else:
                 thumb_tip = hand_landmarks.landmark[mp_hands_instance.HandLandmark.THUMB_TIP]; thumb_dist = math.hypot(thumb_tip.x - palm_center_pt.x, thumb_tip.y - palm_center_pt.y)
                 if thumb_dist > 0.15: gesture = "OPEN"
        except (IndexError, AttributeError): pass
    return gesture

# --- Main Game Function ---
def run_hockey(surface):
    """Runs the Air Hockey game. Accepts the main screen surface.
       Returns next state ('game_select' or 'quit')."""
    print("--- Initializing Air Hockey ---")
    clock = pygame.time.Clock()

    # --- Load Fonts ---
    game_font, score_font, game_over_font, restart_font = None, None, None, None
    try:
        if not pygame.font.get_init(): pygame.font.init()
        try:
            font_path = "consola.ttf" # <<<--- ADJUST FONT PATH/NAME
            print(f"Attempting to load font: {font_path}")
            game_font = pygame.font.Font(font_path, 36); score_font = pygame.font.Font(font_path, 42);
            game_over_font = pygame.font.Font(font_path, 50); restart_font = pygame.font.Font(font_path, 30)
            print(f"Loaded bundled font '{font_path}' successfully.")
        except pygame.error as e:
            print(f"Warning: Error loading bundled font '{font_path}': {e}. Using SysFont.")
            HOCKEY_FONT_NAME = "Arial, Helvetica, sans-serif"; game_font = pygame.font.SysFont(HOCKEY_FONT_NAME, 36);
            score_font = pygame.font.SysFont(HOCKEY_FONT_NAME, 42, bold=True); game_over_font = pygame.font.SysFont(HOCKEY_FONT_NAME, 50, bold=True);
            restart_font = pygame.font.SysFont(HOCKEY_FONT_NAME, 30); print("Using fallback SysFonts.")
    except Exception as e:
        print(f"Error loading SysFonts ({e}). Falling back to default.");
        try:
             if not pygame.font.get_init(): pygame.font.init()
             game_font = pygame.font.SysFont(None, 38); score_font = pygame.font.SysFont(None, 45);
             game_over_font = pygame.font.SysFont(None, 55); restart_font = pygame.font.SysFont(None, 32)
             print("Using absolute default fallback SysFonts.")
        except Exception as e2: print(f"FATAL: Could not load any fonts: {e2}"); return "quit"

    # --- Calculate Dimensions ---
    total_width = surface.get_width(); total_height = surface.get_height()
    cam_panel_width = DEFAULT_CAM_WIDTH; game_width = total_width - cam_panel_width; game_height = total_height

    # --- OpenCV and MediaPipe Setup ---
    cap = None; native_cam_width = 0; native_cam_height = 0; hands = None; mp_hands = None; mp_draw = None; mp_drawing_styles = None # Initialize all to None
    try:
        cap = cv2.VideoCapture(1);
        if not cap or not cap.isOpened():
            print("Warning: Camera 1 failed, trying Camera 0..."); cap = cv2.VideoCapture(0)
            if not cap or not cap.isOpened(): raise IOError("Cannot open webcam")
        native_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); native_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Native camera resolution: {native_cam_width}x{native_cam_height}")
        mp_hands = mp.solutions.hands; hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles # <<<--- INITIALIZE mp_drawing_styles HERE
        print("Camera and MediaPipe initialized for Air Hockey.")
    except Exception as e: print(f"Error initializing camera or MediaPipe: {e}"); return "game_select"

    # --- Calculate Aspect-Corrected Camera Size ---
    target_cam_width = DEFAULT_CAM_WIDTH; aspect_ratio = native_cam_height / native_cam_width if native_cam_width > 0 else 9/16
    target_cam_height = int(target_cam_width * aspect_ratio)
    max_allowable_cam_height = total_height - 30
    if target_cam_height > max_allowable_cam_height:
        target_cam_height = max_allowable_cam_height; target_cam_width = int(target_cam_height / aspect_ratio) if aspect_ratio > 0 else DEFAULT_CAM_WIDTH
    print(f"Target aspect-corrected camera display size: {target_cam_width}x{target_cam_height}")

    # --- Game Restart Loop ---
    while True:
        # Initial Game State
        left_paddle_x = 100; left_paddle_y = game_height // 2 - PADDLE_HEIGHT // 2
        right_paddle_x = game_width - 100 - PADDLE_WIDTH; right_paddle_y = game_height // 2 - PADDLE_HEIGHT // 2
        ball_x = game_width // 2 - BALL_SIZE // 2; ball_y = game_height // 2 - BALL_SIZE // 2
        ball_vel_x = random.choice([-1, 1]) * INITIAL_BALL_VEL_X; ball_vel_y = random.choice([-1, 1]) * INITIAL_BALL_VEL_Y
        left_score = 10; right_score = 10; game_over = False; winner = None
        prev_paddleL_x, prev_paddleL_y = left_paddle_x, left_paddle_y; prev_paddleR_x, prev_paddleR_y = right_paddle_x, right_paddle_y
        hand_assignment = [None, None] # Could be used for more robust hand tracking later

        # --- Gameplay Loop ---
        while not game_over:
            delta_time_ms = clock.tick(60) # Get ms passed since last tick
            delta_time_sec = delta_time_ms / 1000.0 # Convert to seconds

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if cap and cap.isOpened(): cap.release(); print("Camera released on QUIT.")
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if cap and cap.isOpened(): cap.release(); print("Camera released on ESC.")
                        return "game_select"

            # --- Hand Tracking & Paddle Update ---
            success, frame = cap.read();
            if not success: print("Warning: Failed to read frame."); time.sleep(0.05); continue
            frame = cv2.flip(frame, 1); image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Optimization
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            dxl, dyl = 0, 0; dxr, dyr = 0, 0 # Reset velocities
            assigned_left = False; assigned_right = False; current_hands_data = []

            if results.multi_hand_landmarks:
                for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Use middle finger base as center reference
                    center_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    current_hands_data.append({'id': hand_id, 'x': center_landmark.x, 'y': center_landmark.y, 'landmarks': hand_landmarks})

                # Simple X-position assignment
                current_hands_data.sort(key=lambda h: h['x']) # Sort hands by x position
                left_hand_data = current_hands_data[0] if len(current_hands_data) > 0 else None
                right_hand_data = current_hands_data[1] if len(current_hands_data) > 1 else None
                # Handle single hand case (assign based on side)
                if len(current_hands_data) == 1:
                    if current_hands_data[0]['x'] < 0.5: right_hand_data = None
                    else: left_hand_data = None; right_hand_data = current_hands_data[0]

                if left_hand_data:
                    target_lx = left_hand_data['x'] * game_width; target_ly = left_hand_data['y'] * game_height
                    left_paddle_x = target_lx - PADDLE_WIDTH // 2; left_paddle_y = target_ly - PADDLE_HEIGHT // 2; assigned_left = True
                if right_hand_data:
                    target_rx = right_hand_data['x'] * game_width; target_ry = right_hand_data['y'] * game_height
                    right_paddle_x = target_rx - PADDLE_WIDTH // 2; right_paddle_y = target_ry - PADDLE_HEIGHT // 2; assigned_right = True

            # Clamp paddles
            left_paddle_x = max(0, min(left_paddle_x, game_width // 2 - PADDLE_WIDTH))
            left_paddle_y = max(0, min(left_paddle_y, game_height - PADDLE_HEIGHT))
            right_paddle_x = max(game_width // 2, min(right_paddle_x, game_width - PADDLE_WIDTH))
            right_paddle_y = max(0, min(right_paddle_y, game_height - PADDLE_HEIGHT))
            # Calculate velocities
            dxl = left_paddle_x - prev_paddleL_x; dyl = left_paddle_y - prev_paddleL_y
            dxr = right_paddle_x - prev_paddleR_x; dyr = right_paddle_y - prev_paddleR_y
            # Update previous positions
            prev_paddleL_x, prev_paddleL_y = left_paddle_x, left_paddle_y
            prev_paddleR_x, prev_paddleR_y = right_paddle_x, right_paddle_y

            # --- Ball Logic ---
            # Use delta_time_sec for more consistent physics if framerate varies
            ball_x += ball_vel_x * delta_time_sec * 100 # Scale factor to make speeds feel right
            ball_y += ball_vel_y * delta_time_sec * 100
            ball_vel_x *= (FRICTION ** delta_time_sec) # Apply friction based on time
            ball_vel_y *= (FRICTION ** delta_time_sec)
            if ball_y <= 0: ball_y = 0; ball_vel_y *= -1
            elif ball_y >= game_height - BALL_SIZE: ball_y = game_height - BALL_SIZE; ball_vel_y *= -1
            ball_vel_x = max(-MAX_BALL_VEL, min(ball_vel_x, MAX_BALL_VEL)); ball_vel_y = max(-MAX_BALL_VEL, min(ball_vel_y, MAX_BALL_VEL))
            left_paddle_rect = pygame.Rect(left_paddle_x, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
            right_paddle_rect = pygame.Rect(right_paddle_x, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
            ball_rect = pygame.Rect(ball_x, ball_y, BALL_SIZE, BALL_SIZE)
            # Paddle Collisions
            if left_paddle_rect.colliderect(ball_rect) and ball_vel_x < 0: # Check velocity direction
                ball_x = left_paddle_x + PADDLE_WIDTH; ball_vel_x *= -1 # Reverse X
                ball_vel_x += dxl * PADDLE_SPEED_INFLUENCE_X; ball_vel_y += dyl * PADDLE_SPEED_INFLUENCE_Y
                ball_vel_y += random.uniform(-RANDOM_BOUNCE_FACTOR, RANDOM_BOUNCE_FACTOR)
            elif right_paddle_rect.colliderect(ball_rect) and ball_vel_x > 0:
                ball_x = right_paddle_x - BALL_SIZE; ball_vel_x *= -1
                ball_vel_x += dxr * PADDLE_SPEED_INFLUENCE_X; ball_vel_y += dyr * PADDLE_SPEED_INFLUENCE_Y
                ball_vel_y += random.uniform(-RANDOM_BOUNCE_FACTOR, RANDOM_BOUNCE_FACTOR)
            # Scoring
            if ball_x < 0:
                right_score -= 1; ball_x = game_width // 2 - BALL_SIZE // 2; ball_y = game_height // 2 - BALL_SIZE // 2
                ball_vel_x = INITIAL_BALL_VEL_X; ball_vel_y = random.choice([-1, 1]) * INITIAL_BALL_VEL_Y
            elif ball_x > game_width - BALL_SIZE:
                left_score -= 1; ball_x = game_width // 2 - BALL_SIZE // 2; ball_y = game_height // 2 - BALL_SIZE // 2
                ball_vel_x = -INITIAL_BALL_VEL_X; ball_vel_y = random.choice([-1, 1]) * INITIAL_BALL_VEL_Y
            # Check for game over
            if left_score <= 0: game_over = True; winner = "Player 2"
            elif right_score <= 0: game_over = True; winner = "Player 1"

            # --- Drawing ---
            surface.fill(BG_COLOR_HOCKEY) # Fill entire surface might be easier
            draw_hockey_table(surface, game_width, game_height) # Draw table markings in game area
            pygame.draw.rect(surface, PADDLE_L_COLOR, left_paddle_rect)
            pygame.draw.rect(surface, PADDLE_R_COLOR, right_paddle_rect)
            pygame.draw.ellipse(surface, BALL_COLOR, ball_rect)
            display_hockey_score(surface, left_score, right_score, game_width, score_font)
            display_ball_speed(surface, ball_vel_x, ball_vel_y, game_width, game_height, game_font)
            cam_panel_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_panel_rect)
            # Camera Feed Drawing
            if native_cam_width > 0: aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
            else: aspect_corrected_frame = cv2.resize(frame, (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), interpolation=cv2.INTER_AREA)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks: # Draw all detected hands
                    mp_draw.draw_landmarks(aspect_corrected_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing_styles.get_default_hand_landmarks_style(),
                                           mp_drawing_styles.get_default_hand_connections_style())
            small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB")
            cam_panel_center_x = game_width + cam_panel_width // 2; cam_display_area_y_start = 30; cam_display_area_height = total_height - cam_display_area_y_start; cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
            frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y)); surface.blit(pygame_frame, frame_rect.topleft)
            pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_panel_rect, width=2) # Border
            pygame.display.flip()
        # --- End of Gameplay Loop ---

        # --- Game Over State ---
        restart_attempt = False; gesture_check_start_time = time.time()
        print(f"Entering Game Over state... Winner: {winner}")
        # Draw initial game over screen (redraw BG + final state + message)
        surface.fill(BG_COLOR_HOCKEY); draw_hockey_table(surface, game_width, game_height);
        pygame.draw.rect(surface, PADDLE_L_COLOR, left_paddle_rect); pygame.draw.rect(surface, PADDLE_R_COLOR, right_paddle_rect)
        pygame.draw.ellipse(surface, BALL_COLOR, ball_rect); display_hockey_score(surface, left_score, right_score, game_width, score_font)
        cam_panel_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_panel_rect); pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_panel_rect, width=2)
        hockey_game_over_message(surface, winner, game_width, game_height, game_over_font, restart_font) # Pass correct fonts
        pygame.display.flip()

        while game_over: # Game over display loop
            for event in pygame.event.get(): # Event Handling
                if event.type == pygame.QUIT:
                    if cap and cap.isOpened(): cap.release(); print("Camera released on QUIT.")
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if cap and cap.isOpened(): cap.release(); print("Camera released on ESC.")
                        return "game_select"
            # Gesture Check
            if time.time() - gesture_check_start_time > 0.1:
                gesture_check_start_time = time.time(); success, frame = cap.read(); gesture = None
                if success:
                    frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True
                    gesture = get_hand_gesture(frame, results, mp_hands)
                    # Redraw only camera area + game over overlay
                    cam_rect_for_redraw = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_rect_for_redraw)
                    if native_cam_width > 0: aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
                    else: aspect_corrected_frame = cv2.resize(frame, (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), interpolation=cv2.INTER_AREA)
                    if results.multi_hand_landmarks:
                         for hand_landmarks in results.multi_hand_landmarks:
                             mp_draw.draw_landmarks(aspect_corrected_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                    if gesture: cv2.putText(aspect_corrected_frame, gesture, (10, target_cam_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
                    small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB")
                    cam_panel_center_x = game_width + cam_panel_width // 2; cam_display_area_y_start = 30; cam_display_area_height = total_height - cam_display_area_y_start; cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
                    frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y)); surface.blit(pygame_frame, frame_rect.topleft)
                    pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_rect_for_redraw, width=2) # Border
                    # Re-blit game over message overlay
                    hockey_game_over_message(surface, winner, game_width, game_height, game_over_font, restart_font)
                    pygame.display.flip()
                # Check Gestures AFTER drawing
                if gesture == "OPEN": print("Gesture: OPEN - Rematch!"); game_over = False; restart_attempt = True; break
                elif gesture == "FIST": print("Gesture: FIST - Returning to menu...");
                if cap and cap.isOpened(): cap.release(); print("Camera released on FIST gesture.")
                return "game_select"
            clock.tick(15) # Lower tick rate
        # End of Game Over Loop
        if not restart_attempt:
             print("Game Over loop exited without restart signal.");
             if cap and cap.isOpened(): cap.release(); print("Camera released on Game Over exit.")
             return "game_select"
        print("--- Restarting game... ---"); time.sleep(0.5)
    # --- End of Outer Restart Loop ---
    print("Exiting run_hockey function unexpectedly.");
    if cap and cap.isOpened(): cap.release(); print("Camera released on unexpected exit.")
    return "game_select"

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("Running Air Hockey standalone for testing...")
    pygame.init(); pygame.font.init()
    test_screen_width = DEFAULT_GAME_WIDTH + DEFAULT_CAM_WIDTH; test_screen_height = max(DEFAULT_GAME_HEIGHT, DEFAULT_CAM_HEIGHT)
    test_screen = pygame.display.set_mode((test_screen_width, test_screen_height))
    pygame.display.set_caption("Air Hockey Standalone Test")
    result = run_hockey(test_screen) # Call main function
    print(f"Game exited with state: {result}")
    pygame.quit(); sys.exit()