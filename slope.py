# matrix_dodge.py (Modified for Integration with Menu)

import pygame
import sys
import cv2 # Make sure OpenCV is installed (pip install opencv-python)
import mediapipe as mp # Make sure Mediapipe is installed (pip install mediapipe)
import random
import math
import time
from collections import deque

# --- Constants (Can stay global) ---
# Defaults used for calculations or standalone testing
DEFAULT_GAME_WIDTH, DEFAULT_GAME_HEIGHT = 600, 500
DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT = 360, 270

# Colors
MATRIX_GREEN_BRIGHT = (20, 255, 90); MATRIX_GREEN_MED = (10, 200, 70); MATRIX_GREEN_DARK = (0, 100, 30)
BG_COLOR = (0, 5, 0); GRID_COLOR = (0, 40, 10); SCANLINE_COLOR = (0, 30, 5)
BALL_COLOR = MATRIX_GREEN_BRIGHT; BALL_OUTLINE = MATRIX_GREEN_MED
OBSTACLE_COLOR = (255, 0, 0); OBSTACLE_OUTLINE = (150, 0, 0)
TEXT_COLOR = MATRIX_GREEN_MED; SCORE_TEXT_COLOR = MATRIX_GREEN_BRIGHT; SCORE_LABEL_COLOR = TEXT_COLOR
HIGHLIGHT_COLOR = (180, 255, 180); ERROR_COLOR = (255, 60, 60)
CAM_BG = (0, 20, 5); CAM_BORDER_COLOR = MATRIX_GREEN_DARK
LANDMARK_COLOR = MATRIX_GREEN_BRIGHT; CONNECTION_COLOR = MATRIX_GREEN_DARK; DIRECTION_LINE_COLOR = HIGHLIGHT_COLOR
TIMER_TEXT_COLOR = HIGHLIGHT_COLOR
TIMER_LABEL_COLOR = TEXT_COLOR

# Game variables / Tuning parameters
BALL_RADIUS_LOGICAL = 14
SMOOTHING_SPEED = 6.0
INITIAL_SCROLL_SPEED = 3.0; MAX_SCROLL_SPEED = 9.0
TARGET_SPEED_TIME_FACTOR = 0.15; SPEED_LERP_FACTOR = 0.8
OBSTACLE_WIDTH_LOGICAL = 40; OBSTACLE_HEIGHT_LOGICAL = 40
OBSTACLE_SPAWN_INTERVAL_MIN = 0.4; OBSTACLE_SPAWN_INTERVAL_MAX = 1.2
OBSTACLE_MAX_COUNT = 30
SCANLINE_HEIGHT = 1; SCANLINE_ALPHA = 20
PIXELATION_FACTOR = 5 # How much to pixelate the game area

# --- Helper Functions ---
def lerp(a, b, t): return a + (b - a) * t

# --- Drawing Functions (Pixelated for Game Area) ---
def draw_hacker_background_pixelated(target_surface):
    """Draws scanlines onto the pixelated game surface."""
    pixel_w = target_surface.get_width(); pixel_h = target_surface.get_height()
    scanline_surface = pygame.Surface((pixel_w, SCANLINE_HEIGHT), pygame.SRCALPHA); scanline_surface.fill((*SCANLINE_COLOR, SCANLINE_ALPHA))
    for y in range(0, pixel_h, SCANLINE_HEIGHT * 2): target_surface.blit(scanline_surface, (0, y))

def draw_obstacles_pixelated(target_surface, obstacles, pixel_obstacle_w, pixel_obstacle_h, pix_factor):
    """Draws obstacles onto the pixelated game surface."""
    pixel_h = target_surface.get_height()
    for obs_rect in obstacles:
        x_pixel = obs_rect.x // pix_factor; y_pixel = obs_rect.y // pix_factor
        if y_pixel > pixel_h or y_pixel + pixel_obstacle_h < 0: continue
        ob_rect_pixel = pygame.Rect(x_pixel, y_pixel, pixel_obstacle_w, pixel_obstacle_h)
        pygame.draw.rect(target_surface, OBSTACLE_COLOR, ob_rect_pixel)

def draw_player_pixelated(target_surface, player_x, player_y, pixel_ball_radius, pix_factor):
    """Draws the player ball onto the pixelated game surface."""
    player_x_pixel = int(player_x) // pix_factor
    player_y_pixel = int(player_y) // pix_factor
    player_x_pixel = max(pixel_ball_radius, min(target_surface.get_width() - pixel_ball_radius, player_x_pixel))
    player_y_pixel = max(pixel_ball_radius, min(target_surface.get_height() - pixel_ball_radius, player_y_pixel))
    pygame.draw.circle(target_surface, BALL_COLOR, (player_x_pixel, player_y_pixel), pixel_ball_radius)

# --- UI Drawing Functions (Draw directly onto main surface) ---
def display_score(surface, score, total_width, cam_panel_width, label_font, value_font):
    """Displays score top-right in camera area."""
    score_area_width = (cam_panel_width // 2) - 20; score_area_height = 60
    score_area_x = (total_width - cam_panel_width) + cam_panel_width - score_area_width - 15; score_area_y = 15
    score_area_rect = pygame.Rect(score_area_x, score_area_y, score_area_width, score_area_height)
    pygame.draw.rect(surface, CAM_BG, score_area_rect, border_radius=3); pygame.draw.rect(surface, CAM_BORDER_COLOR, score_area_rect, width=1, border_radius=3)
    try:
        score_label_text = label_font.render("SCORE", True, SCORE_LABEL_COLOR); score_value_text = value_font.render(f"{int(score):04}", True, SCORE_TEXT_COLOR)
        surface.blit(score_label_text, (score_area_rect.centerx - score_label_text.get_width() // 2, score_area_rect.y + 8))
        surface.blit(score_value_text, (score_area_rect.centerx - score_value_text.get_width() // 2, score_area_rect.y + 28))
    except pygame.error as e: print(f"Error displaying score: {e}")

def display_timer(surface, elapsed_time, total_width, cam_panel_width, label_font, value_font):
    """Displays the elapsed game time top-left in camera area."""
    timer_area_width = (cam_panel_width // 2) - 20; timer_area_height = 60
    timer_area_x = (total_width - cam_panel_width) + 15; timer_area_y = 15
    timer_area_rect = pygame.Rect(timer_area_x, timer_area_y, timer_area_width, timer_area_height)
    pygame.draw.rect(surface, CAM_BG, timer_area_rect, border_radius=3); pygame.draw.rect(surface, CAM_BORDER_COLOR, timer_area_rect, width=1, border_radius=3)
    time_str = f"{elapsed_time:.1f}s"
    try:
        timer_label_text = label_font.render("TIME", True, TIMER_LABEL_COLOR); timer_value_text = value_font.render(time_str, True, TIMER_TEXT_COLOR)
        surface.blit(timer_label_text, (timer_area_rect.centerx - timer_label_text.get_width() // 2, timer_area_rect.y + 8))
        surface.blit(timer_value_text, (timer_area_rect.centerx - timer_value_text.get_width() // 2, timer_area_rect.y + 28))
    except pygame.error as e: print(f"Error displaying timer: {e}")

def game_over_message(surface, score, final_time, game_width, game_height, go_font, score_lbl_font, time_lbl_font, restart_msg_font):
    """Draws the game over overlay message in the game area."""
    overlay = pygame.Surface((game_width, game_height), pygame.SRCALPHA); overlay.fill((*BG_COLOR, 220)); surface.blit(overlay, (0, 0))
    msg_width = 500; msg_height = 250
    msg_box_rect = pygame.Rect((game_width - msg_width) // 2, (game_height - msg_height) // 2, msg_width, msg_height)
    pygame.draw.rect(surface, CAM_BG, msg_box_rect, border_radius=0); pygame.draw.rect(surface, ERROR_COLOR, msg_box_rect, width=2, border_radius=0)
    try:
        message_text = go_font.render(":: INTEGRITY COMPROMISED ::", True, ERROR_COLOR); score_text = score_lbl_font.render(f"FINAL_SCORE: {int(score)}", True, SCORE_TEXT_COLOR)
        time_str = f"{final_time:.1f} SECONDS"; time_text = time_lbl_font.render(f"SURVIVAL_TIME: {time_str}", True, TIMER_TEXT_COLOR)
        restart_text = restart_msg_font.render("OPEN [RE-INITIATE] | FIST [MENU]", True, TEXT_COLOR) # FIST returns to menu
        msg_rect = message_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 70)); score_rect = score_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 25))
        time_rect = time_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 20)); restart_rect = restart_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 70))
        surface.blit(message_text, msg_rect); surface.blit(score_text, score_rect); surface.blit(time_text, time_rect); surface.blit(restart_text, restart_rect);
    except pygame.error as e: print(f"Error drawing game over message: {e}")

# --- Countdown Function (Corrected & Simplified) ---
def run_countdown(surface, pixel_surface, game_width, game_height, total_width, total_height, cam_panel_width, target_cam_width, target_cam_height, count_font, cap, mp_hands, hands, mp_drawing, landmark_spec, connection_spec):
    """Runs the 3-2-1 countdown animation with simplified drawing."""
    clock = pygame.time.Clock()
    for i in range(3, -1, -1):
        text_str = str(i) if i > 0 else "INITIATE"; color = HIGHLIGHT_COLOR if i > 0 else SCORE_TEXT_COLOR
        try:
            rendered_text = count_font.render(text_str, True, color)
            count_rect = rendered_text.get_rect(center=(game_width // 2, game_height // 2))
        except pygame.error as e: print(f"Error rendering countdown text '{text_str}': {e}"); continue
        start_time = time.time(); duration = 0.8; pause = 0.2
        while time.time() < start_time + duration + pause:
            elapsed = time.time() - start_time; alpha = 0
            if elapsed < duration / 2: alpha = int(255 * (elapsed / (duration / 2)))
            elif elapsed < duration / 2 + pause: alpha = 255
            elif elapsed < duration + pause: alpha = int(255 * (1 - ((elapsed - duration/2 - pause) / (duration/2))))
            alpha = max(0, min(255, alpha));
            # Redraw Full Scene + Fading Text
            pixel_surface.fill(BG_COLOR); draw_hacker_background_pixelated(pixel_surface)
            scaled_game_bg = pygame.transform.scale(pixel_surface, (game_width, game_height))
            surface.fill(BG_COLOR); surface.blit(scaled_game_bg, (0, 0))
            cam_area_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_area_rect); pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_area_rect, width=1)
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                try:
                    aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
                    small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB)
                    pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB");
                    cam_panel_center_x = game_width + cam_panel_width // 2; cam_display_area_y_start = 85; cam_display_area_height = total_height - cam_display_area_y_start; cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
                    frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y))
                    surface.blit(pygame_frame, frame_rect.topleft)
                except Exception as resize_err: print(f"Error resizing/blit camera frame: {resize_err}")
            try:
                rendered_text.set_alpha(alpha); surface.blit(rendered_text, count_rect)
            except pygame.error as e: print(f"Error blitting countdown text: {e}")
            pygame.display.flip(); clock.tick(60)

# --- Hand Tracking Functions ---
def get_hand_target_pos(frame, results, mp_hands_instance, cam_width, cam_height):
    target_x_norm = None; target_y_norm = None; finger_line = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try:
            tip = hand_landmarks.landmark[mp_hands_instance.HandLandmark.INDEX_FINGER_TIP]
            target_x_norm = max(0.0, min(1.0, tip.x)); target_y_norm = max(0.0, min(1.0, tip.y))
            mcp = hand_landmarks.landmark[mp_hands_instance.HandLandmark.INDEX_FINGER_MCP]
            start_pt = (int(mcp.x * cam_width), int(mcp.y * cam_height)); end_pt = (int(tip.x * cam_width), int(tip.y * cam_height))
            finger_line = (start_pt, end_pt);
        except (IndexError, AttributeError): pass
    return target_x_norm, target_y_norm, finger_line

def get_hand_gesture(frame, results, mp_hands_instance):
    gesture = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
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
def run_slope(surface): # <<<--- RENAMED FUNCTION
    """Runs the Matrix Dodge game. Accepts the main screen surface.
       Returns next state ('game_select' or 'quit')."""
    print("--- Initializing Matrix Dodge ---")
    clock = pygame.time.Clock()

    # --- Load Fonts within the function ---
    game_font, score_font, timer_font, countdown_font, game_over_font, restart_font = None, None, None, None, None, None
    try:
        if not pygame.font.get_init(): pygame.font.init() # Ensure font module ready
        try:
            font_path = "consola.ttf" # <<<--- CHANGE FONT FILENAME IF NEEDED
            print(f"Attempting to load font: {font_path}")
            game_font = pygame.font.Font(font_path, 24); score_font = pygame.font.Font(font_path, 30);
            timer_font = pygame.font.Font(font_path, 28); countdown_font = pygame.font.Font(font_path, 90)
            game_over_font = pygame.font.Font(font_path, 40); restart_font = pygame.font.Font(font_path, 26)
            print(f"Loaded bundled font '{font_path}' successfully.")
        except pygame.error as e:
             print(f"Warning: Error loading bundled font '{font_path}': {e}. Trying SysFont.")
             HACKER_FONT_NAME = "Consolas, Lucida Console, Courier New, monospace"
             game_font = pygame.font.SysFont(HACKER_FONT_NAME, 24); score_font = pygame.font.SysFont(HACKER_FONT_NAME, 30, bold=True);
             timer_font = pygame.font.SysFont(HACKER_FONT_NAME, 28, bold=True); countdown_font = pygame.font.SysFont(HACKER_FONT_NAME, 90, bold=True)
             game_over_font = pygame.font.SysFont(HACKER_FONT_NAME, 40, bold=True); restart_font = pygame.font.SysFont(HACKER_FONT_NAME, 26)
             print("Using fallback SysFonts.")
    except Exception as e:
        print(f"Error loading SysFonts ({e}). Falling back to default.")
        try:
             if not pygame.font.get_init(): pygame.font.init()
             game_font = pygame.font.SysFont(None, 26); score_font = pygame.font.SysFont(None, 32); timer_font = pygame.font.SysFont(None, 30, bold=True)
             countdown_font = pygame.font.SysFont(None, 100, bold=True); game_over_font = pygame.font.SysFont(None, 55, bold=True); restart_font = pygame.font.SysFont(None, 28)
             print("Using absolute default fallback SysFonts.")
        except Exception as e2: print(f"FATAL: Could not load any fonts: {e2}"); return "quit"

    # --- Calculate Dimensions ---
    total_width = surface.get_width(); total_height = surface.get_height()
    cam_panel_width = DEFAULT_CAM_WIDTH; game_width = total_width - cam_panel_width; game_height = total_height
    pixel_width = game_width // PIXELATION_FACTOR; pixel_height = game_height // PIXELATION_FACTOR
    game_surface = pygame.Surface((pixel_width, pixel_height)) # Surface for pixelated drawing
    pixel_ball_radius = max(1, BALL_RADIUS_LOGICAL // PIXELATION_FACTOR)
    pixel_obstacle_w = max(1, OBSTACLE_WIDTH_LOGICAL // PIXELATION_FACTOR); pixel_obstacle_h = max(1, OBSTACLE_HEIGHT_LOGICAL // PIXELATION_FACTOR)

    # --- OpenCV and MediaPipe Setup ---
    cap = None; native_cam_width = 0; native_cam_height = 0; hands = None; mp_hands = None; mp_drawing = None; mp_drawing_styles = None; landmark_drawing_spec = None; connection_drawing_spec = None
    try:
        cap = cv2.VideoCapture(0);
        if not cap.isOpened(): raise IOError("Cannot open webcam")
        native_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); native_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Native camera resolution: {native_cam_width}x{native_cam_height}")
        mp_hands = mp.solutions.hands; hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles
        landmark_drawing_spec = mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=1, circle_radius=1); connection_drawing_spec = mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=1)
        print("Camera and MediaPipe initialized for Matrix Dodge.")
    except Exception as e: print(f"Error initializing camera or MediaPipe: {e}"); return "game_select"

    # --- Calculate Aspect-Corrected Camera Size ---
    target_cam_panel_width = DEFAULT_CAM_WIDTH; aspect_ratio = native_cam_height / native_cam_width if native_cam_width > 0 else 9/16
    target_cam_height = int(target_cam_panel_width * aspect_ratio); target_cam_width = target_cam_panel_width
    max_allowable_cam_height = total_height - 85
    if target_cam_height > max_allowable_cam_height:
        target_cam_height = max_allowable_cam_height; target_cam_width = int(target_cam_height / aspect_ratio) if aspect_ratio > 0 else target_cam_panel_width
    print(f"Target aspect-corrected camera display size: {target_cam_width}x{target_cam_height}")

    # --- Game Restart Loop ---
    while True:
        player_x = game_width // 2; player_y = game_height // 2; target_player_x = player_x; target_player_y = player_y
        scroll_speed = INITIAL_SCROLL_SPEED; score = 0; game_over = False
        obstacles = deque(); last_obstacle_spawn_time = 0; next_obstacle_spawn_delay = random.uniform(OBSTACLE_SPAWN_INTERVAL_MIN, OBSTACLE_SPAWN_INTERVAL_MAX)
        game_time = 0.0
        last_drawn_scaled_game_surface = None # Store last frame for game over

        # --- Run countdown (Corrected version) ---
        run_countdown(surface, game_surface, game_width, game_height, total_width, total_height, cam_panel_width, target_cam_width, target_cam_height, countdown_font, cap, mp_hands, hands, mp_drawing, landmark_drawing_spec, connection_drawing_spec)
        game_start_time = time.time(); last_update_time = game_start_time; last_obstacle_spawn_time = game_start_time

        # --- Gameplay Loop ---
        while not game_over:
            current_time = time.time(); delta_time = min(0.1, current_time - last_update_time); last_update_time = current_time; game_time += delta_time
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if cap and cap.isOpened(): cap.release(); print("Camera released on QUIT.")
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if cap and cap.isOpened(): cap.release(); print("Camera released on ESC.")
                        return "game_select"
            # Hand Tracking
            success, frame = cap.read();
            if not success: print("Warning: Failed to read frame."); time.sleep(0.05); continue
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True
            hand_norm_x, hand_norm_y, finger_line_pts = get_hand_target_pos(frame, results, mp_hands, target_cam_width, target_cam_height)
            if hand_norm_x is not None: target_player_x = hand_norm_x * game_width; target_player_y = hand_norm_y * game_height
            # Game Logic
            player_x = lerp(player_x, target_player_x, delta_time * SMOOTHING_SPEED); player_y = lerp(player_y, target_player_y, delta_time * SMOOTHING_SPEED)
            player_x = max(BALL_RADIUS_LOGICAL, min(game_width - BALL_RADIUS_LOGICAL, player_x)); player_y = max(BALL_RADIUS_LOGICAL, min(game_height - BALL_RADIUS_LOGICAL, player_y))
            target_speed = min(MAX_SCROLL_SPEED, INITIAL_SCROLL_SPEED + (game_time * TARGET_SPEED_TIME_FACTOR)); scroll_speed = lerp(scroll_speed, target_speed, delta_time * SPEED_LERP_FACTOR)
            scroll_amount = scroll_speed * delta_time * 60 # Scale speed based on assumed frame rate
            for obs_rect in obstacles: obs_rect.y += scroll_amount
            if obstacles and obstacles[0].y > game_height: obstacles.popleft()
            if current_time - last_obstacle_spawn_time > next_obstacle_spawn_delay and len(obstacles) < OBSTACLE_MAX_COUNT:
                 last_obstacle_spawn_time = current_time; next_obstacle_spawn_delay = random.uniform(OBSTACLE_SPAWN_INTERVAL_MIN, OBSTACLE_SPAWN_INTERVAL_MAX)
                 obs_x = random.randint(0, game_width - OBSTACLE_WIDTH_LOGICAL); obs_y = -OBSTACLE_HEIGHT_LOGICAL
                 obstacles.append(pygame.Rect(obs_x, obs_y, OBSTACLE_WIDTH_LOGICAL, OBSTACLE_HEIGHT_LOGICAL))
            player_collision_rect = pygame.Rect(player_x - BALL_RADIUS_LOGICAL, player_y - BALL_RADIUS_LOGICAL, BALL_RADIUS_LOGICAL*2, BALL_RADIUS_LOGICAL*2)
            if any(player_collision_rect.colliderect(obs_rect) for obs_rect in obstacles): game_over = True; print(f"Game Over! Score: {int(score)}, Time: {game_time:.1f}s")
            if not game_over: score += scroll_speed * delta_time * 0.5
            # Drawing
            game_surface.fill(BG_COLOR); draw_hacker_background_pixelated(game_surface)
            draw_obstacles_pixelated(game_surface, obstacles, pixel_obstacle_w, pixel_obstacle_h, PIXELATION_FACTOR)
            draw_player_pixelated(game_surface, player_x, player_y, pixel_ball_radius, PIXELATION_FACTOR)
            scaled_game_surface = pygame.transform.scale(game_surface, (game_width, game_height))
            last_drawn_scaled_game_surface = scaled_game_surface # Store for game over
            surface.fill(BG_COLOR); surface.blit(scaled_game_surface, (0, 0))
            cam_panel_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_panel_rect)
            display_score(surface, score, total_width, cam_panel_width, game_font, score_font); display_timer(surface, game_time, total_width, cam_panel_width, game_font, timer_font)
            # Camera Feed Drawing
            if native_cam_width > 0: aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
            else: aspect_corrected_frame = cv2.resize(frame, (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), interpolation=cv2.INTER_AREA)
            if results.multi_hand_landmarks: mp_drawing.draw_landmarks(aspect_corrected_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)
            small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB")
            cam_panel_center_x = game_width + cam_panel_width // 2; cam_display_area_y_start = 85; cam_display_area_height = total_height - cam_display_area_y_start; cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
            frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y)); surface.blit(pygame_frame, frame_rect.topleft)
            pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_panel_rect, width=1) # Border
            pygame.display.flip(); clock.tick(60)
        # --- End of Gameplay Loop ---

        # --- Game Over State ---
        restart_attempt = False; gesture_check_start_time = time.time(); final_game_time = game_time
        print("Entering Game Over state...")
        # Draw initial game over screen using the last captured game frame
        if last_drawn_scaled_game_surface: surface.blit(last_drawn_scaled_game_surface, (0, 0))
        else: surface.fill(BG_COLOR) # Fallback
        # Draw static UI elements over it
        cam_panel_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_panel_rect); pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_panel_rect, width=1)
        display_score(surface, score, total_width, cam_panel_width, game_font, score_font); display_timer(surface, final_game_time, total_width, cam_panel_width, game_font, timer_font)
        # Draw game over message overlay
        game_over_message(surface, score, final_game_time, game_width, game_height, game_over_font, game_font, game_font, restart_font)
        pygame.display.flip() # Show initial game over screen

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
                    # --- Redraw only camera area + game over overlay ---
                    cam_rect_for_redraw = pygame.Rect(game_width, 0, cam_panel_width, total_height)
                    pygame.draw.rect(surface, CAM_BG, cam_rect_for_redraw) # Camera BG
                    if native_cam_width > 0: aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
                    else: aspect_corrected_frame = cv2.resize(frame, (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), interpolation=cv2.INTER_AREA)
                    if results.multi_hand_landmarks: mp_drawing.draw_landmarks(aspect_corrected_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)
                    if gesture: cv2.putText(aspect_corrected_frame, gesture, (10, target_cam_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
                    small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB")
                    cam_panel_center_x = game_width + cam_panel_width // 2; cam_display_area_y_start = 85; cam_display_area_height = total_height - cam_display_area_y_start; cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
                    frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y)); surface.blit(pygame_frame, frame_rect.topleft)
                    pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_rect_for_redraw, width=1) # Border
                    # Re-blit game over message overlay (on top of static game frame and updated camera)
                    game_over_message(surface, score, final_game_time, game_width, game_height, game_over_font, game_font, game_font, restart_font)
                    pygame.display.flip() # Update screen
                # Check Gestures AFTER drawing
                if gesture == "OPEN": print("Gesture: OPEN - Re-initiating..."); game_over = False; restart_attempt = True; break
                elif gesture == "FIST": print("Gesture: FIST - Returning to menu...");
                if cap and cap.isOpened(): cap.release(); print("Camera released on FIST gesture.")
                return "game_select" # Return state to main loop
            clock.tick(15) # Lower tick rate
        # End of Game Over Loop
        if not restart_attempt:
             print("Game Over loop exited without restart signal.");
             if cap and cap.isOpened(): cap.release(); print("Camera released on Game Over exit.")
             return "game_select"
        print("--- Restarting game... ---"); time.sleep(0.5)
    # --- End of Outer Restart Loop ---
    print("Exiting run_dodge function unexpectedly.");
    if cap and cap.isOpened(): cap.release(); print("Camera released on unexpected exit.")
    return "game_select"

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("Running Matrix Dodge standalone for testing...")
    pygame.init(); pygame.font.init()
    test_screen_width = DEFAULT_GAME_WIDTH + DEFAULT_CAM_WIDTH; test_screen_height = max(DEFAULT_GAME_HEIGHT, DEFAULT_CAM_HEIGHT)
    test_screen = pygame.display.set_mode((test_screen_width, test_screen_height))
    pygame.display.set_caption("Matrix Dodge Standalone Test")
    result = run_slope(test_screen) # Call main function
    print(f"Game exited with state: {result}")
    pygame.quit(); sys.exit()