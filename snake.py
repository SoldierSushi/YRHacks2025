# snake.py (Modified for Integration with Menu - Aspect Ratio Fix)

import pygame
import sys
import cv2 # Make sure OpenCV is installed (pip install opencv-python)
import mediapipe as mp # Make sure Mediapipe is installed (pip install mediapipe)
import random
import math
import time
from collections import deque

# --- Constants ---
# Defaults used for calculations or standalone testing
DEFAULT_GAME_WIDTH, DEFAULT_GAME_HEIGHT = 600, 500
DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT = 360, 270

# Colors
DARK_BG = (10, 15, 25)
GRID_COLOR = (25, 35, 55)
SNAKE_EYE_COLOR = (255, 50, 50)
FOOD_COLOR_OUTER = (0, 255, 255)
FOOD_COLOR_INNER = (200, 255, 255)
BONUS_STAR_COLOR = (255, 220, 0)
SLOWDOWN_ORB_COLOR = (100, 100, 255)
SPIKE_COLOR = (255, 100, 0)
SPIKE_OUTLINE_COLOR = (150, 50, 0)
SPIKE_CORE_COLOR = (255, 200, 0)
TEXT_COLOR = (200, 210, 230)
SCORE_TEXT_COLOR = (100, 255, 180)
SCORE_LABEL_COLOR = TEXT_COLOR
HIGHLIGHT_COLOR = (255, 255, 100)
ERROR_COLOR = (255, 60, 60)
CAM_BG = (20, 30, 45)
CAM_BORDER_COLOR = (60, 80, 110)
LANDMARK_COLOR = (70, 170, 255)
CONNECTION_COLOR = (220, 220, 220)
DIRECTION_LINE_COLOR = (0, 255, 0)

# Game variables / Tuning parameters
BLOCK_SIZE = 20
FOOD_SIZE = int(BLOCK_SIZE * 1.5); FOOD_RADIUS_OUTER = FOOD_SIZE // 2; FOOD_RADIUS_INNER = FOOD_RADIUS_OUTER // 2
SNAKE_SPEED_INITIAL = 8
INITIAL_SNAKE_LENGTH = 2
SNAKE_BODY_RADIUS = 4; SNAKE_HEAD_RADIUS = 6
TAIL_FADE_LENGTH = 5; TAIL_FADE_START_ALPHA = 150
INITIAL_NUM_SPIKES = 3; MAX_SPIKES = 35; SPIKE_SPAWN_INTERVAL = 12.0; SPIKES_PER_SPAWN = 1
ITEM_LIFETIME = 8.0; BONUS_SPAWN_CHANCE = 0.005; BONUS_STAR_POINTS = 5
SLOWDOWN_DURATION = 5.0; SLOWDOWN_FACTOR = 0.6

# Item Type Enum/Constants
ITEM_BONUS_STAR = 1; ITEM_SLOWDOWN_ORB = 2

# --- Helper Functions ---
def get_random_bright_color():
    while True:
        r = random.randint(50, 255); g = random.randint(50, 255); b = random.randint(50, 255)
        if r + g + b > 250 and max(r,g,b) - min(r,g,b) > 30 : return (r, g, b)

def is_position_valid(pos, snake_body, spikes, bonus_items, food_pos, game_width, game_height, buffer_zone=0):
    x, y = pos
    for dx in range(-buffer_zone, buffer_zone + 1):
        for dy in range(-buffer_zone, buffer_zone + 1):
            check_pos = (x + dx * BLOCK_SIZE, y + dy * BLOCK_SIZE)
            if not (0 <= check_pos[0] < game_width and 0 <= check_pos[1] < game_height): return False
            if check_pos in snake_body: return False
            if check_pos in spikes: return False
            if food_pos and check_pos == food_pos: return False
            for item_data in bonus_items:
                 item_p = item_data[0]
                 if check_pos == item_p: return False
    return True

def generate_initial_obstacles(num_spikes, snake_body, food_pos, game_width, game_height):
    spikes = []; attempts = 0; max_attempts = num_spikes * 20
    while len(spikes) < num_spikes and attempts < max_attempts:
        attempts += 1; sx = random.randrange(0, game_width // BLOCK_SIZE) * BLOCK_SIZE; sy = random.randrange(0, game_height // BLOCK_SIZE) * BLOCK_SIZE
        pos = (sx, sy); start_area_buffer = BLOCK_SIZE * 6
        if abs(pos[0] - game_width//2) < start_area_buffer and abs(pos[1] - game_height//2) < start_area_buffer: continue
        if is_position_valid(pos, snake_body, spikes, [], food_pos, game_width, game_height, buffer_zone=0): spikes.append(pos)
    if len(spikes) < num_spikes: print(f"Warning: Could only generate {len(spikes)} initial spikes.")
    return spikes

def spawn_new_spikes(count, snake_body, spikes, bonus_items, food_pos, game_width, game_height, max_attempts=50):
    newly_spawned = []
    for _ in range(count):
        spawned = False
        for attempt in range(max_attempts):
            sx = random.randrange(0, game_width // BLOCK_SIZE) * BLOCK_SIZE; sy = random.randrange(0, game_height // BLOCK_SIZE) * BLOCK_SIZE
            pos = (sx, sy)
            if is_position_valid(pos, snake_body, spikes + newly_spawned, bonus_items, food_pos, game_width, game_height, buffer_zone=1):
                newly_spawned.append(pos); spawned = True; break
    return newly_spawned

# --- Drawing Functions ---
def draw_tech_background(surface, game_width, game_height):
    for x in range(0, game_width, BLOCK_SIZE * 2): pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, game_height), 1)
    for y in range(0, game_height, BLOCK_SIZE * 2): pygame.draw.line(surface, GRID_COLOR, (0, y), (game_width, y), 1)
    bracket_len = 20; bracket_thick = 2
    pygame.draw.lines(surface, CAM_BORDER_COLOR, False, [(0, bracket_len), (0,0), (bracket_len,0)], bracket_thick)
    pygame.draw.lines(surface, CAM_BORDER_COLOR, False, [(game_width-bracket_len, 0), (game_width,0), (game_width,bracket_len)], bracket_thick)
    pygame.draw.lines(surface, CAM_BORDER_COLOR, False, [(0, game_height-bracket_len), (0,game_height), (bracket_len,game_height)], bracket_thick)
    pygame.draw.lines(surface, CAM_BORDER_COLOR, False, [(game_width-bracket_len, game_height), (game_width,game_height), (game_width, game_height-bracket_len)], bracket_thick)

def draw_snake(surface, snake_body, ghost_tail, head_color, body_color):
    alpha_step = TAIL_FADE_START_ALPHA / (TAIL_FADE_LENGTH + 1)
    for i, segment_pos in enumerate(reversed(ghost_tail)):
        alpha = int(TAIL_FADE_START_ALPHA - (i * alpha_step)); alpha = max(0, alpha)
        try:
            ghost_surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(ghost_surface, (*body_color, alpha), ghost_surface.get_rect(), border_radius=SNAKE_BODY_RADIUS)
            surface.blit(ghost_surface, segment_pos)
        except (ValueError, pygame.error): pass
    for segment in snake_body[1:]: pygame.draw.rect(surface, body_color, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE), border_radius=SNAKE_BODY_RADIUS)
    head = snake_body[0]; head_rect = pygame.Rect(head[0], head[1], BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(surface, head_color, head_rect, border_radius=SNAKE_HEAD_RADIUS)
    eye_pos_x = head[0] + BLOCK_SIZE // 2; eye_pos_y = head[1] + BLOCK_SIZE // 2
    pygame.draw.circle(surface, SNAKE_EYE_COLOR, (eye_pos_x, eye_pos_y), BLOCK_SIZE // 5)

def draw_food(surface, food_pos):
    center_x = food_pos[0] + BLOCK_SIZE // 2; center_y = food_pos[1] + BLOCK_SIZE // 2
    pulse_factor = 0.8 + 0.2 * abs(math.sin(time.time() * 3))
    current_radius_outer = int(FOOD_RADIUS_OUTER * pulse_factor)
    glow_surface = pygame.Surface((FOOD_SIZE * 2, FOOD_SIZE * 2), pygame.SRCALPHA)
    for i in range(4, 0, -1):
         alpha = 150 // (i*i); alpha = max(0, min(255, alpha))
         try: pygame.draw.circle(glow_surface, (*FOOD_COLOR_OUTER, alpha), (FOOD_SIZE, FOOD_SIZE), current_radius_outer + i * 3)
         except (ValueError, pygame.error): pass
    surface.blit(glow_surface, (center_x - FOOD_SIZE, center_y - FOOD_SIZE))
    pygame.draw.circle(surface, FOOD_COLOR_OUTER, (center_x, center_y), current_radius_outer)
    pygame.draw.circle(surface, FOOD_COLOR_INNER, (center_x, center_y), FOOD_RADIUS_INNER)

def draw_spikes(surface, spike_list):
    num_points = 8; radius_outer = BLOCK_SIZE*0.5*0.95; radius_inner = BLOCK_SIZE*0.5*0.40; core_radius = BLOCK_SIZE*0.2
    for pos in spike_list:
        center_x = pos[0] + BLOCK_SIZE/2; center_y = pos[1] + BLOCK_SIZE/2; points = []
        for i in range(num_points):
            angle_deg = (i * 360 / num_points) - 90; angle_rad = math.radians(angle_deg)
            radius = radius_outer if i % 2 == 0 else radius_inner
            px = center_x + radius * math.cos(angle_rad); py = center_y + radius * math.sin(angle_rad)
            points.append((px, py))
        pygame.draw.polygon(surface, SPIKE_COLOR, points)
        pygame.draw.polygon(surface, SPIKE_OUTLINE_COLOR, points, width=1)
        pygame.draw.circle(surface, SPIKE_CORE_COLOR, (center_x, center_y), core_radius)

def draw_bonus_items(surface, bonus_items_list):
    current_time = time.time()
    for pos, item_type, spawn_time in bonus_items_list:
        center_x = pos[0] + BLOCK_SIZE // 2; center_y = pos[1] + BLOCK_SIZE // 2
        time_alive = current_time - spawn_time; life_ratio = max(0, 1.0 - (time_alive / ITEM_LIFETIME))
        alpha = int(255 * life_ratio); alpha = max(0, min(255, alpha));
        if alpha <= 10: continue
        if item_type == ITEM_BONUS_STAR:
             radius = BLOCK_SIZE*0.6; points = []; angle_offset = (current_time * 100) % 360; num_points = 5
             for i in range(num_points * 2):
                angle = math.radians(angle_offset + (i * 180 / num_points)); r = radius if i % 2 == 0 else radius * 0.5
                px = center_x + r * math.cos(angle); py = center_y + r * math.sin(angle); points.append((px, py))
             star_surface = pygame.Surface((BLOCK_SIZE*2, BLOCK_SIZE*2), pygame.SRCALPHA)
             rel_points = [(p[0]-pos[0]+BLOCK_SIZE/2, p[1]-pos[1]+BLOCK_SIZE/2) for p in points]
             try: pygame.draw.polygon(star_surface, (*BONUS_STAR_COLOR, alpha), rel_points)
             except ValueError: pass
             surface.blit(star_surface, (pos[0]-BLOCK_SIZE/2, pos[1]-BLOCK_SIZE/2))
        elif item_type == ITEM_SLOWDOWN_ORB:
            pulse_factor = 0.8 + 0.2 * abs(math.sin(current_time * 2.5 + 1)); current_radius = int((BLOCK_SIZE * 0.5) * pulse_factor)
            orb_surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            try: pygame.draw.circle(orb_surface, (*SLOWDOWN_ORB_COLOR, alpha), (BLOCK_SIZE // 2, BLOCK_SIZE // 2), current_radius)
            except ValueError: pass
            surface.blit(orb_surface, pos)

def display_score(surface, score, total_width, total_height, cam_panel_width, label_font, value_font):
    """Displays the score above the camera area."""
    score_area_width = cam_panel_width - 40
    score_area_height = 60
    score_area_x = (total_width - cam_panel_width) + (cam_panel_width - score_area_width) // 2
    score_area_y = 15
    score_area_rect = pygame.Rect(score_area_x, score_area_y, score_area_width, score_area_height)
    pygame.draw.rect(surface, CAM_BG, score_area_rect, border_radius=5)
    pygame.draw.rect(surface, CAM_BORDER_COLOR, score_area_rect, width=2, border_radius=5)
    try:
        score_label_text = label_font.render("SCORE", True, SCORE_LABEL_COLOR)
        score_value_text = value_font.render(f"{score:03}", True, SCORE_TEXT_COLOR)
        surface.blit(score_label_text, (score_area_rect.centerx - score_label_text.get_width() // 2, score_area_rect.y + 8))
        surface.blit(score_value_text, (score_area_rect.centerx - score_value_text.get_width() // 2, score_area_rect.y + 28))
    except pygame.error as e: print(f"Error displaying score: {e}")

def game_over_message(surface, score, game_width, game_height, go_font, score_lbl_font, restart_msg_font):
    """Draws the game over overlay message in the game area."""
    overlay = pygame.Surface((game_width, game_height), pygame.SRCALPHA); overlay.fill((*DARK_BG, 210)); surface.blit(overlay, (0, 0))
    msg_width = 450; msg_height = 220; msg_box_rect = pygame.Rect((game_width - msg_width) // 2, (game_height - msg_height) // 2, msg_width, msg_height)
    pygame.draw.rect(surface, CAM_BG, msg_box_rect, border_radius=10); pygame.draw.rect(surface, ERROR_COLOR, msg_box_rect, width=3, border_radius=10)
    try:
        message_text = go_font.render(":: SIGNAL LOST ::", True, ERROR_COLOR); score_text = score_lbl_font.render(f"FINAL SCORE: {score}", True, SCORE_TEXT_COLOR)
        restart_text = restart_msg_font.render("FIST [RETRY] | OPEN [MENU]", True, TEXT_COLOR)
        msg_rect = message_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 50)); score_rect = score_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery)); restart_rect = restart_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 50))
        surface.blit(message_text, msg_rect); surface.blit(score_text, score_rect); surface.blit(restart_text, restart_rect);
    except pygame.error as e: print(f"Error drawing game over message: {e}")

def run_countdown(surface, game_width, game_height, total_width, total_height, cam_panel_width, target_cam_height, count_font, cap, mp_hands, hands, mp_drawing, mp_drawing_styles):
    """Runs the 3-2-1 countdown animation."""
    clock = pygame.time.Clock()
    overlay = pygame.Surface((game_width, game_height), pygame.SRCALPHA)
    for i in range(3, -1, -1):
        surface.fill(DARK_BG); draw_tech_background(surface, game_width, game_height)
        cam_area_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_area_rect); pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_area_rect, width=2)
        text_str = str(i) if i > 0 else "ENGAGE!"; color = HIGHLIGHT_COLOR if i > 0 else SCORE_TEXT_COLOR
        overlay.fill((0, 0, 0, 0))
        try:
            count_text = count_font.render(text_str, True, color);
            count_rect = count_text.get_rect(center=(game_width // 2, game_height // 2));
            overlay.blit(count_text, count_rect)
        except pygame.error as e: print(f"Error rendering countdown text: {e}"); continue

        start_time = time.time(); duration = 0.8; pause = 0.2
        while time.time() < start_time + duration + pause:
            elapsed = time.time() - start_time; alpha = 0
            if elapsed < duration / 2: alpha = int(255 * (elapsed / (duration / 2)))
            elif elapsed < duration / 2 + pause: alpha = 255
            elif elapsed < duration + pause: alpha = int(255 * (1 - ((elapsed - duration/2 - pause) / (duration/2))))
            alpha = max(0, min(255, alpha)); overlay.set_alpha(alpha)
            # Redraw static parts
            surface.fill(DARK_BG); draw_tech_background(surface, game_width, game_height)
            cam_area_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_area_rect); pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_area_rect, width=2)
            # Blit countdown text overlay
            surface.blit(overlay, (0,0))

            # Update camera feed during countdown
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                # Resize using aspect-correct target height
                target_cam_width = int(target_cam_height * (frame.shape[1] / frame.shape[0])) if frame.shape[0] > 0 else DEFAULT_CAM_WIDTH
                aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
                small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB)
                pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB");

                # Center the potentially non-standard frame in the panel
                cam_panel_center_x = game_width + cam_panel_width // 2
                cam_display_area_y_start = 85
                cam_display_area_height = total_height - cam_display_area_y_start
                cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
                frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y))
                surface.blit(pygame_frame, frame_rect.topleft)

            pygame.display.flip(); clock.tick(60)

# --- Hand Tracking Function ---
def get_hand_gesture_and_direction(frame, results, mp_hands_instance, cam_width, cam_height):
    """Processes hand landmarks to find direction and gesture."""
    direction = None; gesture = None; finger_line = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try: # Gesture
            fingertips_ids = [mp_hands_instance.HandLandmark.INDEX_FINGER_TIP, mp_hands_instance.HandLandmark.MIDDLE_FINGER_TIP, mp_hands_instance.HandLandmark.RING_FINGER_TIP, mp_hands_instance.HandLandmark.PINKY_TIP]
            palm_center_approx_id = mp_hands_instance.HandLandmark.MIDDLE_FINGER_MCP; palm_center_pt = hand_landmarks.landmark[palm_center_approx_id]
            fingers_folded = 0; tip_threshold = 0.12
            for tip_id in fingertips_ids:
                tip_pt = hand_landmarks.landmark[tip_id]; distance = math.hypot(tip_pt.x - palm_center_pt.x, tip_pt.y - palm_center_pt.y)
                if distance < tip_threshold: fingers_folded += 1
            if fingers_folded >= 3: gesture = "FIST"
            else:
                 thumb_tip = hand_landmarks.landmark[mp_hands_instance.HandLandmark.THUMB_TIP]; thumb_dist = math.hypot(thumb_tip.x - palm_center_pt.x, thumb_tip.y - palm_center_pt.y)
                 if thumb_dist > 0.15: gesture = "OPEN"
        except (IndexError, AttributeError): pass
        try: # Direction
            mcp = hand_landmarks.landmark[mp_hands_instance.HandLandmark.INDEX_FINGER_MCP]; tip = hand_landmarks.landmark[mp_hands_instance.HandLandmark.INDEX_FINGER_TIP]
            dx = tip.x - mcp.x; dy = tip.y - mcp.y; orientation_threshold = 0.04
            start_pt = (int(mcp.x * cam_width), int(mcp.y * cam_height)); end_pt = (int(tip.x * cam_width), int(tip.y * cam_height)); finger_line = (start_pt, end_pt)
            if abs(dx) > abs(dy):
                if abs(dx) > orientation_threshold: direction = (1, 0) if dx > 0 else (-1, 0)
            elif abs(dy) > abs(dx):
                 if abs(dy) > orientation_threshold: direction = (0, -1) if dy < 0 else (0, 1)
        except (IndexError, AttributeError): pass
    return direction, gesture, finger_line

# --- Main Game Function ---
def run_snake(surface):
    """Runs the Snake game. Accepts the main screen surface.
       Returns next state ('game_select' or 'quit')."""
    print("--- Initializing Snake Game ---")
    clock = pygame.time.Clock()

    # --- Load Fonts within the function ---
    game_font, score_font, countdown_font, game_over_font, restart_font = None, None, None, None, None
    try:
        if not pygame.font.get_init(): pygame.font.init() # Ensure font module ready
        # --- TRY BUNDLED FONT FIRST ---
        try:
            font_path = "consola.ttf" # <--- CHANGE THIS to your font file name/path
            game_font = pygame.font.Font(font_path, 24)
            score_font = pygame.font.Font(font_path, 30)
            countdown_font = pygame.font.Font(font_path, 150)
            game_over_font = pygame.font.Font(font_path, 45)
            restart_font = pygame.font.Font(font_path, 26)
            print(f"Loaded bundled font '{font_path}' successfully.")
        except pygame.error as e:
             print(f"Warning: Error loading bundled font '{font_path}': {e}. Trying SysFont.")
             # Fallback to SysFont if bundled font fails
             MAIN_FONT_NAME = "Consolas, Courier New, Monaco, monospace" # Preferred SysFonts
             game_font = pygame.font.SysFont(MAIN_FONT_NAME, 24)
             score_font = pygame.font.SysFont(MAIN_FONT_NAME, 30, bold=True)
             countdown_font = pygame.font.SysFont(MAIN_FONT_NAME, 150, bold=True)
             game_over_font = pygame.font.SysFont(MAIN_FONT_NAME, 45, bold=True)
             restart_font = pygame.font.SysFont(MAIN_FONT_NAME, 26)
             print("Using fallback SysFonts.")

    except Exception as e:
        print(f"Error loading SysFonts ({e}). Falling back to default.")
        # Absolute fallback
        try:
             if not pygame.font.get_init(): pygame.font.init()
             game_font = pygame.font.SysFont(None, 26)
             score_font = pygame.font.SysFont(None, 32)
             countdown_font = pygame.font.SysFont(None, 150)
             game_over_font = pygame.font.SysFont(None, 55)
             restart_font = pygame.font.SysFont(None, 28)
             print("Using absolute default fallback SysFonts.")
        except Exception as e2:
             print(f"FATAL: Could not load any fonts: {e2}")
             # Cannot continue without fonts
             return "quit" # Critical error

    # Get dimensions from the passed surface
    total_width = surface.get_width()
    total_height = surface.get_height()
    # Camera panel width is fixed for layout
    cam_panel_width = DEFAULT_CAM_WIDTH
    game_width = total_width - cam_panel_width # Calculate game area width
    game_height = total_height # Game area uses full window height

    # --- OpenCV and MediaPipe Setup ---
    cap = None; native_cam_width = 0; native_cam_height = 0
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): raise IOError("Cannot open webcam")
        native_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Native camera resolution: {native_cam_width}x{native_cam_height}")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        print("Camera and MediaPipe initialized for Snake.")
    except Exception as e:
        print(f"Error initializing camera or MediaPipe: {e}")
        if cap and cap.isOpened(): cap.release()
        return "game_select"

    # --- Calculate Aspect-Corrected Camera Size ---
    target_cam_width = DEFAULT_CAM_WIDTH # Target panel width
    aspect_ratio = native_cam_height / native_cam_width if native_cam_width > 0 else 9/16
    target_cam_height = int(target_cam_width * aspect_ratio)
    # Make sure height isn't excessive for the panel (adjust width if necessary)
    max_allowable_cam_height = total_height - 90 # Leave space for score etc.
    if target_cam_height > max_allowable_cam_height:
        print("Warning: Calculated camera height too large, adjusting width based on height.")
        target_cam_height = max_allowable_cam_height
        target_cam_width = int(target_cam_height / aspect_ratio) if aspect_ratio > 0 else DEFAULT_CAM_WIDTH

    print(f"Target aspect-corrected camera size: {target_cam_width}x{target_cam_height}")

    # --- Game Restart Loop ---
    while True:
        # Initial Game State for each new game
        mid_x = (game_width // BLOCK_SIZE // 2) * BLOCK_SIZE; mid_y = (game_height // BLOCK_SIZE // 2) * BLOCK_SIZE
        snake_body = [ (mid_x - i * BLOCK_SIZE, mid_y) for i in range(INITIAL_SNAKE_LENGTH) ]; snake_body.reverse()
        direction = (1, 0); change_to = direction; score = 0; game_over = False
        ghost_tail = deque(maxlen=TAIL_FADE_LENGTH); bonus_items = []; spikes = []
        current_head_color = get_random_bright_color()
        current_body_color = get_random_bright_color()
        while math.dist(current_head_color, current_body_color) < 50: current_body_color = get_random_bright_color()
        slowdown_end_time = 0; current_speed_factor = 1.0
        last_spike_spawn_time = 0; game_start_time = 0

        # Place initial food & spikes
        while True:
            food_x = random.randrange(0, game_width // BLOCK_SIZE) * BLOCK_SIZE; food_y = random.randrange(0, game_height // BLOCK_SIZE) * BLOCK_SIZE
            food_pos = (food_x, food_y)
            if is_position_valid(food_pos, snake_body, spikes, bonus_items, None, game_width, game_height): break
        spikes = generate_initial_obstacles(INITIAL_NUM_SPIKES, snake_body, food_pos, game_width, game_height)

        # Run countdown
        run_countdown(surface, game_width, game_height, total_width, total_height, cam_panel_width, target_cam_height, countdown_font, cap, mp_hands, hands, mp_drawing, mp_drawing_styles)
        last_update_time = time.time(); game_start_time = last_update_time; last_spike_spawn_time = game_start_time

        # --- Gameplay Loop ---
        while not game_over:
            current_time = time.time()
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if cap and cap.isOpened(): cap.release(); print("Camera released on QUIT.")
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if cap and cap.isOpened(): cap.release(); print("Camera released on ESC.")
                        return "game_select"

            # --- Hand Tracking ---
            success, frame = cap.read();
            if not success: print("Warning: Failed to read frame."); time.sleep(0.05); continue
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True
            hand_direction, _, finger_line_pts = get_hand_gesture_and_direction(frame, results, mp_hands, target_cam_width, target_cam_height)

            # --- Update Direction ---
            if hand_direction:
                current_dx, current_dy = direction; new_dx, new_dy = hand_direction
                if not (current_dx == -new_dx and current_dx != 0) and not (current_dy == -new_dy and current_dy != 0): change_to = hand_direction

            # --- Update Speed ---
            if slowdown_end_time and current_time > slowdown_end_time: slowdown_end_time = 0; current_speed_factor = 1.0
            actual_speed = SNAKE_SPEED_INITIAL * current_speed_factor; snake_speed_delay = 1.0 / actual_speed if actual_speed > 0 else 1.0

            # --- Game Logic Update ---
            if current_time - last_update_time >= snake_speed_delay:
                last_update_time = current_time; direction = change_to
                head_x, head_y = snake_body[0]; new_head = (head_x + direction[0] * BLOCK_SIZE, head_y + direction[1] * BLOCK_SIZE)
                # Collision Check
                collision_type = None
                if not (0 <= new_head[0] < game_width and 0 <= new_head[1] < game_height): collision_type = "Wall"
                elif new_head in snake_body[1:]: collision_type = "Self"
                elif new_head in spikes: collision_type = "Spike"
                if collision_type: game_over = True; print(f"Collision: {collision_type}"); continue
                snake_body.insert(0, new_head) # Move
                # Item Interaction
                ate_something = False; previous_score = score; item_collected_idx = -1
                # Iterate safely over a copy if modifying list during iteration
                for i, item_data in enumerate(bonus_items):
                    item_pos, item_type, _ = item_data
                    if new_head == item_pos:
                        ate_something = True; item_collected_idx = i
                        if item_type == ITEM_BONUS_STAR: score += BONUS_STAR_POINTS; print(f"Collected: Star (+{BONUS_STAR_POINTS})")
                        elif item_type == ITEM_SLOWDOWN_ORB: slowdown_end_time = current_time + SLOWDOWN_DURATION; current_speed_factor = SLOWDOWN_FACTOR; print(f"Collected: Slowdown")
                        break # Found one, stop checking
                if item_collected_idx != -1: del bonus_items[item_collected_idx]
                # Food Interaction
                if not ate_something and new_head == food_pos:
                    ate_something = True; score += 1; print("Collected: Food (+1)")
                    while True:
                        food_x = random.randrange(0, game_width // BLOCK_SIZE) * BLOCK_SIZE; food_y = random.randrange(0, game_height // BLOCK_SIZE) * BLOCK_SIZE
                        food_pos = (food_x, food_y)
                        if is_position_valid(food_pos, snake_body, spikes, bonus_items, None, game_width, game_height): break
                # Color Change Trigger
                if score > 0 and score % 5 == 0 and score != previous_score:
                    current_head_color = get_random_bright_color()
                    current_body_color = get_random_bright_color()
                    while math.dist(current_head_color, current_body_color) < 50: current_body_color = get_random_bright_color()
                    print(f"Snake color randomized!")
                # Tail Management
                if not ate_something: ghost_tail.append(snake_body.pop())
                else: ghost_tail.clear()

            # --- Dynamic Spawning ---
            if current_time - last_spike_spawn_time > SPIKE_SPAWN_INTERVAL and len(spikes) < MAX_SPIKES:
                last_spike_spawn_time = current_time
                new_spikes = spawn_new_spikes(SPIKES_PER_SPAWN, snake_body, spikes, bonus_items, food_pos, game_width, game_height)
                if new_spikes: spikes.extend(new_spikes); print(f"Spawned {len(new_spikes)} spike(s). Total: {len(spikes)}")
            bonus_items = [(p, t, st) for p, t, st in bonus_items if current_time - st < ITEM_LIFETIME]
            if not bonus_items and random.random() < BONUS_SPAWN_CHANCE:
                 item_type = random.choice([ITEM_BONUS_STAR, ITEM_SLOWDOWN_ORB]); attempts = 0
                 while attempts < 50:
                     bx = random.randrange(0, game_width // BLOCK_SIZE) * BLOCK_SIZE; by = random.randrange(0, game_height // BLOCK_SIZE) * BLOCK_SIZE
                     bpos = (bx, by)
                     if is_position_valid(bpos, snake_body, spikes, bonus_items, food_pos, game_width, game_height, buffer_zone=1):
                         bonus_items.append((bpos, item_type, current_time)); print(f"Spawned: {'Star' if item_type == ITEM_BONUS_STAR else 'Slowdown'}"); break
                     attempts += 1

            # --- Drawing ---
            surface.fill(DARK_BG)
            draw_tech_background(surface, game_width, game_height)
            cam_panel_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_panel_rect);
            draw_spikes(surface, spikes)
            draw_food(surface, food_pos)
            draw_bonus_items(surface, bonus_items)
            draw_snake(surface, snake_body, ghost_tail, current_head_color, current_body_color)
            display_score(surface, score, total_width, total_height, cam_panel_width, game_font, score_font)

            # Draw Camera Feed (Aspect Corrected)
            if native_cam_width > 0:
                aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
            else:
                 aspect_corrected_frame = cv2.resize(frame, (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), interpolation=cv2.INTER_AREA)
            # Draw landmarks on resized (approximate)
            if results.multi_hand_landmarks:
                 mp_drawing.draw_landmarks(aspect_corrected_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                 # Scaling finger line is complex, skipping for stability
                 # if finger_line_pts: cv2.line(aspect_corrected_frame, scaled_start, scaled_end, DIRECTION_LINE_COLOR, 2)
            small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB)
            pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB")
            # Center frame in panel
            cam_panel_center_x = game_width + cam_panel_width // 2
            cam_display_area_y_start = 85; cam_display_area_height = total_height - cam_display_area_y_start
            cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
            frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y))
            surface.blit(pygame_frame, frame_rect.topleft)
            # Draw border last
            pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_panel_rect, width=2)

            pygame.display.flip(); clock.tick(60)
        # --- End of Gameplay Loop ---

        # --- Game Over State ---
        restart_attempt = False; gesture_check_start_time = time.time()
        print("Entering Game Over state...")
        # Draw initial game over screen
        surface.fill(DARK_BG); draw_tech_background(surface, game_width, game_height)
        cam_panel_rect = pygame.Rect(game_width, 0, cam_panel_width, total_height); pygame.draw.rect(surface, CAM_BG, cam_panel_rect); pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_panel_rect, width=2)
        display_score(surface, score, total_width, total_height, cam_panel_width, game_font, score_font)
        game_over_message(surface, score, game_width, game_height, game_over_font, game_font, restart_font)
        pygame.display.flip()

        while game_over: # Game over display loop
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if cap and cap.isOpened(): cap.release(); print("Camera released on QUIT.")
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if cap and cap.isOpened(): cap.release(); print("Camera released on ESC.")
                        return "game_select"
            # Gesture Check
            if time.time() - gesture_check_start_time > 0.1:
                gesture_check_start_time = time.time()
                success, frame = cap.read(); gesture = None
                if success:
                    frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True
                    _, gesture, finger_line_pts = get_hand_gesture_and_direction(frame, results, mp_hands, target_cam_width, target_cam_height)
                    # Redraw only camera area + game over overlay for efficiency
                    cam_rect_for_redraw = pygame.Rect(game_width, 0, cam_panel_width, total_height)
                    pygame.draw.rect(surface, CAM_BG, cam_rect_for_redraw) # Camera BG
                    # Aspect correct camera feed
                    if native_cam_width > 0: aspect_corrected_frame = cv2.resize(frame, (target_cam_width, target_cam_height), interpolation=cv2.INTER_AREA)
                    else: aspect_corrected_frame = cv2.resize(frame, (DEFAULT_CAM_WIDTH, DEFAULT_CAM_HEIGHT), interpolation=cv2.INTER_AREA)
                    # Draw landmarks
                    if results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(aspect_corrected_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                        # Skipping finger line scaling
                    if gesture: cv2.putText(aspect_corrected_frame, gesture, (10, target_cam_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
                    small_frame_rgb = cv2.cvtColor(aspect_corrected_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (target_cam_width, target_cam_height), "RGB")
                    # Center frame
                    cam_panel_center_x = game_width + cam_panel_width // 2; cam_display_area_y_start = 85; cam_display_area_height = total_height - cam_display_area_y_start; cam_panel_center_y = cam_display_area_y_start + cam_display_area_height // 2
                    frame_rect = pygame_frame.get_rect(center=(cam_panel_center_x, cam_panel_center_y))
                    surface.blit(pygame_frame, frame_rect.topleft) # Blit centered
                    pygame.draw.rect(surface, CAM_BORDER_COLOR, cam_rect_for_redraw, width=2) # Border
                    # Re-blit game over message overlay
                    game_over_message(surface, score, game_width, game_height, game_over_font, game_font, restart_font)
                    pygame.display.flip() # Update screen
                # Check Gestures AFTER drawing
                if gesture == "FIST":
                    print("Gesture: FIST - Retrying game..."); game_over = False; restart_attempt = True; break
                elif gesture == "OPEN":
                    print("Gesture: OPEN - Returning to menu...")
                    if cap and cap.isOpened(): cap.release(); print("Camera released on OPEN gesture.")
                    return "game_select"
            clock.tick(15) # Lower tick rate
        # End of Game Over Loop
        if not restart_attempt: # If exited via ESC or QUIT
             print("Game Over loop exited without restart signal.")
             if cap and cap.isOpened(): cap.release(); print("Camera released on Game Over exit.")
             return "game_select" # Should have already returned, but safety fallback
        print("--- Restarting game... ---")
        time.sleep(0.5)
    # --- End of Outer Restart Loop ---
    print("Exiting run_snake function unexpectedly.")
    if cap and cap.isOpened(): cap.release(); print("Camera released on unexpected exit.")
    return "game_select"

# --- Standalone Test Block ---
if __name__ == '__main__':
    print("Running Snake game standalone for testing...")
    pygame.init()
    pygame.font.init()
    test_screen_width = DEFAULT_GAME_WIDTH + DEFAULT_CAM_WIDTH
    test_screen_height = max(DEFAULT_GAME_HEIGHT, DEFAULT_CAM_HEIGHT)
    test_screen = pygame.display.set_mode((test_screen_width, test_screen_height))
    pygame.display.set_caption("Snake Standalone Test")
    result = run_snake(test_screen)
    print(f"Game exited with state: {result}")
    pygame.quit()
    sys.exit()