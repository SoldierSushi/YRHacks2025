import pygame
import sys
import cv2
import mediapipe as mp
import random
import math
import time
from collections import deque

# --- Pygame Setup ---
pygame.init()
pygame.font.init()

# --- Screen Dimensions ---
WIDTH, HEIGHT = 600, 500      # Logical game area size
CAM_WIDTH, CAM_HEIGHT = 360, 270 # Camera view size

# --- Pixelation Setup ---
PIXELATION_FACTOR = 5
PIXEL_WIDTH = WIDTH // PIXELATION_FACTOR
PIXEL_HEIGHT = HEIGHT // PIXELATION_FACTOR

TOTAL_WIDTH = WIDTH + CAM_WIDTH
TOTAL_HEIGHT = max(HEIGHT, CAM_HEIGHT)

screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
game_surface = pygame.Surface((PIXEL_WIDTH, PIXEL_HEIGHT))

pygame.display.set_caption("::: MATRIX_DODGE_v2.1 :::") # Updated version

# --- Matrix Theme Style Elements ---
# Colors (Same as before)
MATRIX_GREEN_BRIGHT = (20, 255, 90); MATRIX_GREEN_MED = (10, 200, 70); MATRIX_GREEN_DARK = (0, 100, 30)
BG_COLOR = (0, 5, 0); GRID_COLOR = (0, 40, 10); SCANLINE_COLOR = (0, 30, 5)
BALL_COLOR = MATRIX_GREEN_BRIGHT; BALL_OUTLINE = MATRIX_GREEN_MED
OBSTACLE_COLOR = (255, 0, 0); OBSTACLE_OUTLINE = (150, 0, 0)
TEXT_COLOR = MATRIX_GREEN_MED; SCORE_TEXT_COLOR = MATRIX_GREEN_BRIGHT; SCORE_LABEL_COLOR = TEXT_COLOR
HIGHLIGHT_COLOR = (180, 255, 180); ERROR_COLOR = (255, 60, 60)
CAM_BG = (0, 20, 5); CAM_BORDER_COLOR = MATRIX_GREEN_DARK
LANDMARK_COLOR = MATRIX_GREEN_BRIGHT; CONNECTION_COLOR = MATRIX_GREEN_DARK; DIRECTION_LINE_COLOR = HIGHLIGHT_COLOR
# <<< RE-ADDED: Timer Colors >>>
TIMER_TEXT_COLOR = HIGHLIGHT_COLOR
TIMER_LABEL_COLOR = TEXT_COLOR
# <<< END RE-ADDED >>>


# Fonts
try:
    HACKER_FONT_NAME = "Consolas, Lucida Console, Courier New, monospace"
    game_font = pygame.font.SysFont(HACKER_FONT_NAME, 24); score_font = pygame.font.SysFont(HACKER_FONT_NAME, 30, bold=True)
    # <<< RE-ADDED: Timer Font >>>
    timer_font = pygame.font.SysFont(HACKER_FONT_NAME, 28, bold=True)
    # <<< END RE-ADDED >>>
    countdown_font = pygame.font.SysFont(HACKER_FONT_NAME, 90, bold=True); game_over_font = pygame.font.SysFont(HACKER_FONT_NAME, 40, bold=True)
    restart_font = pygame.font.SysFont(HACKER_FONT_NAME, 26)
except Exception as e:
    print(f"Warning: Font loading error ({e}). Falling back to default.")
    game_font = pygame.font.SysFont(None, 26); score_font = pygame.font.SysFont(None, 32)
    # <<< RE-ADDED: Timer Font Fallback >>>
    timer_font = pygame.font.SysFont(None, 30, bold=True)
    # <<< END RE-ADDED >>>
    countdown_font = pygame.font.SysFont(None, 100, bold=True); game_over_font = pygame.font.SysFont(None, 55, bold=True)
    restart_font = pygame.font.SysFont(None, 28)

# --- Game Variables ---
BALL_RADIUS_LOGICAL = 14; BALL_RADIUS_PIXEL = max(1, BALL_RADIUS_LOGICAL // PIXELATION_FACTOR)
SMOOTHING_SPEED = 6.0
INITIAL_SCROLL_SPEED = 3.0; MAX_SCROLL_SPEED = 9.0
TARGET_SPEED_TIME_FACTOR = 0.15; SPEED_LERP_FACTOR = 0.8
OBSTACLE_WIDTH_LOGICAL = 40; OBSTACLE_HEIGHT_LOGICAL = 40
OBSTACLE_WIDTH_PIXEL = max(1, OBSTACLE_WIDTH_LOGICAL // PIXELATION_FACTOR)
OBSTACLE_HEIGHT_PIXEL = max(1, OBSTACLE_HEIGHT_LOGICAL // PIXELATION_FACTOR)
OBSTACLE_SPAWN_INTERVAL_MIN = 0.4; OBSTACLE_SPAWN_INTERVAL_MAX = 1.2
OBSTACLE_MAX_COUNT = 30
SCANLINE_HEIGHT = 1; SCANLINE_ALPHA = 20
clock = pygame.time.Clock()

# --- OpenCV and MediaPipe Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Could not open webcam."); sys.exit()
mp_hands = mp.solutions.hands; hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles
landmark_drawing_spec = mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=1, circle_radius=1)
connection_drawing_spec = mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=1)

# --- Helper Functions ---
def lerp(a, b, t): return a + (b - a) * t

# --- Drawing Functions (Pixelated for Game Area) ---
def draw_hacker_background_pixelated(target_surface):
    pixel_w = target_surface.get_width(); pixel_h = target_surface.get_height()
    scanline_surface = pygame.Surface((pixel_w, SCANLINE_HEIGHT), pygame.SRCALPHA); scanline_surface.fill((*SCANLINE_COLOR, SCANLINE_ALPHA))
    for y in range(0, pixel_h, SCANLINE_HEIGHT * 2): target_surface.blit(scanline_surface, (0, y))

def draw_obstacles_pixelated(target_surface, obstacles):
    pixel_h = target_surface.get_height()
    for obs_rect in obstacles:
        x_pixel = obs_rect.x // PIXELATION_FACTOR; y_pixel = obs_rect.y // PIXELATION_FACTOR
        if y_pixel > pixel_h or y_pixel + OBSTACLE_HEIGHT_PIXEL < 0: continue
        ob_rect_pixel = pygame.Rect(x_pixel, y_pixel, OBSTACLE_WIDTH_PIXEL, OBSTACLE_HEIGHT_PIXEL)
        pygame.draw.rect(target_surface, OBSTACLE_COLOR, ob_rect_pixel)

def draw_player_pixelated(target_surface, player_x, player_y):
    player_x_pixel = int(player_x) // PIXELATION_FACTOR
    player_y_pixel = int(player_y) // PIXELATION_FACTOR
    pygame.draw.circle(target_surface, BALL_COLOR, (player_x_pixel, player_y_pixel), BALL_RADIUS_PIXEL)

# --- UI Drawing Functions ---
# <<< MODIFIED: Score Display slightly adjusted to make space >>>
def display_score(score):
    """Displays score top-right in camera area."""
    score_area_width = (CAM_WIDTH // 2) - 20 # Use roughly half the width
    score_area_height = 60
    score_area_x = WIDTH + CAM_WIDTH - score_area_width - 15 # Align right
    score_area_y = 15
    score_area_rect = pygame.Rect(score_area_x, score_area_y, score_area_width, score_area_height)
    pygame.draw.rect(screen, CAM_BG, score_area_rect, border_radius=3); pygame.draw.rect(screen, CAM_BORDER_COLOR, score_area_rect, width=1, border_radius=3)
    score_label_text = game_font.render("SCORE", True, SCORE_LABEL_COLOR) # Simpler label
    score_value_text = score_font.render(f"{int(score):04}", True, SCORE_TEXT_COLOR)
    screen.blit(score_label_text, (score_area_rect.centerx - score_label_text.get_width() // 2, score_area_rect.y + 8))
    screen.blit(score_value_text, (score_area_rect.centerx - score_value_text.get_width() // 2, score_area_rect.y + 28))
# <<< END MODIFIED >>>

# <<< RE-ADDED: Timer Display function >>>
def display_timer(elapsed_time):
    """Displays the elapsed game time top-left in camera area."""
    timer_area_width = (CAM_WIDTH // 2) - 20 # Use roughly half the width
    timer_area_height = 60
    timer_area_x = WIDTH + 15 # Align left
    timer_area_y = 15 # Same Y as score

    timer_area_rect = pygame.Rect(timer_area_x, timer_area_y, timer_area_width, timer_area_height)
    pygame.draw.rect(screen, CAM_BG, timer_area_rect, border_radius=3)
    pygame.draw.rect(screen, CAM_BORDER_COLOR, timer_area_rect, width=1, border_radius=3)

    time_str = f"{elapsed_time:.1f}s" # Seconds with one decimal place

    timer_label_text = game_font.render("TIME", True, TIMER_LABEL_COLOR)
    timer_value_text = timer_font.render(time_str, True, TIMER_TEXT_COLOR)

    screen.blit(timer_label_text, (timer_area_rect.centerx - timer_label_text.get_width() // 2, timer_area_rect.y + 8))
    screen.blit(timer_value_text, (timer_area_rect.centerx - timer_value_text.get_width() // 2, timer_area_rect.y + 28))
# <<< END RE-ADDED >>>

# <<< MODIFIED: Game Over Message includes final time >>>
def game_over_message(score, final_time):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((*BG_COLOR, 220)); screen.blit(overlay, (0, 0))
    msg_width = 500; msg_height = 250 # Slightly taller box
    msg_box_rect = pygame.Rect((WIDTH - msg_width) // 2, (HEIGHT - msg_height) // 2, msg_width, msg_height)
    pygame.draw.rect(screen, CAM_BG, msg_box_rect, border_radius=0); pygame.draw.rect(screen, ERROR_COLOR, msg_box_rect, width=2, border_radius=0)
    message_text = game_over_font.render(":: INTEGRITY COMPROMISED ::", True, ERROR_COLOR)
    score_text = game_font.render(f"FINAL_SCORE: {int(score)}", True, SCORE_TEXT_COLOR)
    time_str = f"{final_time:.1f} SECONDS"
    time_text = game_font.render(f"SURVIVAL_TIME: {time_str}", True, TIMER_TEXT_COLOR) # Use timer color
    restart_text = restart_font.render("OPEN [RE-INITIATE] | FIST [TERMINATE]", True, TEXT_COLOR)

    msg_rect = message_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 70)) # Adjust spacing
    score_rect = score_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 25))
    time_rect = time_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 20))
    restart_rect = restart_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 70))

    screen.blit(message_text, msg_rect); screen.blit(score_text, score_rect);
    screen.blit(time_text, time_rect) # Blit time
    screen.blit(restart_text, restart_rect); pygame.display.flip()
# <<< END MODIFIED >>>


def run_countdown():
    # (Same as before)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for i in range(3, -1, -1):
        game_surface.fill(BG_COLOR); draw_hacker_background_pixelated(game_surface); scaled_bg = pygame.transform.scale(game_surface, (WIDTH, HEIGHT)); screen.blit(scaled_bg, (0, 0))
        cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=1)
        text_str = str(i) if i > 0 else "INITIATE"; color = HIGHLIGHT_COLOR if i > 0 else SCORE_TEXT_COLOR
        overlay.fill((0, 0, 0, 0)); count_text = countdown_font.render(text_str, True, color); count_rect = count_text.get_rect(center=(WIDTH // 2, HEIGHT // 2)); overlay.blit(count_text, count_rect)
        start_time = time.time(); duration = 0.8; pause = 0.2
        while time.time() < start_time + duration + pause:
            elapsed = time.time() - start_time; alpha = 0
            if elapsed < duration / 2: alpha = int(255 * (elapsed / (duration / 2)))
            elif elapsed < duration / 2 + pause: alpha = 255
            elif elapsed < duration + pause: alpha = int(255 * (1 - ((elapsed - duration/2 - pause) / (duration/2))))
            alpha = max(0, min(255, alpha)); overlay.set_alpha(alpha)
            screen.blit(scaled_bg, (0, 0)); cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=1); screen.blit(overlay, (0,0))
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1); small_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT)); small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB");
                cam_y_offset = 85; cam_y_offset = max(cam_y_offset, (TOTAL_HEIGHT - CAM_HEIGHT) // 2); screen.blit(pygame_frame, (WIDTH + 2, cam_y_offset))
            pygame.display.flip(); clock.tick(60)

# --- Hand Tracking Functions ---
def get_hand_target_pos(frame, results):
    # (Same as before)
    target_x = None; target_y = None; finger_line = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try:
            tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]; target_x = max(0.0, min(1.0, tip.x)); target_y = max(0.0, min(1.0, tip.y))
            mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            start_pt = (int(mcp.x * CAM_WIDTH), int(mcp.y * CAM_HEIGHT)); end_pt = (int(tip.x * CAM_WIDTH), int(tip.y * CAM_HEIGHT))
            finger_line = (start_pt, end_pt); return target_x, target_y, finger_line
        except (IndexError, AttributeError): pass
    return None, None, None

def get_hand_gesture(frame, results):
     # (Same as before)
     gesture = None
     if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try:
            fingertips_ids=[mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
            palm_center_approx_id=mp_hands.HandLandmark.MIDDLE_FINGER_MCP; palm_center_pt = hand_landmarks.landmark[palm_center_approx_id]
            fingers_folded = 0; tip_threshold = 0.12
            for tip_id in fingertips_ids:
                tip_pt = hand_landmarks.landmark[tip_id]; distance = math.hypot(tip_pt.x - palm_center_pt.x, tip_pt.y - palm_center_pt.y)
                if distance < tip_threshold: fingers_folded += 1
            if fingers_folded >= 3: gesture = "FIST"
            else:
                 thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]; thumb_dist = math.hypot(thumb_tip.x - palm_center_pt.x, thumb_tip.y - palm_center_pt.y)
                 if thumb_dist > 0.15: gesture = "OPEN"
        except (IndexError, AttributeError): pass
     return gesture

# --- Main Game Loop ---
def game_loop():
    while True: # Outer loop for restarting
        # Initial Game State
        player_x = WIDTH // 2; player_y = HEIGHT // 2
        target_player_x = player_x; target_player_y = player_y
        scroll_speed = INITIAL_SCROLL_SPEED; score = 0; game_over = False
        obstacles = deque()
        last_obstacle_spawn_time = 0
        next_obstacle_spawn_delay = random.uniform(OBSTACLE_SPAWN_INTERVAL_MIN, OBSTACLE_SPAWN_INTERVAL_MAX)
        # <<< RE-ADDED: game_time variable >>>
        game_time = 0.0
        # <<< END RE-ADDED >>>

        run_countdown()
        game_start_time = time.time(); last_update_time = game_start_time
        last_obstacle_spawn_time = game_start_time

        # Gameplay Loop
        while not game_over:
            current_time = time.time(); delta_time = min(0.1, current_time - last_update_time)
            last_update_time = current_time
            # <<< RE-ADDED: Update game timer >>>
            game_time += delta_time
            # <<< END RE-ADDED >>>

            for event in pygame.event.get():
                if event.type == pygame.QUIT: cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()

            # Hand Tracking
            success, frame = cap.read();
            if not success: time.sleep(0.05); continue
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True

            # Get Target Position from Hand
            hand_norm_x, hand_norm_y, finger_line_pts = get_hand_target_pos(frame, results)
            if hand_norm_x is not None:
                target_player_x = hand_norm_x * WIDTH
                target_player_y = hand_norm_y * HEIGHT

            # Update Player X and Y (Smoothed)
            player_x = lerp(player_x, target_player_x, delta_time * SMOOTHING_SPEED)
            player_y = lerp(player_y, target_player_y, delta_time * SMOOTHING_SPEED)
            player_x = max(BALL_RADIUS_LOGICAL, min(WIDTH - BALL_RADIUS_LOGICAL, player_x))
            player_y = max(BALL_RADIUS_LOGICAL, min(HEIGHT - BALL_RADIUS_LOGICAL, player_y))

            # Update Scroll Speed based on time
            target_speed = INITIAL_SCROLL_SPEED + (game_time * TARGET_SPEED_TIME_FACTOR)
            target_speed = min(MAX_SCROLL_SPEED, target_speed)
            scroll_speed = lerp(scroll_speed, target_speed, delta_time * SPEED_LERP_FACTOR)

            # Update Obstacles (Scrolling Down)
            for obs_rect in obstacles: obs_rect.y += scroll_speed
            if obstacles and obstacles[0].y > HEIGHT: obstacles.popleft()

            # Obstacle Spawning
            if current_time - last_obstacle_spawn_time > next_obstacle_spawn_delay and len(obstacles) < OBSTACLE_MAX_COUNT:
                 last_obstacle_spawn_time = current_time
                 next_obstacle_spawn_delay = random.uniform(OBSTACLE_SPAWN_INTERVAL_MIN, OBSTACLE_SPAWN_INTERVAL_MAX)
                 obs_x = random.randint(0, WIDTH - OBSTACLE_WIDTH_LOGICAL)
                 obs_y = -OBSTACLE_HEIGHT_LOGICAL
                 new_obstacle = pygame.Rect(obs_x, obs_y, OBSTACLE_WIDTH_LOGICAL, OBSTACLE_HEIGHT_LOGICAL)
                 obstacles.append(new_obstacle)

            # Collision Detection (Player vs Obstacles)
            player_collision_rect = pygame.Rect(player_x - BALL_RADIUS_LOGICAL, player_y - BALL_RADIUS_LOGICAL, BALL_RADIUS_LOGICAL*2, BALL_RADIUS_LOGICAL*2)
            collided_obstacle = False
            for obs_rect in obstacles:
                 if player_collision_rect.colliderect(obs_rect): # Simple check now player can be anywhere
                     collided_obstacle = True; break
            if collided_obstacle: game_over = True; print(f"Game Over: Hit Obstacle")

            # Update Score
            if not game_over: score += scroll_speed * delta_time * 0.5

            # --- Drawing ---
            game_surface.fill(BG_COLOR); draw_hacker_background_pixelated(game_surface)
            draw_obstacles_pixelated(game_surface, obstacles)
            draw_player_pixelated(game_surface, player_x, player_y) # Pass both coords
            scaled_game_surface = pygame.transform.scale(game_surface, (WIDTH, HEIGHT))
            screen.fill(BG_COLOR); screen.blit(scaled_game_surface, (0, 0))
            cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=1)
            # <<< RE-ADDED: Call display_timer >>>
            display_score(score)
            display_timer(game_time)
            # <<< END RE-ADDED >>>
            small_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            if results.multi_hand_landmarks:
                 mp_drawing.draw_landmarks(small_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)
                 if finger_line_pts: cv2.line(small_frame, finger_line_pts[0], finger_line_pts[1], DIRECTION_LINE_COLOR, 2)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB");
            cam_y_offset = 85; cam_y_offset = max(cam_y_offset, (TOTAL_HEIGHT - CAM_HEIGHT) // 2)
            screen.blit(pygame_frame, (WIDTH + 2, cam_y_offset))
            pygame.display.flip(); clock.tick(60)

        # --- Game Over State ---
        # <<< RE-ADDED: Pass final game_time >>>
        game_over_message(score, game_time); time.sleep(0.5)
        # <<< END RE-ADDED >>>
        restart_attempt, quit_attempt = False, False; gesture_check_start_time = time.time()
        while game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: quit_attempt = True; break
            if quit_attempt: break
            if time.time() - gesture_check_start_time > 0.1:
                gesture_check_start_time = time.time()
                success, frame = cap.read()
                if success:
                    frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True
                    gesture = get_hand_gesture(frame, results)
                    small_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
                    if results.multi_hand_landmarks: mp_drawing.draw_landmarks(small_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)
                    if gesture: cv2.putText(small_frame, gesture, (10, CAM_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
                    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")
                    cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=1)
                    # <<< RE-ADDED: Display final score AND time >>>
                    display_score(score)
                    display_timer(game_time)
                    # <<< END RE-ADDED >>>
                    cam_y_offset = 85; cam_y_offset = max(cam_y_offset, (TOTAL_HEIGHT - CAM_HEIGHT) // 2)
                    screen.blit(pygame_frame, (WIDTH + 2, cam_y_offset)); pygame.display.flip()
                    # Keep OPEN=Restart, FIST=Quit
                    if gesture == "OPEN": print("Gesture: OPEN - Re-initiating..."); game_over = False; restart_attempt = True; break
                    elif gesture == "FIST": print("Gesture: FIST - Terminating..."); quit_attempt = True; break
            clock.tick(15)

        if quit_attempt: break
        if restart_attempt: time.sleep(0.5); continue

    # --- Cleanup ---
    print("/// PROCESS TERMINATED ///")
    cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()

# --- Start ---
if __name__ == '__main__':
    game_loop()