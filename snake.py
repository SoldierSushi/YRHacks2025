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
# No mixer needed for this version

# --- Adjust Sizes ---
WIDTH, HEIGHT = 600, 500
CAM_WIDTH, CAM_HEIGHT = 360, 270
TOTAL_WIDTH = WIDTH + CAM_WIDTH
TOTAL_HEIGHT = max(HEIGHT, CAM_HEIGHT) # Window height adapts

screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("Snake Arena - Random Colors")

# --- Style Elements ---
# Colors (Base colors remain)
DARK_BG = (10, 15, 25)
GRID_COLOR = (25, 35, 55)
# Snake colors are now generated randomly
SNAKE_EYE_COLOR = (255, 50, 50)
FOOD_COLOR_OUTER = (0, 255, 255)
FOOD_COLOR_INNER = (200, 255, 255)
BONUS_STAR_COLOR = (255, 220, 0)
SLOWDOWN_ORB_COLOR = (100, 100, 255)
SPIKE_COLOR = (255, 100, 0)
SPIKE_OUTLINE_COLOR = (150, 50, 0)
SPIKE_CORE_COLOR = (255, 200, 0)
TEXT_COLOR = (200, 210, 230)
SCORE_TEXT_COLOR = (100, 255, 180) # Color for score value text
SCORE_LABEL_COLOR = TEXT_COLOR    # Color for "SCORE" label
HIGHLIGHT_COLOR = (255, 255, 100)
ERROR_COLOR = (255, 60, 60)
CAM_BG = (20, 30, 45)
CAM_BORDER_COLOR = (60, 80, 110)
LANDMARK_COLOR = (70, 170, 255)
CONNECTION_COLOR = (220, 220, 220)
DIRECTION_LINE_COLOR = (0, 255, 0)

# Fonts
try:
    MAIN_FONT_NAME = "Consolas, Courier New, Monaco, monospace"
    game_font = pygame.font.SysFont(MAIN_FONT_NAME, 24)
    score_font = pygame.font.SysFont(MAIN_FONT_NAME, 30, bold=True)
    countdown_font = pygame.font.SysFont(MAIN_FONT_NAME, 150, bold=True)
    game_over_font = pygame.font.SysFont(MAIN_FONT_NAME, 45, bold=True)
    restart_font = pygame.font.SysFont(MAIN_FONT_NAME, 26)
except Exception as e:
    print(f"Warning: Font loading error ({e}). Falling back to default.")
    # Fallbacks...
    game_font = pygame.font.SysFont(None, 26)
    score_font = pygame.font.SysFont(None, 32)
    countdown_font = pygame.font.SysFont(None, 150, bold=True)
    game_over_font = pygame.font.SysFont(None, 55, bold=True)
    restart_font = pygame.font.SysFont(None, 28)

# Game variables
BLOCK_SIZE = 20
FOOD_SIZE = int(BLOCK_SIZE * 1.5); FOOD_RADIUS_OUTER = FOOD_SIZE // 2; FOOD_RADIUS_INNER = FOOD_RADIUS_OUTER // 2
SNAKE_SPEED_INITIAL = 8
INITIAL_SNAKE_LENGTH = 2
SNAKE_BODY_RADIUS = 4; SNAKE_HEAD_RADIUS = 6
TAIL_FADE_LENGTH = 5; TAIL_FADE_START_ALPHA = 150
INITIAL_NUM_SPIKES = 3; MAX_SPIKES = 35; SPIKE_SPAWN_INTERVAL = 12.0; SPIKES_PER_SPAWN = 1
ITEM_LIFETIME = 8.0; BONUS_SPAWN_CHANCE = 0.005; BONUS_STAR_POINTS = 5
SLOWDOWN_DURATION = 5.0; SLOWDOWN_FACTOR = 0.6
clock = pygame.time.Clock()

# --- OpenCV and MediaPipe Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Could not open webcam."); sys.exit()
mp_hands = mp.solutions.hands; hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles

# --- Item Type Enum/Constants ---
ITEM_BONUS_STAR = 1; ITEM_SLOWDOWN_ORB = 2

# --- Helper Functions ---

# <<< ADDED: Function to get a random visible color >>>
def get_random_bright_color():
    """Generates a random RGB color that is likely visible on a dark background."""
    while True:
        r = random.randint(50, 255) # Avoid very dark red
        g = random.randint(50, 255) # Avoid very dark green
        b = random.randint(50, 255) # Avoid very dark blue
        # Ensure minimum brightness (sum of components) and avoid pure greyish
        if r + g + b > 250 and max(r,g,b) - min(r,g,b) > 30 :
             return (r, g, b)
# <<< END ADDED >>>

def is_position_valid(pos, snake_body, spikes, bonus_items, food_pos, buffer_zone=0):
    x, y = pos
    for dx in range(-buffer_zone, buffer_zone + 1):
        for dy in range(-buffer_zone, buffer_zone + 1):
            check_pos = (x + dx * BLOCK_SIZE, y + dy * BLOCK_SIZE)
            if not (0 <= check_pos[0] < WIDTH and 0 <= check_pos[1] < HEIGHT): return False
            if check_pos in snake_body: return False
            if check_pos in spikes: return False
            if food_pos and check_pos == food_pos: return False
            for item_p, _, _ in bonus_items:
                if check_pos == item_p: return False
    return True

def generate_initial_obstacles(num_spikes, snake_body, food_pos):
    spikes = []; attempts = 0; max_attempts = num_spikes * 20
    while len(spikes) < num_spikes and attempts < max_attempts:
        attempts += 1; sx = random.randrange(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE; sy = random.randrange(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
        pos = (sx, sy); start_area_buffer = BLOCK_SIZE * 6
        if abs(pos[0] - WIDTH//2) < start_area_buffer and abs(pos[1] - HEIGHT//2) < start_area_buffer: continue
        if is_position_valid(pos, snake_body, spikes, [], food_pos, buffer_zone=0): spikes.append(pos)
    if len(spikes) < num_spikes: print(f"Warning: Could only generate {len(spikes)} initial spikes.")
    return spikes

def spawn_new_spikes(count, snake_body, spikes, bonus_items, food_pos, max_attempts=50):
    newly_spawned = []
    for _ in range(count):
        spawned = False
        for attempt in range(max_attempts):
            sx = random.randrange(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE; sy = random.randrange(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
            pos = (sx, sy)
            if is_position_valid(pos, snake_body, spikes + newly_spawned, bonus_items, food_pos, buffer_zone=1):
                newly_spawned.append(pos); spawned = True; break
    return newly_spawned

# --- Drawing Functions ---
def draw_tech_background():
    # (Same as before)
    for x in range(0, WIDTH, BLOCK_SIZE * 2): pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, BLOCK_SIZE * 2): pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y), 1)
    bracket_len = 20; bracket_thick = 2
    pygame.draw.lines(screen, CAM_BORDER_COLOR, False, [(0, bracket_len), (0,0), (bracket_len,0)], bracket_thick)
    pygame.draw.lines(screen, CAM_BORDER_COLOR, False, [(WIDTH-bracket_len, 0), (WIDTH,0), (WIDTH,bracket_len)], bracket_thick)
    pygame.draw.lines(screen, CAM_BORDER_COLOR, False, [(0, HEIGHT-bracket_len), (0,HEIGHT), (bracket_len,HEIGHT)], bracket_thick)
    pygame.draw.lines(screen, CAM_BORDER_COLOR, False, [(WIDTH-bracket_len, HEIGHT), (WIDTH,HEIGHT), (WIDTH, HEIGHT-bracket_len)], bracket_thick)

def draw_snake(snake_body, ghost_tail, head_color, body_color):
    # (Same as before - uses passed colors)
    alpha_step = TAIL_FADE_START_ALPHA / (TAIL_FADE_LENGTH + 1)
    for i, segment_pos in enumerate(reversed(ghost_tail)):
        alpha = int(TAIL_FADE_START_ALPHA - (i * alpha_step)); alpha = max(0, alpha)
        ghost_surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(ghost_surface, (*body_color, alpha), ghost_surface.get_rect(), border_radius=SNAKE_BODY_RADIUS)
        screen.blit(ghost_surface, segment_pos)
    for segment in snake_body[1:]: pygame.draw.rect(screen, body_color, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE), border_radius=SNAKE_BODY_RADIUS)
    head = snake_body[0]; head_rect = pygame.Rect(head[0], head[1], BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, head_color, head_rect, border_radius=SNAKE_HEAD_RADIUS)
    eye_pos_x = head[0] + BLOCK_SIZE // 2; eye_pos_y = head[1] + BLOCK_SIZE // 2
    pygame.draw.circle(screen, SNAKE_EYE_COLOR, (eye_pos_x, eye_pos_y), BLOCK_SIZE // 5)

def draw_food(food_pos):
    # (Same as before)
    center_x = food_pos[0] + BLOCK_SIZE // 2; center_y = food_pos[1] + BLOCK_SIZE // 2
    pulse_factor = 0.8 + 0.2 * abs(math.sin(time.time() * 3))
    current_radius_outer = int(FOOD_RADIUS_OUTER * pulse_factor)
    glow_surface = pygame.Surface((FOOD_SIZE * 2, FOOD_SIZE * 2), pygame.SRCALPHA)
    for i in range(4, 0, -1):
         alpha = 150 // (i*i); alpha = max(0, min(255, alpha))
         pygame.draw.circle(glow_surface, (*FOOD_COLOR_OUTER, alpha), (FOOD_SIZE, FOOD_SIZE), current_radius_outer + i * 3)
    screen.blit(glow_surface, (center_x - FOOD_SIZE, center_y - FOOD_SIZE))
    pygame.draw.circle(screen, FOOD_COLOR_OUTER, (center_x, center_y), current_radius_outer)
    pygame.draw.circle(screen, FOOD_COLOR_INNER, (center_x, center_y), FOOD_RADIUS_INNER)

def draw_spikes(spike_list):
    # (Same as before)
    num_points = 8; radius_outer = BLOCK_SIZE*0.5*0.95; radius_inner = BLOCK_SIZE*0.5*0.40; core_radius = BLOCK_SIZE*0.2
    for pos in spike_list:
        center_x = pos[0] + BLOCK_SIZE/2; center_y = pos[1] + BLOCK_SIZE/2; points = []
        for i in range(num_points):
            angle_deg = (i * 360 / num_points) - 90; angle_rad = math.radians(angle_deg)
            radius = radius_outer if i % 2 == 0 else radius_inner
            px = center_x + radius * math.cos(angle_rad); py = center_y + radius * math.sin(angle_rad)
            points.append((px, py))
        pygame.draw.polygon(screen, SPIKE_COLOR, points)
        pygame.draw.polygon(screen, SPIKE_OUTLINE_COLOR, points, width=1)
        pygame.draw.circle(screen, SPIKE_CORE_COLOR, (center_x, center_y), core_radius)

def draw_bonus_items(bonus_items_list):
    # (Same as before)
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
             pygame.draw.polygon(star_surface, (*BONUS_STAR_COLOR, alpha), rel_points); screen.blit(star_surface, (pos[0]-BLOCK_SIZE/2, pos[1]-BLOCK_SIZE/2))
        elif item_type == ITEM_SLOWDOWN_ORB:
            pulse_factor = 0.8 + 0.2 * abs(math.sin(current_time * 2.5 + 1)); current_radius = int((BLOCK_SIZE * 0.5) * pulse_factor)
            orb_surface = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(orb_surface, (*SLOWDOWN_ORB_COLOR, alpha), (BLOCK_SIZE // 2, BLOCK_SIZE // 2), current_radius); screen.blit(orb_surface, pos)

# <<< MODIFIED: Position score above camera >>>
def display_score(score):
    """Displays the score above the camera feed area."""
    # Define area for score above camera
    score_area_width = CAM_WIDTH - 40 # Leave some padding
    score_area_height = 60
    score_area_x = WIDTH + (CAM_WIDTH - score_area_width) // 2 # Center horizontally in cam area
    score_area_y = 15 # Position near the top

    score_area_rect = pygame.Rect(score_area_x, score_area_y, score_area_width, score_area_height)

    # Draw background panel and border for the score
    pygame.draw.rect(screen, CAM_BG, score_area_rect, border_radius=5)
    pygame.draw.rect(screen, CAM_BORDER_COLOR, score_area_rect, width=2, border_radius=5)

    # Render text
    score_label_text = game_font.render("SCORE", True, SCORE_LABEL_COLOR)
    score_value_text = score_font.render(f"{score:03}", True, SCORE_TEXT_COLOR) # Padded score

    # Position text inside the score area
    screen.blit(score_label_text, (score_area_rect.centerx - score_label_text.get_width() // 2, score_area_rect.y + 8))
    screen.blit(score_value_text, (score_area_rect.centerx - score_value_text.get_width() // 2, score_area_rect.y + 28))
# <<< END MODIFIED >>>


def game_over_message(score):
    # (Same as before)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((*DARK_BG, 210)); screen.blit(overlay, (0, 0))
    msg_width = 450; msg_height = 220; msg_box_rect = pygame.Rect((WIDTH - msg_width) // 2, (HEIGHT - msg_height) // 2, msg_width, msg_height)
    pygame.draw.rect(screen, CAM_BG, msg_box_rect, border_radius=10); pygame.draw.rect(screen, ERROR_COLOR, msg_box_rect, width=3, border_radius=10)
    message_text = game_over_font.render(":: SIGNAL LOST ::", True, ERROR_COLOR); score_text = game_font.render(f"FINAL SCORE: {score}", True, SCORE_TEXT_COLOR)
    restart_text = restart_font.render("FIST [RETRY] | OPEN [EXIT]", True, TEXT_COLOR)
    msg_rect = message_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery - 50)); score_rect = score_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery)); restart_rect = restart_text.get_rect(center=(msg_box_rect.centerx, msg_box_rect.centery + 50))
    screen.blit(message_text, msg_rect); screen.blit(score_text, score_rect); screen.blit(restart_text, restart_rect); pygame.display.flip()

def run_countdown():
    # (Same as before, but score display isn't drawn here)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for i in range(3, -1, -1):
        screen.fill(DARK_BG); draw_tech_background()
        # Draw Camera BG/Border
        cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=2)

        text_str = str(i) if i > 0 else "ENGAGE!"; color = HIGHLIGHT_COLOR if i > 0 else SCORE_TEXT_COLOR
        overlay.fill((0, 0, 0, 0)); count_text = countdown_font.render(text_str, True, color); count_rect = count_text.get_rect(center=(WIDTH // 2, HEIGHT // 2)); overlay.blit(count_text, count_rect)
        start_time = time.time(); duration = 0.8; pause = 0.2
        while time.time() < start_time + duration + pause:
            elapsed = time.time() - start_time; alpha = 0
            if elapsed < duration / 2: alpha = int(255 * (elapsed / (duration / 2)))
            elif elapsed < duration / 2 + pause: alpha = 255
            elif elapsed < duration + pause: alpha = int(255 * (1 - ((elapsed - duration/2 - pause) / (duration/2))))
            alpha = max(0, min(255, alpha)); overlay.set_alpha(alpha)
            # Redraw base elements
            screen.fill(DARK_BG); draw_tech_background()
            cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=2)
            screen.blit(overlay, (0,0)) # Blit faded text
            # Update camera view during countdown
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1); small_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT)); small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB");
                # Position camera feed below score area
                cam_y_offset = 85 # Estimate based on score area height + padding
                cam_y_offset = max(cam_y_offset, (TOTAL_HEIGHT - CAM_HEIGHT) // 2) # Ensure centered if window is tall
                screen.blit(pygame_frame, (WIDTH + 2, cam_y_offset)) # Adjust Y offset
            pygame.display.flip(); clock.tick(60)

# --- Hand Tracking Function ---
def get_hand_gesture_and_direction(frame, results):
    # (Same as before)
    direction = None; gesture = None; finger_line = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try: # Gesture
            fingertips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
            palm_center_approx_id = mp_hands.HandLandmark.MIDDLE_FINGER_MCP; palm_center_pt = hand_landmarks.landmark[palm_center_approx_id]
            fingers_folded = 0; tip_threshold = 0.12
            for tip_id in fingertips_ids:
                tip_pt = hand_landmarks.landmark[tip_id]; distance = math.hypot(tip_pt.x - palm_center_pt.x, tip_pt.y - palm_center_pt.y)
                if distance < tip_threshold: fingers_folded += 1
            if fingers_folded >= 3: gesture = "FIST"
            else:
                 thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]; thumb_dist = math.hypot(thumb_tip.x - palm_center_pt.x, thumb_tip.y - palm_center_pt.y)
                 if thumb_dist > 0.15: gesture = "OPEN"
        except (IndexError, AttributeError): pass
        try: # Direction
            mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]; tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dx = tip.x - mcp.x; dy = tip.y - mcp.y; orientation_threshold = 0.04
            start_pt = (int(mcp.x * CAM_WIDTH), int(mcp.y * CAM_HEIGHT)); end_pt = (int(tip.x * CAM_WIDTH), int(tip.y * CAM_HEIGHT)); finger_line = (start_pt, end_pt)
            if abs(dx) > abs(dy):
                if abs(dx) > orientation_threshold: direction = (1, 0) if dx > 0 else (-1, 0)
            elif abs(dy) > abs(dx):
                 if abs(dy) > orientation_threshold: direction = (0, -1) if dy < 0 else (0, 1)
        except (IndexError, AttributeError): pass
    return direction, gesture, finger_line

# --- Main Game Loop ---
def game_loop():
    while True: # Outer loop for restarting
        # Initial Game State
        mid_x = (WIDTH // BLOCK_SIZE // 2) * BLOCK_SIZE; mid_y = (HEIGHT // BLOCK_SIZE // 2) * BLOCK_SIZE
        snake_body = [ (mid_x - i * BLOCK_SIZE, mid_y) for i in range(INITIAL_SNAKE_LENGTH) ]; snake_body.reverse()
        direction = (1, 0); change_to = direction; score = 0; game_over = False
        ghost_tail = deque(maxlen=TAIL_FADE_LENGTH); bonus_items = []; spikes = []
        # <<< MODIFIED: Use random colors, remove palette index >>>
        current_head_color = get_random_bright_color()
        current_body_color = get_random_bright_color()
        # Ensure initial head/body colors are different enough (optional)
        while math.dist(current_head_color, current_body_color) < 50:
            current_body_color = get_random_bright_color()
        # <<< END MODIFIED >>>
        slowdown_end_time = 0; current_speed_factor = 1.0
        last_spike_spawn_time = 0; game_start_time = 0

        # Place initial food & spikes
        while True:
            food_x = random.randrange(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE; food_y = random.randrange(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
            food_pos = (food_x, food_y)
            if is_position_valid(food_pos, snake_body, spikes, bonus_items, None): break
        spikes = generate_initial_obstacles(INITIAL_NUM_SPIKES, snake_body, food_pos)

        run_countdown()
        last_update_time = time.time(); game_start_time = last_update_time; last_spike_spawn_time = game_start_time

        # Gameplay Loop
        while not game_over:
            current_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()

            # Hand Tracking
            success, frame = cap.read();
            if not success: time.sleep(0.05); continue
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False; results = hands.process(rgb_frame); rgb_frame.flags.writeable = True
            hand_direction, _, finger_line_pts = get_hand_gesture_and_direction(frame, results)

            # Update Direction Intention
            if hand_direction:
                current_dx, current_dy = direction; new_dx, new_dy = hand_direction
                if not (current_dx == -new_dx and current_dx != 0) and not (current_dy == -new_dy and current_dy != 0): change_to = hand_direction

            # Update Speed
            if slowdown_end_time and current_time > slowdown_end_time: slowdown_end_time = 0; current_speed_factor = 1.0
            actual_speed = SNAKE_SPEED_INITIAL * current_speed_factor; snake_speed_delay = 1.0 / actual_speed if actual_speed > 0 else 1.0

            # Game Logic Update
            if current_time - last_update_time >= snake_speed_delay:
                last_update_time = current_time; direction = change_to
                head_x, head_y = snake_body[0]; new_head = (head_x + direction[0] * BLOCK_SIZE, head_y + direction[1] * BLOCK_SIZE)

                # Collision Detection
                collision_type = None
                if not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT): collision_type = "Wall"
                elif new_head in snake_body[1:]: collision_type = "Self"
                elif new_head in spikes: collision_type = "Spike"
                if collision_type: game_over = True; print(f"Collision: {collision_type}"); continue

                snake_body.insert(0, new_head) # Move snake forward

                # Item Interaction & Score Handling
                ate_something = False; previous_score = score

                # Bonus Items
                item_collected_idx = -1
                for i, (item_pos, item_type, _) in enumerate(bonus_items):
                    if new_head == item_pos:
                        ate_something = True; item_collected_idx = i
                        if item_type == ITEM_BONUS_STAR: score += BONUS_STAR_POINTS; print(f"Collected: Star (+{BONUS_STAR_POINTS})")
                        elif item_type == ITEM_SLOWDOWN_ORB: slowdown_end_time = current_time + SLOWDOWN_DURATION; current_speed_factor = SLOWDOWN_FACTOR; print(f"Collected: Slowdown")
                        break
                if item_collected_idx != -1: del bonus_items[item_collected_idx]

                # Standard Food
                if not ate_something and new_head == food_pos:
                    ate_something = True; score += 1; print("Collected: Food (+1)")
                    while True: # Find new food pos
                        food_x = random.randrange(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE; food_y = random.randrange(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
                        food_pos = (food_x, food_y)
                        if is_position_valid(food_pos, snake_body, spikes, bonus_items, None): break

                # <<< MODIFIED: Trigger random color change every 5 points >>>
                if score // 5 > previous_score // 5:
                    current_head_color = get_random_bright_color()
                    current_body_color = get_random_bright_color()
                    # Optional: Ensure new colors are different enough
                    while math.dist(current_head_color, current_body_color) < 50:
                         current_body_color = get_random_bright_color()
                    print(f"Snake color randomized! Head: {current_head_color}, Body: {current_body_color}")
                # <<< END MODIFIED >>>

                # Remove tail or clear ghost effect
                if not ate_something: ghost_tail.append(snake_body.pop())
                else: ghost_tail.clear()

            # Dynamic Spike Spawning
            if current_time - last_spike_spawn_time > SPIKE_SPAWN_INTERVAL and len(spikes) < MAX_SPIKES:
                last_spike_spawn_time = current_time
                new_spikes = spawn_new_spikes(SPIKES_PER_SPAWN, snake_body, spikes, bonus_items, food_pos)
                if new_spikes: spikes.extend(new_spikes); print(f"Spawned {len(new_spikes)} new spike(s). Total: {len(spikes)}")

            # Bonus Item Spawning & Despawning
            bonus_items = [(p, t, st) for p, t, st in bonus_items if current_time - st < ITEM_LIFETIME]
            if not bonus_items and random.random() < BONUS_SPAWN_CHANCE:
                 item_type = random.choice([ITEM_BONUS_STAR, ITEM_SLOWDOWN_ORB]); attempts = 0
                 while attempts < 50:
                     bx = random.randrange(0, WIDTH // BLOCK_SIZE) * BLOCK_SIZE; by = random.randrange(0, HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
                     bpos = (bx, by)
                     if is_position_valid(bpos, snake_body, spikes, bonus_items, food_pos, buffer_zone=1):
                         bonus_items.append((bpos, item_type, current_time)); print(f"Spawned: {'Star' if item_type == ITEM_BONUS_STAR else 'Slowdown'}"); break
                     attempts += 1

            # --- Drawing ---
            screen.fill(DARK_BG)
            draw_tech_background()
            # Draw Camera BG/Border first
            cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT)
            pygame.draw.rect(screen, CAM_BG, cam_area_rect)
            pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=2)

            # Draw Game Elements
            draw_spikes(spikes)
            draw_food(food_pos)
            draw_bonus_items(bonus_items)
            # Use current random colors
            draw_snake(snake_body, ghost_tail, current_head_color, current_body_color)
            # Draw Score (now positioned above camera)
            display_score(score)

            # Draw Camera Feed below score
            small_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            if results.multi_hand_landmarks:
                 mp_drawing.draw_landmarks(small_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                 if finger_line_pts: cv2.line(small_frame, finger_line_pts[0], finger_line_pts[1], DIRECTION_LINE_COLOR, 2)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")
            # Position camera feed below score area
            cam_y_offset = 85 # Y position to start drawing camera (adjust if score area height changes)
            cam_y_offset = max(cam_y_offset, (TOTAL_HEIGHT - CAM_HEIGHT) // 2) # Center if window is tall
            screen.blit(pygame_frame, (WIDTH + 2, cam_y_offset)) # Blit inside border

            pygame.display.flip(); clock.tick(60)


        # --- Game Over State ---
        game_over_message(score); time.sleep(0.5)
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
                    _, gesture, finger_line_pts = get_hand_gesture_and_direction(frame, results)
                    # Update Camera View below score
                    small_frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
                    if results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(small_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                        if finger_line_pts: cv2.line(small_frame, finger_line_pts[0], finger_line_pts[1], DIRECTION_LINE_COLOR, 2)
                    if gesture: cv2.putText(small_frame, gesture, (10, CAM_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
                    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB); pygame_frame = pygame.image.frombuffer(small_frame_rgb.tobytes(), (CAM_WIDTH, CAM_HEIGHT), "RGB")
                    # Redraw camera bg/border first
                    cam_area_rect = pygame.Rect(WIDTH, 0, CAM_WIDTH, TOTAL_HEIGHT); pygame.draw.rect(screen, CAM_BG, cam_area_rect); pygame.draw.rect(screen, CAM_BORDER_COLOR, cam_area_rect, width=2)
                    # Redraw score display
                    display_score(score) # Keep score visible during game over
                    # Position camera below score
                    cam_y_offset = 85 # Y position to start drawing camera
                    cam_y_offset = max(cam_y_offset, (TOTAL_HEIGHT - CAM_HEIGHT) // 2)
                    screen.blit(pygame_frame, (WIDTH + 2, cam_y_offset)); pygame.display.flip()
                    # Check Gestures
                    if gesture == "FIST": print("Gesture: FIST - Retrying..."); game_over = False; restart_attempt = True; break
                    elif gesture == "OPEN": print("Gesture: OPEN - Exiting..."); quit_attempt = True; break
            clock.tick(15)

        if quit_attempt: break
        if restart_attempt: time.sleep(0.5); continue

    # --- Cleanup ---
    print("Exiting game.")
    cap.release(); cv2.destroyAllWindows(); pygame.quit(); sys.exit()

# --- Start ---
if __name__ == '__main__':
    game_loop()