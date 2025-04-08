import pygame
import sys
import random
import math

# --- Initialization ---
pygame.init()

# --- Screen Setup ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Open Resort - Menu System") # Keep caption updated


# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LINE_CYAN = (0, 100, 130)
DOT_COLORS = [ (0, 180, 200), (0, 210, 230), (30, 220, 240) ]
HOVER_YELLOW = (255, 255, 0)
PANEL_BLUE = (20, 50, 100)
PANEL_HOVER_BLUE = (40, 80, 150)
PANEL_TEXT_COLOR = WHITE
# Make back button text color dynamic based on hover
BACK_BUTTON_TEXT_COLOR = BLACK # Default text color for back button
BACK_BUTTON_COLOR = WHITE      # Background color for back button (non-hover)
BACK_HOVER_COLOR = HOVER_YELLOW # Background color for back button (hover)


# --- Fonts ---
title_font = pygame.font.Font(None, 80)
option_font = pygame.font.Font(None, 55)
panel_font = pygame.font.Font(None, 40)
small_font = pygame.font.Font(None, 30)

# --- Pattern Parameters ---
NUM_DOTS = 45
MIN_RADIUS = 3
MAX_RADIUS = 12
MAX_CONNECTION_DISTANCE = 150
DOT_SPEED = 0.3
dots = []

# --- Menu Options ---
main_menu_options = ["select game", "quit"]
main_option_rects = []

# --- Game Panel Data ---
# No longer need 'img_path' here unless you plan to use it differently later
game_panels_data = [
    {'id': 'game_1', 'name': 'ice hockey'},
    {'id': 'game_2', 'name': 'snake'},
    {'id': 'game_3', 'name': 'test'},
]
game_panels = []
panel_width = 250
panel_height = 180 # Height remains the same for consistency
panel_spacing = 40

# --- Pattern Functions (generate_dots, update_dots, draw_pattern) ---
# (Keep these functions exactly as they were)
def generate_dots():
    global dots
    dots = []
    for _ in range(NUM_DOTS):
        pos = (random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
        radius = max(1, random.randint(MIN_RADIUS, MAX_RADIUS))
        color = random.choice(DOT_COLORS)
        angle = random.uniform(0, 2 * math.pi)
        vel_x = math.cos(angle) * DOT_SPEED
        vel_y = math.sin(angle) * DOT_SPEED
        velocity = pygame.Vector2(vel_x, vel_y)
        dots.append({ 'pos': pygame.Vector2(pos), 'radius': radius, 'color': color, 'velocity': velocity })

def update_dots():
    for dot in dots:
        dot['pos'] += dot['velocity']
        if dot['pos'].x < -MAX_RADIUS: dot['pos'].x = SCREEN_WIDTH + MAX_RADIUS
        elif dot['pos'].x > SCREEN_WIDTH + MAX_RADIUS: dot['pos'].x = -MAX_RADIUS
        if dot['pos'].y < -MAX_RADIUS: dot['pos'].y = SCREEN_HEIGHT + MAX_RADIUS
        elif dot['pos'].y > SCREEN_HEIGHT + MAX_RADIUS: dot['pos'].y = -MAX_RADIUS

def draw_pattern(surface):
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            dot1 = dots[i]
            dot2 = dots[j]
            distance = dot1['pos'].distance_to(dot2['pos'])
            if distance < MAX_CONNECTION_DISTANCE:
                try:
                    pygame.draw.aaline(surface, LINE_CYAN, dot1['pos'], dot2['pos'])
                except TypeError: pass
    for dot in dots:
         try:
             pygame.draw.circle(surface, dot['color'], (int(dot['pos'].x), int(dot['pos'].y)), dot['radius'])
         except TypeError: pass

# --- Text Drawing Helper ---
def draw_text(text, font, color, surface, x, y, center=True):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    if center:
        textrect.center = (x, y)
    else:
        textrect.topleft = (x, y)
    surface.blit(textobj, textrect)
    return textrect

# --- Calculate Panel Layout ---
def setup_panels():
    """Calculates positions and rects for game panels."""
    global game_panels
    game_panels = []
    num_panels = len(game_panels_data)
    cols = 3
    rows = math.ceil(num_panels / cols)

    total_grid_width = cols * panel_width + (cols - 1) * panel_spacing
    start_x = (SCREEN_WIDTH - total_grid_width) // 2

    total_grid_height = rows * panel_height + (rows - 1) * panel_spacing
    start_y = (SCREEN_HEIGHT - total_grid_height) // 2 + 40

    for i, data in enumerate(game_panels_data):
        row = i // cols
        col = i % cols
        panel_x = start_x + col * (panel_width + panel_spacing)
        panel_y = start_y + row * (panel_height + panel_spacing)
        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        # --- Image Loading Removed ---
        # No longer loading or storing images here

        game_panels.append({
            'id': data['id'],
            'name': data['name'],
            'rect': rect,
            # 'image': None # No longer needed
        })

# --- Main Menu Function ---
# (main_menu_screen remains unchanged from the previous version)
def main_menu_screen(surface):
    global main_option_rects
    main_option_rects = []
    clock = pygame.time.Clock()
    while True:
        mx, my = pygame.mouse.get_pos()
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: generate_dots()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: clicked = True
        update_dots()
        surface.fill(BLACK)
        draw_pattern(surface)
        draw_text("open resort", title_font, WHITE, surface, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
        button_y_start = SCREEN_HEIGHT // 2 + 50
        temp_rects = []
        for i, option in enumerate(main_menu_options):
            button_x = SCREEN_WIDTH // 2
            button_y = button_y_start + i * 60 # Adjusted spacing slightly
            button_rect_for_collision = pygame.Rect(0, 0, 350, 70)
            button_rect_for_collision.center = (button_x, button_y)
            text_color = WHITE
            is_hovering = button_rect_for_collision.collidepoint(mx, my)
            if is_hovering: text_color = HOVER_YELLOW
            actual_text_rect = draw_text(option, option_font, text_color, surface, button_x, button_y)
            temp_rects.append(actual_text_rect)
            if clicked and is_hovering:
                if option == "select game": return "game_select"
                elif option == "quit": return "quit"
        main_option_rects = temp_rects
        pygame.display.flip()
        clock.tick(60)


# --- Game Selection Screen Function ---
def game_select_screen(surface):
    """Displays game panels (text only) and handles interaction. Returns next state."""
    clock = pygame.time.Clock()
    if not game_panels:
       setup_panels()

    back_button_rect = pygame.Rect(20, SCREEN_HEIGHT - 60, 100, 40)

    while True:
        mx, my = pygame.mouse.get_pos()
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_SPACE: generate_dots()
                 if event.key == pygame.K_ESCAPE: return "main_menu"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: clicked = True

        update_dots()
        surface.fill(BLACK)
        draw_pattern(surface)
        draw_text("select game", option_font, WHITE, surface, SCREEN_WIDTH // 2, 90) # Title for this screen

        # Draw game panels
        for panel in game_panels:
            is_hovering = panel['rect'].collidepoint(mx, my)
            panel_color = LINE_CYAN if is_hovering else BLACK
            pygame.draw.rect(surface, panel_color, panel['rect'], border_radius=10)
            pygame.draw.rect(surface, WHITE, panel['rect'], width=1, border_radius=10) # Outline

            # --- Image Drawing Removed ---

            # Draw panel name, now always centered vertically
            text_y = panel['rect'].centery # Center the text vertically
            draw_text(panel['name'], panel_font, PANEL_TEXT_COLOR, surface, panel['rect'].centerx, text_y)

            if clicked and is_hovering:
                print(f"Selected game: {panel['name']} (ID: {panel['id']})")
                # Return the game ID to start that specific game
                return panel['id'] # <-- CHANGE THIS to return ID

        # Draw Back Button
        back_hover = back_button_rect.collidepoint(mx, my)
        back_bg_color = BACK_HOVER_COLOR if back_hover else BACK_BUTTON_COLOR
        # Optionally change text color on hover too for better contrast
        back_txt_color = HOVER_YELLOW if back_hover else WHITE # Keep text black, or change if needed
        draw_text("back", small_font, back_txt_color, surface, back_button_rect.centerx, back_button_rect.centery)

        if clicked and back_hover:
            return "main_menu"

        pygame.display.flip()
        clock.tick(60)

# --- Placeholder Game Functions ---
# Add dummy functions for each game ID to avoid errors
def run_game_1(surface):
    """Placeholder function for game 1."""
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "game_select" # Go back to selection screen

        surface.fill(BLACK) # Simple black screen for the placeholder
        draw_text("Ice Hockey Game Running!", title_font, WHITE, surface, SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50)
        draw_text("Press ESC to return", option_font, WHITE, surface, SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50)

        pygame.display.flip()
        clock.tick(60)

def run_game_2(surface):
    """Placeholder function for game 2."""
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "game_select"

        surface.fill(BLACK)
        draw_text("Snake Game Running!", title_font, WHITE, surface, SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50)
        draw_text("Press ESC to return", option_font, WHITE, surface, SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50)

        pygame.display.flip()
        clock.tick(60)

def run_game_3(surface):
    """Placeholder function for game 3."""
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "game_select"

        surface.fill(BLACK)
        draw_text("Test Game Running!", title_font, WHITE, surface, SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50)
        draw_text("Press ESC to return", option_font, WHITE, surface, SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50)

        pygame.display.flip()
        clock.tick(60)


# --- Main Game Loop ---
def run_game():
    """Main function to manage game states."""
    game_state = "main_menu"
    generate_dots()

    while True:
        if game_state == "main_menu":
            next_state = main_menu_screen(screen)
        elif game_state == "game_select":
            next_state = game_select_screen(screen)
        # --- Add Elif for Actual Game States ---
        elif game_state == "game_1":
            next_state = run_game_1(screen) # Call placeholder game 1
        elif game_state == "game_2":
            next_state = run_game_2(screen) # Call placeholder game 2
        elif game_state == "game_3":
            next_state = run_game_3(screen) # Call placeholder game 3
        # --- End Game State Elifs ---
        elif game_state == "quit":
            break
        else:
            print(f"Unknown game state: '{game_state}'. Returning to main menu.")
            next_state = "main_menu"

        game_state = next_state

    pygame.quit()
    sys.exit()

# --- Run the Application ---
if __name__ == "__main__":
    run_game()