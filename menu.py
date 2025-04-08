import pygame
import sys
import random
import math
import time # Might be needed by menu elements later, good to keep

# --- Import your game files ---
try:
    # Make sure your snake game file is named 'snake.py'
    import snake
    # You would import other game modules here later
    # import ice_hockey_game
    # import test_game

    GAME_MODULES = { # Map game IDs to their run functions
        'game_1': None, # Placeholder for Ice Hockey (e.g., ice_hockey_game.run_hockey)
        'game_2': snake.run_snake, # Reference to the function in snake.py
        'game_3': None, # Placeholder for Test Game (e.g., test_game.run_test)
    }
    print("Game modules imported successfully.")
except ImportError as e:
    print(f"Error importing game modules: {e}")
    print("Ensure game files (e.g., snake.py) are in the same directory.")
    # Fallback if imports fail, prevents crashes but games won't run
    GAME_MODULES = {'game_1': None, 'game_2': None, 'game_3': None}

# --- Initialization ---
pygame.init() # Initialize Pygame core modules
pygame.font.init() # Initialize Font module

# --- Screen Setup ---
# Determine dimensions based on typical game + camera setup
# You can adjust these defaults if needed
DEFAULT_GAME_WIDTH = 600
DEFAULT_GAME_HEIGHT = 500
DEFAULT_CAM_WIDTH = 360
DEFAULT_CAM_HEIGHT = 270

SCREEN_WIDTH = DEFAULT_GAME_WIDTH + DEFAULT_CAM_WIDTH # Total window width
SCREEN_HEIGHT = max(DEFAULT_GAME_HEIGHT, DEFAULT_CAM_HEIGHT) # Total window height
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Open Resort - Menu System")

# --- Colors (For Menu) ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LINE_CYAN = (0, 100, 130) # Background pattern lines
DOT_COLORS = [ (0, 180, 200), (0, 210, 230), (30, 220, 240) ] # Background pattern dots
HOVER_YELLOW = (255, 255, 0) # Hover color for menu text
PANEL_BLUE = (20, 50, 100)        # Color for game panel background
PANEL_HOVER_BLUE = (40, 80, 150) # Color for hovered panel
PANEL_TEXT_COLOR = WHITE
BACK_BUTTON_TEXT_COLOR = BLACK # Default text color for back button
BACK_BUTTON_COLOR = WHITE      # Background color for back button (non-hover)
BACK_HOVER_COLOR = HOVER_YELLOW # Background color for back button (hover)


# --- Fonts (For Menu) ---
# Use a generic try-except for menu fonts
try:
    title_font = pygame.font.Font(None, 80)
    option_font = pygame.font.Font(None, 55)
    panel_font = pygame.font.Font(None, 40)
    small_font = pygame.font.Font(None, 30)
except pygame.error as e:
     print(f"Menu Font Error: {e}. Using default fonts.")
     title_font = pygame.font.Font(pygame.font.get_default_font(), 60)
     option_font = pygame.font.Font(pygame.font.get_default_font(), 40)
     panel_font = pygame.font.Font(pygame.font.get_default_font(), 30)
     small_font = pygame.font.Font(pygame.font.get_default_font(), 25)


# --- Pattern Parameters (For Menu Background) ---
NUM_DOTS = 45
MIN_RADIUS = 3
MAX_RADIUS = 12
MAX_CONNECTION_DISTANCE = 150
DOT_SPEED = 0.3
dots = [] # List to hold dot data

# --- Menu Options ---
main_menu_options = ["select game", "quit"]
main_option_rects = []

# --- Game Panel Data ---
game_panels_data = [
    {'id': 'game_1', 'name': 'ice hockey'},
    {'id': 'game_2', 'name': 'snake'},
    {'id': 'game_3', 'name': 'test'},
]
game_panels = [] # Will store dicts with calculated rects
panel_width = 250
panel_height = 180
panel_spacing = 40

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

# --- Text Drawing Helper (For Menu) ---
def draw_text(text, font, color, surface, x, y, center=True):
    """Helper function to draw text on the menu screen."""
    try:
        textobj = font.render(text, True, color)
        textrect = textobj.get_rect()
        if center:
            textrect.center = (x, y)
        else:
            textrect.topleft = (x, y)
        surface.blit(textobj, textrect)
        return textrect
    except pygame.error as e:
        print(f"Error rendering text '{text}': {e}")
        return None # Return None if rendering fails


# --- Calculate Panel Layout (For Game Select Screen) ---
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
    start_y = (SCREEN_HEIGHT - total_grid_height) // 2 + 40 # Position grid lower

    for i, data in enumerate(game_panels_data):
        row = i // cols
        col = i % cols
        panel_x = start_x + col * (panel_width + panel_spacing)
        panel_y = start_y + row * (panel_height + panel_spacing)
        rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        game_panels.append({
            'id': data['id'],
            'name': data['name'],
            'rect': rect,
        })

# --- Main Menu Screen Function ---
def main_menu_screen(surface):
    """Displays the main menu and handles interaction. Returns next state."""
    global main_option_rects
    main_option_rects = []
    clock = pygame.time.Clock()

    while True:
        mx, my = pygame.mouse.get_pos()
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: generate_dots() # Regenerate background
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: clicked = True

        update_dots() # Update background animation
        surface.fill(BLACK)
        draw_pattern(surface) # Draw animated background
        draw_text("open resort", title_font, WHITE, surface, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)

        # Draw options
        button_y_start = SCREEN_HEIGHT // 2 + 50
        temp_rects = []
        for i, option in enumerate(main_menu_options):
            button_x = SCREEN_WIDTH // 2
            button_y = button_y_start + i * 60
            # Define clickable area (doesn't need drawing)
            button_rect_for_collision = pygame.Rect(0, 0, 350, 70)
            button_rect_for_collision.center = (button_x, button_y)

            text_color = WHITE
            is_hovering = button_rect_for_collision.collidepoint(mx, my)
            if is_hovering: text_color = HOVER_YELLOW

            # Draw the text and store its actual rect
            actual_text_rect = draw_text(option, option_font, text_color, surface, button_x, button_y)
            if actual_text_rect: # Only append if text was drawn successfully
                 temp_rects.append(actual_text_rect)

            # Check click AFTER drawing
            if clicked and is_hovering:
                if option == "select game":
                    return "game_select"
                elif option == "quit":
                    return "quit"

        main_option_rects = temp_rects # Use the actual text rects for clicking later if needed

        pygame.display.flip()
        clock.tick(60)

# --- Game Selection Screen Function ---
def game_select_screen(surface):
    """Displays game panels and handles interaction. Returns next state."""
    clock = pygame.time.Clock()
    if not game_panels: # Calculate panel layout if needed
       setup_panels()

    back_button_rect = pygame.Rect(20, SCREEN_HEIGHT - 60, 100, 40)

    while True:
        mx, my = pygame.mouse.get_pos()
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_SPACE: generate_dots() # Regenerate background
                 if event.key == pygame.K_ESCAPE: return "main_menu" # Go back
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: clicked = True

        update_dots() # Update background animation
        surface.fill(BLACK)
        draw_pattern(surface) # Draw animated background
        draw_text("select game", option_font, WHITE, surface, SCREEN_WIDTH // 2, 90) # Title

        # Draw game panels
        for panel in game_panels:
            is_hovering = panel['rect'].collidepoint(mx, my)
            panel_color = LINE_CYAN if is_hovering else BLACK
            pygame.draw.rect(surface, panel_color, panel['rect'], border_radius=10)
            pygame.draw.rect(surface, WHITE, panel['rect'], width=1, border_radius=10) # Outline

            # Draw panel name, centered vertically
            text_y = panel['rect'].centery
            draw_text(panel['name'], panel_font, PANEL_TEXT_COLOR, surface, panel['rect'].centerx, text_y)

            # Check click AFTER drawing
            if clicked and is_hovering:
                print(f"Selected game: {panel['name']} (ID: {panel['id']})")
                return panel['id'] # Return the game ID

        # Draw Back Button
        back_hover = back_button_rect.collidepoint(mx, my)
        back_bg_color = BACK_HOVER_COLOR if back_hover else WHITE
        back_txt_color = BLACK

        pygame.draw.rect(surface, back_bg_color, back_button_rect, border_radius=5)
        draw_text("back", small_font, back_txt_color, surface, back_button_rect.centerx, back_button_rect.centery)

        if clicked and back_hover:
            return "main_menu"

        pygame.display.flip()
        clock.tick(60)


# --- Main Game Loop ---
def run_game():
    """Main function to manage game states."""
    game_state = "main_menu"
    generate_dots() # Initial background pattern generation

    while True:
        if game_state == "main_menu":
            next_state = main_menu_screen(screen)
        elif game_state == "game_select":
            next_state = game_select_screen(screen)

        # --- Look up and run game state function from GAME_MODULES ---
        elif game_state in GAME_MODULES: # Check if the state is a known game ID
            run_func = GAME_MODULES.get(game_state) # Get the associated function
            if run_func:
                print(f"--- Entering Game: {game_state} ---")
                # Call the specific game's run function, passing the screen
                next_state = run_func(screen)
                print(f"--- Exiting Game: {game_state}, next state: {next_state} ---")
                # Optional: Regenerate menu background after game exits
                generate_dots() # Generate fresh background for menu return
            else:
                # Handle case where game ID exists but function is None (not linked yet)
                print(f"Warning: No run function linked for game '{game_state}'. Returning to game select.")
                next_state = "game_select" # Go back if function missing

        elif game_state == "quit":
            break # Exit the main loop
        else:
            # Fallback for unknown states potentially returned by games or errors
            print(f"Unknown game state encountered: '{game_state}'. Returning to game select.")
            next_state = "game_select" # Fallback to game selection

        game_state = next_state # Update the current state for the next loop iteration

    print("Exiting application.")
    pygame.quit()
    sys.exit()

# --- Run the Application ---
if __name__ == "__main__":
    # Calculate panel layout once at the start (optional, game_select does it too)
    setup_panels()
    run_game() # Start the main state machine