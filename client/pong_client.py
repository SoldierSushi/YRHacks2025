import pygame
import cv2
import numpy as np
import mediapipe as mp
import socket
import json
import threading

class PongClient:
    def __init__(self, host="127.0.0.1", port=8001):
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Hand-Controlled Pong")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # Initialize OpenCV and MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize socket
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Attempting to connect to {host}:{port}...")
            self.client.connect((host, port))
            print("Successfully connected to server!")
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            print("Please check that:")
            print("1. The server is running")
            print("2. The IP address is correct")
            print("3. Both computers are on the same network")
            raise

        # Game state
        self.ball = {"x": 400, "y": 300}
        self.paddle_y = 250
        self.opponent_y = 250
        self.player_score = 0
        self.opponent_score = 0
        self.opponent_name = ""
        self.waiting = True
        self.running = True

        # Start network thread
        self.network_thread = threading.Thread(target=self.receive_data)
        self.network_thread.start()

    def get_hand_position(self):
        success, image = self.cap.read()
        if not success:
            return None

        # Flip the image horizontally
        image = cv2.flip(image, 1)
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the y position of the index finger tip
                index_y = hand_landmarks.landmark[8].y
                return int(index_y * 600)  # Scale to screen height
        
        return None

    def receive_data(self):
        while self.running:
            try:
                data = self.client.recv(4096).decode()
                if not data:
                    break

                game_state = json.loads(data)
                
                if "status" in game_state:
                    if game_state["status"] == "waiting":
                        self.waiting = True
                    elif game_state["status"] == "start":
                        self.waiting = False
                        self.opponent_name = game_state["opponent"]
                else:
                    self.ball = game_state["ball"]
                    self.opponent_y = game_state["player2"]["y"]
                    self.player_score = game_state["player1"]["score"]
                    self.opponent_score = game_state["player2"]["score"]
            except:
                break

    def send_paddle_position(self, y):
        try:
            self.client.send(str(y).encode())
        except:
            pass

    def run(self):
        # Send player name
        self.client.send("Player".encode())

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Get hand position
            hand_y = self.get_hand_position()
            if hand_y is not None:
                self.paddle_y = hand_y
                self.send_paddle_position(hand_y)

            # Draw everything
            self.screen.fill((0, 0, 0))
            
            if self.waiting:
                text = self.font.render("Waiting for opponent...", True, (255, 255, 255))
                self.screen.blit(text, (250, 250))
            else:
                # Draw paddles
                pygame.draw.rect(self.screen, (255, 255, 255), (50, self.paddle_y, 20, 100))
                pygame.draw.rect(self.screen, (255, 255, 255), (730, self.opponent_y, 20, 100))
                
                # Draw ball
                pygame.draw.circle(self.screen, (255, 255, 255), (self.ball["x"], self.ball["y"]), 10)
                
                # Draw scores
                score_text = self.font.render(f"{self.player_score} - {self.opponent_score}", True, (255, 255, 255))
                self.screen.blit(score_text, (350, 50))
                
                # Draw opponent name
                name_text = self.font.render(f"Opponent: {self.opponent_name}", True, (255, 255, 255))
                self.screen.blit(name_text, (550, 50))

            pygame.display.flip()
            self.clock.tick(60)

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        self.client.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pong_client.py <server_ip>")
        print("Example: python pong_client.py 192.168.1.10")
        sys.exit(1)
    
    # Parse the server address
    server_address = sys.argv[1]
    if ":" in server_address:
        host, port = server_address.split(":")
        port = int(port)
    else:
        host = server_address
        port = 8001  # Default port
    
    print(f"Starting client, connecting to server at {host}:{port}...")
    client = PongClient(host=host, port=port)
    client.run() 