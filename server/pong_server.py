import socket
import threading
import json
import time

class PongServer:
    def __init__(self, host="0.0.0.0", port=8001):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, self.port))
        self.server.listen()

        # Game state
        self.players = {}
        self.ball = {"x": 400, "y": 300, "dx": 5, "dy": 5}
        self.paddle_height = 100
        self.paddle_width = 20
        self.screen_width = 800
        self.screen_height = 600
        self.waiting_for_player = None

    def handle_client(self, client):
        # Wait for player name
        name = client.recv(1024).decode()
        self.players[client] = {
            "name": name,
            "y": self.screen_height // 2 - self.paddle_height // 2,
            "score": 0
        }

        # If no waiting player, make this player wait
        if not self.waiting_for_player:
            self.waiting_for_player = client
            client.send(json.dumps({"status": "waiting"}).encode())
        else:
            # Start game with both players
            self.waiting_for_player.send(json.dumps({"status": "start", "opponent": name}).encode())
            client.send(json.dumps({"status": "start", "opponent": self.players[self.waiting_for_player]["name"]}).encode())
            
            # Start game thread
            game_thread = threading.Thread(target=self.run_game, args=(client, self.waiting_for_player))
            game_thread.start()
            self.waiting_for_player = None

    def run_game(self, player1, player2):
        while True:
            try:
                # Update ball position
                self.ball["x"] += self.ball["dx"]
                self.ball["y"] += self.ball["dy"]

                # Ball collision with top and bottom
                if self.ball["y"] <= 0 or self.ball["y"] >= self.screen_height:
                    self.ball["dy"] *= -1

                # Ball collision with paddles
                if (self.ball["x"] <= 50 and 
                    self.players[player1]["y"] <= self.ball["y"] <= self.players[player1]["y"] + self.paddle_height):
                    self.ball["dx"] *= -1
                elif (self.ball["x"] >= self.screen_width - 50 and 
                      self.players[player2]["y"] <= self.ball["y"] <= self.players[player2]["y"] + self.paddle_height):
                    self.ball["dx"] *= -1

                # Ball out of bounds
                if self.ball["x"] <= 0:
                    self.players[player2]["score"] += 1
                    self.reset_ball()
                elif self.ball["x"] >= self.screen_width:
                    self.players[player1]["score"] += 1
                    self.reset_ball()

                # Send game state to both players
                game_state = {
                    "ball": self.ball,
                    "player1": {
                        "y": self.players[player1]["y"],
                        "score": self.players[player1]["score"]
                    },
                    "player2": {
                        "y": self.players[player2]["y"],
                        "score": self.players[player2]["score"]
                    }
                }

                player1.send(json.dumps(game_state).encode())
                player2.send(json.dumps(game_state).encode())

                # Receive paddle positions
                try:
                    data = player1.recv(1024).decode()
                    if data:
                        self.players[player1]["y"] = int(data)
                except:
                    pass

                try:
                    data = player2.recv(1024).decode()
                    if data:
                        self.players[player2]["y"] = int(data)
                except:
                    pass

                time.sleep(0.016)  # ~60 FPS
            except:
                break

    def reset_ball(self):
        self.ball = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2,
            "dx": 5,
            "dy": 5
        }

    def start(self):
        print(f"Server started on {self.host}:{self.port}")
        while True:
            client, addr = self.server.accept()
            print(f"New connection from {addr}")
            thread = threading.Thread(target=self.handle_client, args=(client,))
            thread.start()

if __name__ == "__main__":
    server = PongServer()
    server.start() 