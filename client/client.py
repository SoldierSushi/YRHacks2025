import socket
import threading
import json
from protocols import Protocols

class Client:
    def __init__(self, host="127.0.0.1", port=8001):
        self.nickname = None
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"Attempting to connect to {host}:{port}...")
            print("\nNetwork Diagnostics:")
            print(f"1. Target IP: {host}")
            print(f"2. Target Port: {port}")
            
            # Try to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            except Exception:
                local_ip = "Unknown"
            finally:
                s.close()
            
            print(f"3. Your IP: {local_ip}")
            print("\nConnection Status:")
            
            self.server.connect((host, port))
            print("✅ Successfully connected to server!")
        except ConnectionRefusedError:
            print("❌ Connection refused. Please check:")
            print("   - Is the server running?")
            print("   - Are you on the same network?")
            print("   - Is the IP address correct?")
            raise
        except socket.gaierror:
            print("❌ Invalid IP address. Please check the IP address format.")
            raise
        except OSError as e:
            if "No route to host" in str(e):
                print("❌ No route to host. Please check:")
                print("   - Are both computers on the same network?")
                print("   - Is the IP address correct?")
                print("   - Try pinging the server IP to test connectivity")
            else:
                print(f"❌ Connection error: {str(e)}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            raise

        self.closed = False
        self.started = False
        self.questions = []
        self.current_question_index = 0
        self.opponent_question_index = 0
        self.opponent_data = None
        self.winner = None

    def start(self):
        receive_thread = threading.Thread(target=self.receive)
        receive_thread.start()

    def send(self, request, message):
        data = {"type": request, "data": message}
        self.server.send(json.dumps(data).encode("ascii"))

    def receive(self):
        while not self.closed:
            try:
                data = self.server.recv(1024).decode("ascii")
                message = json.loads(data)
                self.handle_response(message)
            except:
                break
        
        self.close()

    def close(self):
        self.closed = True
        self.server.close()

    def client_validate_answer(self, attempt):
        question = self.get_current_question()
        answer = eval(question)
        if answer == int(attempt):
            self.current_question_index += 1

    def handle_response(self, response):
        r_type = response.get("type")
        data = response.get("data")

        if r_type == Protocols.Response.QUESTIONS:
            self.questions = data
        elif r_type == Protocols.Response.OPPONENT:
            self.opponent_data = data
        elif r_type == Protocols.Response.OPPONENT_ADVANCE:
            self.opponent_question_index += 1
        elif r_type == Protocols.Response.START:
            self.started = True
        elif r_type == Protocols.Response.WINNER:
            self.winner = data
            self.close()
        elif r_type == Protocols.Response.OPPONENT_LEFT:
            self.close()

    def get_current_question(self):
        if self.current_question_index >= len(self.questions):
            return ""
        return self.questions[self.current_question_index]