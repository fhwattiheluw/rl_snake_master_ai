import traceback
import pygame
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import sys
from enum import Enum
from stable_baselines3 import PPO

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeSoloTester:
    def __init__(self, model_path, width=70, height=70, cell_size=10):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # --- SOLO TESTER FEATURES ---
        self.name = os.path.splitext(os.path.basename(model_path))[0]
        self.run_count = 1
        
        # Load Model dengan custom_objects agar mencegah error Learning Rate
        print(f"Loading Model ({self.name})...")
        self.model = PPO.load(model_path, custom_objects={
            "learning_rate": 0.0, 
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        })
        
        # Colors
        self.BLACK = (20, 20, 20)
        self.WHITE = (255, 255, 255)
        self.GRID_COLOR = (40, 40, 40)
        
        self.S1_HEAD = (50, 255, 50)
        self.S1_BODY = (0, 180, 0)
        
        self.FOOD_COLOR = (255, 50, 50)
        self.MAZE_COLOR = (140, 140, 150)

        # Inisialisasi Pygame
        pygame.init()
        window_width = self.width * self.cell_size
        window_height = self.height * self.cell_size + 80
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(f"Snake AI Solo Tester - {self.name}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.setup_maze()
        self.reset()

    def setup_maze(self):
        # Mempertahankan layout maze dari tournament_app
        self.maze = set()
        for x in range(15, 56):
            self.maze.add((x, 15))
            self.maze.add((x, 54))
        for y in range(25, 46):
            self.maze.add((15, y))
            self.maze.add((54, y))
        for x in range(32, 39):
            for y in range(32, 39):
                self.maze.add((x, y))

    def reset(self):
        # Posisi awal sama dengan Player 1 di arena turnamen
        self.snake = [(10, 35), (9, 35), (8, 35)]
        self.dir = Direction.RIGHT
        self.score = 0
        self.alive = True
        
        self.food = self._generate_food()
        self.game_over = False

    def _generate_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake and food not in self.maze:
                return food

    def _get_obs(self):
        if not self.alive or not self.snake:
            return np.zeros(11, dtype=np.float32)

        head = self.snake[0]
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_dir_idx = directions.index(self.dir)

        # Sensor Bahaya (Disesuaikan untuk mode solo, menghapus cek tabrakan musuh)
        danger_straight = self._is_collision(head, self.dir)
        danger_right = self._is_collision(head, directions[(current_dir_idx + 1) % 4])
        danger_left = self._is_collision(head, directions[(current_dir_idx - 1) % 4])

        dir_up = self.dir == Direction.UP
        dir_right = self.dir == Direction.RIGHT
        dir_down = self.dir == Direction.DOWN
        dir_left = self.dir == Direction.LEFT

        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        # Menghasilkan Observation Space (11,) sesuai standar turnamen
        return np.array([
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_up, food_down, food_left, food_right
        ], dtype=np.float32)

    def _is_collision(self, position, direction):
        x, y = position
        
        if direction == Direction.UP: y -= 1
        elif direction == Direction.DOWN: y += 1
        elif direction == Direction.LEFT: x -= 1
        elif direction == Direction.RIGHT: x += 1
            
        new_pos = (x, y)

        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        if new_pos in self.maze: return True
        if new_pos in self.snake: return True
        
        return False
        
    def _update_direction(self, action, current_dir):
        if action == 0: return current_dir
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = directions.index(current_dir)
        if action == 1: return directions[(idx + 1) % 4]
        elif action == 2: return directions[(idx - 1) % 4]
        return current_dir

    def step(self):
        if self.game_over: return
        
        if self.alive:
            obs = self._get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            self.dir = self._update_direction(action, self.dir)
            
        new_head = None
        
        if self.alive:
            hx, hy = self.snake[0]
            if self.dir == Direction.UP: hy -= 1
            elif self.dir == Direction.DOWN: hy += 1
            elif self.dir == Direction.LEFT: hx -= 1
            elif self.dir == Direction.RIGHT: hx += 1
            new_head = (hx, hy)
            
        if self.alive:
            if (new_head[0] < 0 or new_head[0] >= self.width or
                new_head[1] < 0 or new_head[1] >= self.height or
                new_head in self.maze or
                new_head in self.snake):
                self.alive = False
                
        if not self.alive:
            self.game_over = True
            return
            
        if self.alive:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                self.food = self._generate_food()
            else:
                self.snake.pop()

    def render(self):
        self.screen.fill(self.BLACK)
        
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (x, 0), (x, self.height * self.cell_size))
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y), (self.width * self.cell_size, y))

        for mx, my in self.maze:
            rect = pygame.Rect(mx * self.cell_size, my * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.MAZE_COLOR, rect)
            pygame.draw.rect(self.screen, (100, 100, 110), rect, 1)
            
        food_rect = pygame.Rect(self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.FOOD_COLOR, food_rect)

        if self.alive or self.game_over:
            for i, (x, y) in enumerate(self.snake):
                color = self.S1_HEAD if i == 0 else self.S1_BODY
                if not self.alive: color = (100, 100, 100)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)

        ui_y = self.height * self.cell_size
        pygame.draw.rect(self.screen, (30, 30, 30), (0, ui_y, self.width * self.cell_size, 80))
        
        status_text = f"Model: {self.name}  |  Score: {self.score}  |  Run: {self.run_count}"
        t1 = self.font.render(status_text, True, self.S1_HEAD if self.alive else (150, 150, 150))
        self.screen.blit(t1, (self.width * self.cell_size // 2 - t1.get_width() // 2, ui_y + 25))

        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size())
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            go_t = self.font.render("SIMULATION OVER", True, self.WHITE)
            score_t = self.font.render(f"Final Score: {self.score}", True, (255, 215, 0))
            restart_t = self.small_font.render("Press SPACE to Restart or ESC to Quit", True, self.WHITE)
            
            cx = (self.width * self.cell_size) // 2
            cy = (self.height * self.cell_size) // 2
            
            self.screen.blit(go_t, go_t.get_rect(center=(cx, cy - 40)))
            self.screen.blit(score_t, score_t.get_rect(center=(cx, cy)))
            self.screen.blit(restart_t, restart_t.get_rect(center=(cx, cy + 40)))

        pygame.display.flip()

    def run(self, fps=30):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_SPACE and self.game_over:
                        self.run_count += 1
                        self.reset()
                        
            self.step()
            self.render()
            self.clock.tick(fps)
            
        pygame.quit()

class SoloLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake AI - Solo Tester Launcher")
        self.root.geometry("500x320")
        self.root.configure(bg="#1e1e2f")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_col = "#1e1e2f"
        fg_col = "#e0e0e0"
        btn_col = "#3a3a52"
        btn_hover = "#4a4a68"
        accent = "#00d2ff"
        
        style.configure("TFrame", background=bg_col)
        style.configure("TLabel", background=bg_col, foreground=fg_col, font=("Segoe UI", 12))
        style.configure("Header.TLabel", font=("Segoe UI", 22, "bold"), foreground=accent)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground="#8a8a8a")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"), foreground="#a0a0a0")
        
        style.configure("TButton", 
                       font=("Segoe UI", 11, "bold"), 
                       background=btn_col, 
                       foreground=fg_col,
                       borderwidth=0,
                       focuscolor=btn_col,
                       padding=10)
        style.map("TButton", background=[("active", btn_hover)])
        
        style.configure("Start.TButton", background="#00c853", foreground="#ffffff", font=("Segoe UI", 14, "bold"))
        style.map("Start.TButton", background=[("active", "#00e676")])

        main_frame = ttk.Frame(root, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="SNAKE SOLO TESTER", style="Header.TLabel").pack(pady=(0, 5))
        ttk.Label(main_frame, text="Uji performa pergerakan dan hindaran model Anda", style="SubHeader.TLabel").pack(pady=(0, 25))
        
        p1_frame = ttk.Frame(main_frame)
        p1_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(p1_frame, text="Model Anda (.zip):", font=("Segoe UI", 12, "bold"), foreground="#4caf50").pack(anchor=tk.W)
        self.lbl_model = ttk.Label(p1_frame, text="No model selected", style="Status.TLabel")
        self.lbl_model.pack(anchor=tk.W, pady=(2, 5))
        ttk.Button(p1_frame, text="Browse Model...", command=self.load_model).pack(anchor=tk.W)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        self.btn_start = ttk.Button(main_frame, text="MULAI SIMULASI", style="Start.TButton", command=self.start_test)
        self.btn_start.pack(fill=tk.X)
        self.btn_start.state(['disabled'])
        
        self.model_path = None

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select Model to Test",
            filetypes=[("Stable Baselines3 Model", "*.zip")]
        )
        if path:
            filename = os.path.basename(path)
            model_name = os.path.splitext(filename)[0]
            self.model_path = path
            self.lbl_model.config(text=f"Loaded: {model_name}", foreground="#4caf50")
            self.btn_start.state(['!disabled'])

    def start_test(self):
        if not self.model_path:
            messagebox.showerror("Error", "Please load a model!")
            return
            
        self.root.quit()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = SoloLauncher(root)
    root.mainloop() 

    if app.model_path:
        try:
            game = SnakeSoloTester(app.model_path, width=70, height=70, cell_size=10)
            game.run(fps=30)
        except Exception as e:
            print("\n--- ERROR SAAT MENJALANKAN TESTER ---")
            traceback.print_exc()
            print("---------------------------------------\n")