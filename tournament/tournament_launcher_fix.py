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

class SnakeTournament1v1:
    def __init__(self, model1_path, model2_path, round_timer_mins, width=70, height=70, cell_size=10):
        self.width = width
        self.height = height
        
        # --- TOURNAMENT FEATURES ---
        self.name1 = os.path.splitext(os.path.basename(model1_path))[0]
        self.name2 = os.path.splitext(os.path.basename(model2_path))[0]
        
        self.match_wins1 = 0
        self.match_wins2 = 0
        self.round_number = 1
        
        # --- TIMER FEATURE ---
        if round_timer_mins == "Unlimited":
            self.round_time_limit = None
        else:
            self.round_time_limit = int(round_timer_mins) * 60 * 1000 
            
        # --- LOAD MODELS (ANTI-DEADLOCK & SAFE UNPICKLE) ---
        # Mengabaikan parameter training yang tidak diperlukan saat inference (main game)
        safe_custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        
        print(f"Loading Model 1 ({self.name1})...")
        try:
            # Memaksa load ke CPU untuk menghindari hang saat inisialisasi Environment PyTorch
            self.model1 = PPO.load(model1_path, device="cpu", custom_objects=safe_custom_objects)
            self.model1_loaded = True
            print("Model 1 Loaded Successfully!")
        except Exception as e:
            print(f"Error loading Model 1: {e}")
            self.model1 = None
            self.model1_loaded = False

        print(f"Loading Model 2 ({self.name2})...")
        try:
            self.model2 = PPO.load(model2_path, device="cpu", custom_objects=safe_custom_objects)
            self.model2_loaded = True
            print("Model 2 Loaded Successfully!")
        except Exception as e:
            print(f"Error loading Model 2: {e}")
            self.model2 = None
            self.model2_loaded = False
        
        # Colors - Modernized Palette
        self.BLACK = (10, 10, 15)
        self.WHITE = (250, 250, 250)
        self.GRID_COLOR = (255, 255, 255, 30) 
        
        # Neon Colors for snakes
        self.S1_HEAD = (0, 255, 128)
        self.S1_BODY = (0, 200, 100)
        
        self.S2_HEAD = (0, 191, 255)
        self.S2_BODY = (0, 150, 255)
        
        self.FOOD_COLOR = (255, 60, 60)
        self.MAZE_COLOR = (120, 120, 140)

        pygame.init()
        
        info = pygame.display.Info()
        self.screen_w = info.current_w
        self.screen_h = info.current_h
        
        self.cell_size = self.screen_h // self.height
        
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.FULLSCREEN)
        pygame.display.set_caption("Snake AI Tournament: 1 vs 1")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 42)
        self.small_font = pygame.font.Font(None, 28)
        self.tiny_font = pygame.font.Font(None, 20)
        
        try:
            bg_surface = pygame.image.load("bg.jpeg").convert()
            self.bg_image = pygame.transform.smoothscale(bg_surface, (self.screen_w, self.screen_h))
        except pygame.error as e:
            print(f"Peringatan: Gambar 'bg.jpeg' tidak ditemukan. Menggunakan warna solid. ({e})")
            self.bg_image = None
            
        self.arena_surface = pygame.Surface((self.width * self.cell_size, self.height * self.cell_size), pygame.SRCALPHA)

        self.setup_maze()
        self.reset()
        
    def setup_maze(self):
        self.maze = set()
        for x in range(30, 41):
            self.maze.add((x, 20))
            self.maze.add((x, 50))
        for y in range(30, 41):
            self.maze.add((20, y))
            self.maze.add((50, y))

    def reset(self):
        self.snake1 = [(10, 10), (9, 10), (8, 10)] if self.model1_loaded else []
        self.dir1 = Direction.RIGHT
        self.score1 = 0
        self.alive1 = self.model1_loaded
        
        self.snake2 = [(60, 60), (61, 60), (62, 60)] if self.model2_loaded else []
        self.dir2 = Direction.LEFT
        self.score2 = 0
        self.alive2 = self.model2_loaded
        
        self.food = self._generate_food()
        self.game_over = False
        self.winner = None
        
        self.round_start_time = pygame.time.get_ticks()
        self.time_left = self.round_time_limit
        self.time_up = False

        if not self.alive1 and not self.alive2:
            self.game_over = True
            self.winner = "No valid models"

    def _generate_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake1 and food not in self.snake2 and food not in self.maze:
                return food
                
    def _get_obs(self, snake_idx):
        if snake_idx == 1:
            snake = self.snake1
            direction = self.dir1
            alive = self.alive1
        else:
            snake = self.snake2
            direction = self.dir2
            alive = self.alive2
            
        if not alive or not snake:
            return np.zeros(11, dtype=np.float32)

        head = snake[0]
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_dir_idx = directions.index(direction)

        danger_straight = self._is_collision(head, direction, snake_idx)
        danger_right = self._is_collision(head, directions[(current_dir_idx + 1) % 4], snake_idx)
        danger_left = self._is_collision(head, directions[(current_dir_idx - 1) % 4], snake_idx)

        dir_up = direction == Direction.UP
        dir_right = direction == Direction.RIGHT
        dir_down = direction == Direction.DOWN
        dir_left = direction == Direction.LEFT

        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]

        return np.array([
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_up, food_down, food_left, food_right
        ], dtype=np.float32)

    def _is_collision(self, position, direction, snake_idx):
        x, y = position
        
        if direction == Direction.UP: y -= 1
        elif direction == Direction.DOWN: y += 1
        elif direction == Direction.LEFT: x -= 1
        elif direction == Direction.RIGHT: x += 1
            
        new_pos = (x, y)

        if x < 0 or x >= self.width or y < 0 or y >= self.height: return True
        if new_pos in self.maze: return True
        
        snake = self.snake1 if snake_idx == 1 else self.snake2
        if new_pos in snake: return True
        
        other_snake = self.snake2 if snake_idx == 1 else self.snake1
        if new_pos in other_snake: return True
        
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
        
        if self.round_time_limit is not None:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.round_start_time
            self.time_left = max(0, self.round_time_limit - elapsed)
            
            if self.time_left == 0:
                self.time_up = True
                self.alive1 = False
                self.alive2 = False
        
        if self.alive1 and self.model1_loaded:
            obs1 = self._get_obs(1)
            action1, _ = self.model1.predict(obs1, deterministic=True)
            self.dir1 = self._update_direction(action1, self.dir1)
            
        if self.alive2 and self.model2_loaded:
            obs2 = self._get_obs(2)
            action2, _ = self.model2.predict(obs2, deterministic=True)
            self.dir2 = self._update_direction(action2, self.dir2)
            
        new_head1 = None
        new_head2 = None
        
        if self.alive1:
            hx, hy = self.snake1[0]
            if self.dir1 == Direction.UP: hy -= 1
            elif self.dir1 == Direction.DOWN: hy += 1
            elif self.dir1 == Direction.LEFT: hx -= 1
            elif self.dir1 == Direction.RIGHT: hx += 1
            new_head1 = (hx, hy)
            
        if self.alive2:
            hx, hy = self.snake2[0]
            if self.dir2 == Direction.UP: hy -= 1
            elif self.dir2 == Direction.DOWN: hy += 1
            elif self.dir2 == Direction.LEFT: hx -= 1
            elif self.dir2 == Direction.RIGHT: hx += 1
            new_head2 = (hx, hy)
            
        die1 = False
        die2 = False
        
        if self.alive1 and self.alive2 and new_head1 == new_head2:
            die1 = die2 = True
            
        if self.alive1 and not die1:
            if (new_head1[0] < 0 or new_head1[0] >= self.width or
                new_head1[1] < 0 or new_head1[1] >= self.height or
                new_head1 in self.maze or
                new_head1 in self.snake1 or
                (self.alive2 and new_head1 in self.snake2)):
                die1 = True
                
        if self.alive2 and not die2:
            if (new_head2[0] < 0 or new_head2[0] >= self.width or
                new_head2[1] < 0 or new_head2[1] >= self.height or
                new_head2 in self.maze or
                new_head2 in self.snake2 or
                (self.alive1 and new_head2 in self.snake1)):
                die2 = True
                
        if die1: self.alive1 = False
        if die2: self.alive2 = False
        
        if not self.alive1 and not self.alive2:
            self.game_over = True
            if self.score1 > self.score2: 
                self.winner = f"{self.name1} (Green)"
                self.match_wins1 += 1
            elif self.score2 > self.score1: 
                self.winner = f"{self.name2} (Blue)"
                self.match_wins2 += 1
            else: 
                self.winner = "Draw"
            return
            
        food_eaten = False
        if self.alive1:
            self.snake1.insert(0, new_head1)
            if new_head1 == self.food:
                self.score1 += 1
                food_eaten = True
            else:
                self.snake1.pop()
                
        if self.alive2:
            self.snake2.insert(0, new_head2)
            if new_head2 == self.food and not food_eaten:
                self.score2 += 1
                food_eaten = True
            elif new_head2 == self.food and food_eaten:
                pass
            else:
                self.snake2.pop()
                
        if food_eaten:
            self.food = self._generate_food()

    def render(self):
        if self.bg_image:
            self.screen.blit(self.bg_image, (0, 0))
        else:
            self.screen.fill(self.BLACK)

        self.arena_surface.fill((10, 10, 15, 180)) 
        
        for x in range(0, self.width * self.cell_size, self.cell_size):
            pygame.draw.line(self.arena_surface, self.GRID_COLOR, (x, 0), (x, self.height * self.cell_size))
        for y in range(0, self.height * self.cell_size, self.cell_size):
            pygame.draw.line(self.arena_surface, self.GRID_COLOR, (0, y), (self.width * self.cell_size, y))

        for mx, my in self.maze:
            rect = pygame.Rect(mx * self.cell_size, my * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.arena_surface, self.MAZE_COLOR, rect)
            
        food_rect = pygame.Rect(self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.arena_surface, self.FOOD_COLOR, food_rect)

        if self.model1_loaded and (self.alive1 or self.game_over):
            for i, (x, y) in enumerate(self.snake1):
                color = self.S1_HEAD if i == 0 else self.S1_BODY
                if not self.alive1: color = (80, 80, 80)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.arena_surface, color, rect)
                pygame.draw.rect(self.arena_surface, (255,255,255,100), rect, 1)
                
        if self.model2_loaded and (self.alive2 or self.game_over):
            for i, (x, y) in enumerate(self.snake2):
                color = self.S2_HEAD if i == 0 else self.S2_BODY
                if not self.alive2: color = (80, 80, 80)
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.arena_surface, color, rect)
                pygame.draw.rect(self.arena_surface, (255,255,255,100), rect, 1)

        self.screen.blit(self.arena_surface, (0, 0))

        ui_x = self.width * self.cell_size
        ui_width = self.screen_w - ui_x
        
        ui_overlay = pygame.Surface((ui_width, self.screen_h), pygame.SRCALPHA)
        ui_overlay.fill((15, 15, 25, 220)) 
        self.screen.blit(ui_overlay, (ui_x, 0))
        
        center_ui_x = ui_x + (ui_width // 2)
        
        tt = self.font.render(f"ROUND {self.round_number}", True, (255, 215, 0))
        self.screen.blit(tt, (center_ui_x - tt.get_width() // 2, 50))
        
        wt = self.font.render(f"Wins: {self.match_wins1} - {self.match_wins2}", True, self.WHITE)
        self.screen.blit(wt, (center_ui_x - wt.get_width() // 2, 90))
        
        if self.round_time_limit is not None:
            seconds_left = self.time_left // 1000
            mins = seconds_left // 60
            secs = seconds_left % 60
            timer_color = self.WHITE if seconds_left > 10 else (255, 50, 50)
            timer_text = self.font.render(f"Time: {mins:02d}:{secs:02d}", True, timer_color)
            self.screen.blit(timer_text, (center_ui_x - timer_text.get_width() // 2, 130))
        
        p1_title = self.font.render(f"Player 1: {self.name1}", True, self.S1_HEAD)
        self.screen.blit(p1_title, (ui_x + 30, 200))
        if self.model1_loaded:
            p1_score = self.font.render(f"Score: {self.score1}", True, self.WHITE)
            self.screen.blit(p1_score, (ui_x + 30, 240))
            p1_status = self.font.render(f"Status: {'ALIVE' if self.alive1 else 'DEAD'}", True, self.S1_HEAD if self.alive1 else (150, 150, 150))
            self.screen.blit(p1_status, (ui_x + 30, 280))
        else:
            p1_status = self.font.render("Status: LOAD FAILED", True, (255, 60, 60))
            self.screen.blit(p1_status, (ui_x + 30, 240))
        
        p2_title = self.font.render(f"Player 2: {self.name2}", True, self.S2_HEAD)
        self.screen.blit(p2_title, (ui_x + 30, 400))
        if self.model2_loaded:
            p2_score = self.font.render(f"Score: {self.score2}", True, self.WHITE)
            self.screen.blit(p2_score, (ui_x + 30, 440))
            p2_status = self.font.render(f"Status: {'ALIVE' if self.alive2 else 'DEAD'}", True, self.S2_HEAD if self.alive2 else (150, 150, 150))
            self.screen.blit(p2_status, (ui_x + 30, 480))
        else:
            p2_status = self.font.render("Status: LOAD FAILED", True, (255, 60, 60))
            self.screen.blit(p2_status, (ui_x + 30, 440))
        
        campus_text_1 = self.small_font.render("Institut Teknologi dan Bisnis Stikom Ambon", True, (200, 200, 200))
        campus_text_2 = self.tiny_font.render("Program Studi S1 Teknik Informatika", True, (150, 150, 150))
        self.screen.blit(campus_text_1, (center_ui_x - campus_text_1.get_width() // 2, self.screen_h - 140))
        self.screen.blit(campus_text_2, (center_ui_x - campus_text_2.get_width() // 2, self.screen_h - 110))

        menu_info = self.tiny_font.render("Press 'M' to Return to Menu", True, (100, 100, 100))
        self.screen.blit(menu_info, (center_ui_x - menu_info.get_width() // 2, self.screen_h - 50))
        quit_info = self.tiny_font.render("Press 'ESC' to Quit Game", True, (100, 100, 100))
        self.screen.blit(quit_info, (center_ui_x - quit_info.get_width() // 2, self.screen_h - 25))

        if self.game_over:
            go_overlay = pygame.Surface((ui_x, self.height * self.cell_size), pygame.SRCALPHA)
            go_overlay.fill((0, 0, 0, 190))
            self.screen.blit(go_overlay, (0, 0))
            
            go_text = "TIME'S UP!" if getattr(self, 'time_up', False) else f"ROUND {self.round_number} OVER"
            go_t = self.font.render(go_text, True, (255, 60, 60) if getattr(self, 'time_up', False) else self.WHITE)
            win_t = self.font.render(f"Round Winner: {self.winner}", True, (255, 215, 0))
            score_t = self.font.render(f"Series: {self.name1} {self.match_wins1} - {self.match_wins2} {self.name2}", True, self.WHITE)
            
            alpha = (pygame.time.get_ticks() % 1000) // 4
            restart_t = self.small_font.render("Press SPACE for Next Round", True, (255, 255, 255))
            restart_t.set_alpha(alpha)
            
            cx = ui_x // 2
            cy = (self.height * self.cell_size) // 2
            
            self.screen.blit(go_t, go_t.get_rect(center=(cx, cy - 60)))
            self.screen.blit(win_t, win_t.get_rect(center=(cx, cy - 20)))
            self.screen.blit(score_t, score_t.get_rect(center=(cx, cy + 20)))
            self.screen.blit(restart_t, restart_t.get_rect(center=(cx, cy + 80)))

        pygame.display.flip()

    def run(self, fps=30):
        running = True
        return_to_menu = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_m: 
                        running = False
                        return_to_menu = True
                    if event.key == pygame.K_SPACE and self.game_over:
                        self.round_number += 1
                        self.reset()
                        
            self.step()
            self.render()
            self.clock.tick(fps)
            
        pygame.quit()
        return return_to_menu 


class TournamentLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake AI Tournament - 1v1 Arena Launcher")
        self.root.geometry("600x560")
        self.root.configure(bg="#0f111a") 
        
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_col = "#0f111a"
        fg_col = "#e0e0e0"
        btn_col = "#1f2233"
        btn_hover = "#2a2d42"
        accent = "#00e5ff"
        
        style.configure("TFrame", background=bg_col)
        style.configure("TLabel", background=bg_col, foreground=fg_col, font=("Segoe UI", 12))
        style.configure("Header.TLabel", font=("Segoe UI", 24, "bold"), foreground=accent)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground="#8a8a8a")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"), foreground="#a0a0a0")
        style.configure("Footer.TLabel", font=("Segoe UI", 9), foreground="#6c7086", background=bg_col)
        
        style.configure("TButton", 
                       font=("Segoe UI", 11, "bold"), 
                       background=btn_col, 
                       foreground=fg_col,
                       borderwidth=1,
                       bordercolor="#333",
                       focuscolor=btn_col,
                       padding=10)
        style.map("TButton", background=[("active", btn_hover)])
        
        style.configure("Start.TButton", background="#00b894", foreground="#ffffff", font=("Segoe UI", 14, "bold"))
        style.map("Start.TButton", background=[("active", "#55efc4")])

        main_frame = ttk.Frame(root, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="SNAKE AI TOURNAMENT", style="Header.TLabel", anchor="center").pack(fill=tk.X, pady=(0, 5))
        ttk.Label(main_frame, text="Load your .zip models trained via Stable Baselines3", style="SubHeader.TLabel", anchor="center").pack(fill=tk.X, pady=(0, 20))
        
        p1_frame = ttk.Frame(main_frame)
        p1_frame.pack(fill=tk.X, pady=10)
        ttk.Label(p1_frame, text="Player 1 (Green Snake):", font=("Segoe UI", 12, "bold"), foreground="#00b894").pack(anchor=tk.W)
        self.lbl_p1 = ttk.Label(p1_frame, text="No model selected", style="Status.TLabel")
        self.lbl_p1.pack(anchor=tk.W, pady=(2, 5))
        ttk.Button(p1_frame, text="Browse Model 1...", command=lambda: self.load_model(1)).pack(anchor=tk.W)
        
        p2_frame = ttk.Frame(main_frame)
        p2_frame.pack(fill=tk.X, pady=10)
        ttk.Label(p2_frame, text="Player 2 (Blue Snake):", font=("Segoe UI", 12, "bold"), foreground="#00cec9").pack(anchor=tk.W)
        self.lbl_p2 = ttk.Label(p2_frame, text="No model selected", style="Status.TLabel")
        self.lbl_p2.pack(anchor=tk.W, pady=(2, 5))
        ttk.Button(p2_frame, text="Browse Model 2...", command=lambda: self.load_model(2)).pack(anchor=tk.W)
        
        timer_frame = ttk.Frame(main_frame)
        timer_frame.pack(fill=tk.X, pady=20)
        ttk.Label(timer_frame, text="Round Timer (Minutes):", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        
        self.timer_var = tk.StringVar(value="2")
        timer_combo = ttk.Combobox(timer_frame, textvariable=self.timer_var, values=["1", "2", "3", "5", "Unlimited"], width=10, state="readonly")
        timer_combo.pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        self.btn_start = ttk.Button(main_frame, text="START ARENA MATCH", style="Start.TButton", command=self.start_tournament)
        self.btn_start.pack(fill=tk.X, pady=5)
        self.btn_start.state(['disabled'])

        footer_text = "Institut Teknologi dan Bisnis Stikom Ambon\nProgram Studi S1 Teknik Informatika"
        ttk.Label(main_frame, text=footer_text, style="Footer.TLabel", justify="center", anchor="center").pack(fill=tk.X, side=tk.BOTTOM, pady=15)
        
        self.model1_path = None
        self.model2_path = None
        self.start_game = False

    def load_model(self, player_num):
        path = filedialog.askopenfilename(
            title=f"Select Model for Player {player_num}",
            filetypes=[("Stable Baselines3 Model", "*.zip")]
        )
        if path:
            filename = os.path.basename(path)
            model_name = os.path.splitext(filename)[0]
            if player_num == 1:
                self.model1_path = path
                self.lbl_p1.config(text=f"Loaded: {model_name}", foreground="#00b894")
            else:
                self.model2_path = path
                self.lbl_p2.config(text=f"Loaded: {model_name}", foreground="#00cec9")
                
            self.check_ready()

    def check_ready(self):
        if self.model1_path and self.model2_path:
            self.btn_start.state(['!disabled'])

    def start_tournament(self):
        if not self.model1_path or not self.model2_path:
            messagebox.showerror("Error", "Please load models for both players!")
            return
            
        self.start_game = True
        
        # --- PERBAIKAN SIKLUS DEADLOCK OS ---
        # 1. Tarik GUI dari layer OS tanpa menghancurkan thread utamanya (mencegah Wayland Crash)
        self.root.withdraw() 
        # 2. Hentikan antrean eksekusi agar script bisa melanjutkan logika ke bagian '__main__'
        self.root.quit()     


if __name__ == '__main__':
    while True: 
        root = tk.Tk()
        app = TournamentLauncher(root)
        root.mainloop() 
    
        if app.start_game and app.model1_path and app.model2_path:
            try:
                timer_val = app.timer_var.get()
                game = SnakeTournament1v1(app.model1_path, app.model2_path, round_timer_mins=timer_val, width=70, height=70, cell_size=10)
                return_to_menu = game.run(fps=30)
                
                # Pembersihan root dilakukan SECARA AMAN setelah sesi game usai sepenuhnya
                root.destroy() 
                
                if not return_to_menu:
                    break 
            except Exception as e:
                print("\n--- ERROR SAAT MENJALANKAN TURNAMEN ---")
                traceback.print_exc()
                print("---------------------------------------\n")
                root.destroy() # Pembersihan fallback jika eksekusi Pygame error di tengah jalan
                break 
        else:
            break