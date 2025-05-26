import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import math
import functools
import copy
import time
import threading 
import queue as thread_queue 
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures 
import os

# --- Constants ---
PLAYER_HUMAN = 'X'
PLAYER_AI = 'O'
EMPTY_CELL = ' '
threads_reserved = 2 

try:
    cpu_cores = os.cpu_count()
    NUM_AI_PROCESSES = max(1, (cpu_cores - threads_reserved if cpu_cores and cpu_cores > threads_reserved else 1))
except (ImportError, AttributeError, TypeError, NotImplementedError):
    NUM_AI_PROCESSES = 1 
# Removed print statement from here

COLOR_ROOT_BG = "#EAEAEA"       
COLOR_FRAME_BG = "#FFFFFF"     
COLOR_TEXT_PRIMARY = "#212529" 
COLOR_TEXT_SECONDARY = "#495057" 
COLOR_ACCENT_PRIMARY = "#007BFF" 
COLOR_ACCENT_SECONDARY = "#0056b3" 
COLOR_HINT = "#FFC107"        
COLOR_HINT_BG = "#FFC107"     # Added missing color constant
COLOR_HINT_TEXT = "#212529"   
COLOR_HINT_SUGGEST_BG = "#FFF3CD" 
COLOR_SUCCESS = "#28A745"
COLOR_DANGER = "#DC3545"
COLOR_HUMAN_MOVE_BG = "#E0EFFF" 
COLOR_AI_MOVE_BG = "#FFE0E0"   
COLOR_WIN_HIGHLIGHT = "#B8F5B8" 
COLOR_TEXT_ON_ACCENT = "#FFFFFF" 


# --- Game Logic (Unchanged) ---
def check_win_for_eval(board, player, k_to_win): # ... (same)
    n = len(board)
    for r in range(n):
        for c in range(n - k_to_win + 1):
            if all(board[r][c+i] == player for i in range(k_to_win)): return True
    for c in range(n):
        for r in range(n - k_to_win + 1):
            if all(board[r+i][c] == player for i in range(k_to_win)): return True
    for r in range(n - k_to_win + 1):
        for c in range(n - k_to_win + 1):
            if all(board[r+i][c+i] == player for i in range(k_to_win)): return True
    for r in range(n - k_to_win + 1):
        for c in range(k_to_win - 1, n):
            if all(board[r+i][c-i] == player for i in range(k_to_win)): return True
    return False

def get_winning_line(board, player, k_to_win): # ... (same)
    n = len(board)
    for r_idx in range(n):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx][c_idx+i] == player for i in range(k_to_win)): return [(r_idx, c_idx+i) for i in range(k_to_win)]
    for c_idx in range(n):
        for r_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx] == player for i in range(k_to_win)): return [(r_idx+i, c_idx) for i in range(k_to_win)]
    for r_idx in range(n - k_to_win + 1):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx+i] == player for i in range(k_to_win)): return [(r_idx+i, c_idx+i) for i in range(k_to_win)]
    for r_idx in range(n - k_to_win + 1):
        for c_idx in range(k_to_win - 1, n):
            if all(board[r_idx+i][c_idx-i] == player for i in range(k_to_win)): return [(r_idx+i, c_idx-i) for i in range(k_to_win)]
    return []

def is_board_full(board): # ... (same)
    return all(cell != EMPTY_CELL for row in board for cell in row)

def get_available_moves(board): # ... (same)
    moves = []
    n = len(board)
    mid = n // 2
    if n % 2 == 1 and board[mid][mid] == EMPTY_CELL: moves.append((mid,mid))
    for r in range(n):
        for c in range(n):
            if board[r][c] == EMPTY_CELL and not (n % 2 == 1 and r == mid and c == mid) : moves.append((r,c))
    return moves

# --- Minimax AI (Unchanged) ---
@functools.lru_cache(maxsize=200000) 
def minimax(board_tuple, depth, is_maximizing_for_x, k_to_win, alpha, beta): # ... (same)
    nodes_here = 1; max_depth_here = depth
    board = [list(row) for row in board_tuple]
    if check_win_for_eval(board, PLAYER_HUMAN, k_to_win): return (1000 - depth, nodes_here, max_depth_here)
    if check_win_for_eval(board, PLAYER_AI, k_to_win): return (-1000 + depth, nodes_here, max_depth_here)
    if is_board_full(board): return (0, nodes_here, max_depth_here)
    available_moves_list = get_available_moves(board) 
    if is_maximizing_for_x:
        max_eval = -math.inf
        for r_move, c_move in available_moves_list:
            board[r_move][c_move] = PLAYER_HUMAN 
            board_tuple_next = tuple(map(tuple, board))
            eval_score, child_nodes, child_depth = minimax(board_tuple_next, depth + 1, False, k_to_win, alpha, beta)
            nodes_here += child_nodes; max_depth_here = max(max_depth_here, child_depth)
            board[r_move][c_move] = EMPTY_CELL
            if eval_score > max_eval: max_eval = eval_score
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        return (max_eval, nodes_here, max_depth_here)
    else: 
        min_eval = math.inf
        for r_move, c_move in available_moves_list:
            board[r_move][c_move] = PLAYER_AI 
            board_tuple_next = tuple(map(tuple, board))
            eval_score, child_nodes, child_depth = minimax(board_tuple_next, depth + 1, True, k_to_win, alpha, beta)
            nodes_here += child_nodes; max_depth_here = max(max_depth_here, child_depth)
            board[r_move][c_move] = EMPTY_CELL
            if eval_score < min_eval: min_eval = eval_score
            beta = min(beta, eval_score)
            if beta <= alpha: break
        return (min_eval, nodes_here, max_depth_here)

def evaluate_single_move_for_player_process(board_state_tuple_arg, move_arg, k_to_win_arg, current_player_token, is_human_player_turn_for_hint): # ... (same)
    r, c = move_arg
    temp_board = [list(row) for row in board_state_tuple_arg]
    temp_board[r][c] = current_player_token 
    next_turn_is_human_x = (current_player_token == PLAYER_AI)
    score, nodes_in_branch, depth_in_branch = minimax(
        tuple(map(tuple, temp_board)), 0, next_turn_is_human_x, k_to_win_arg, -math.inf, math.inf
    )
    return {'move': move_arg, 'score': score, 'nodes': nodes_in_branch, 'depth': depth_in_branch}

def find_best_move_generic_manager(board_config, k_to_win_config, progress_q, result_q, player_token_for_move): # ... (same)
    global NUM_AI_PROCESSES 
    aggregated_nodes_explored = 0; aggregated_max_depth = 0
    is_maximizing_search = (player_token_for_move == PLAYER_HUMAN)
    current_best_score_so_far = -math.inf if is_maximizing_search else math.inf 
    best_move_so_far = None 
    board_tuple_config = tuple(map(tuple, board_config))
    available_moves_config = get_available_moves([list(row) for row in board_tuple_config])
    if not available_moves_config:
        result_q.put({'best_move_data': (None, 0), 'top_moves_list': [], 'total_nodes': 0, 'max_search_depth': 0})
        return
    progress_q.put({'total': len(available_moves_config), 'type': 'start'})
    evaluated_moves_details = [] 
    start_time_total_search = time.monotonic() 
    with ProcessPoolExecutor(max_workers=NUM_AI_PROCESSES) as executor:
        future_to_move_eval = {
            executor.submit(evaluate_single_move_for_player_process, board_tuple_config, move, k_to_win_config, player_token_for_move, is_maximizing_search): move
            for move in available_moves_config
        }
        num_completed = 0
        for future in concurrent.futures.as_completed(future_to_move_eval):
            try:
                move_eval_data = future.result() 
                evaluated_moves_details.append(move_eval_data)
                aggregated_nodes_explored += move_eval_data['nodes']
                aggregated_max_depth = max(aggregated_max_depth, move_eval_data['depth'])
                current_move_score = move_eval_data['score']
                if is_maximizing_search: 
                    if current_move_score > current_best_score_so_far: current_best_score_so_far = current_move_score; best_move_so_far = move_eval_data['move']
                else: 
                    if current_move_score < current_best_score_so_far: current_best_score_so_far = current_move_score; best_move_so_far = move_eval_data['move']
                if best_move_so_far is None: best_move_so_far = move_eval_data['move']; current_best_score_so_far = current_move_score
            except Exception as exc:
                err_move = future_to_move_eval[future]
                print(f'Move evaluation for {err_move} generated an exception in process: {exc}')
            num_completed += 1
            time_elapsed_so_far = time.monotonic() - start_time_total_search
            avg_time_per_move = time_elapsed_so_far / num_completed if num_completed > 0 else 0
            moves_remaining = len(available_moves_config) - num_completed
            estimated_time_left = avg_time_per_move * moves_remaining if avg_time_per_move > 0 else float('inf')
            current_nps = aggregated_nodes_explored / time_elapsed_so_far if time_elapsed_so_far > 0.0001 else 0
            progress_q.put({'current': num_completed, 'time_elapsed': time_elapsed_so_far, 'estimated_left': estimated_time_left, 'type': 'progress', 'current_nodes': aggregated_nodes_explored, 'current_max_depth': aggregated_max_depth, 'current_best_score': current_best_score_so_far, 'current_nps': current_nps})
    if best_move_so_far is None and available_moves_config: 
        best_move_so_far = available_moves_config[0]
        fallback_eval = next((em for em in evaluated_moves_details if em['move'] == best_move_so_far), None)
        if fallback_eval: current_best_score_so_far = fallback_eval['score']
        else: current_best_score_so_far = -math.inf if is_maximizing_search else math.inf
    evaluated_moves_details.sort(key=lambda x: x['score'], reverse=is_maximizing_search)
    result_q.put({'best_move_data': (best_move_so_far, current_best_score_so_far), 'top_moves_list': evaluated_moves_details[:5], 'total_nodes': aggregated_nodes_explored, 'max_search_depth': aggregated_max_depth})

# --- Tkinter GUI ---
class TicTacToeGUI:
    def __init__(self, root_window):
        self.root = root_window
        # Print worker info once when GUI initializes
        print(f"Initializing GUI. Using {NUM_AI_PROCESSES} AI worker processes for calculations.") # MOVED HERE
        
        self.root.title(f"Modern Tic-Tac-Toe AI ({NUM_AI_PROCESSES} Proc.)")
        self.root.configure(bg=COLOR_ROOT_BG) 
        try: self.default_font_family = tkfont.nametofont("TkDefaultFont").actual()["family"]
        except (KeyError, AttributeError, tk.TclError): self.default_font_family = "Segoe UI" if "win" in self.root.tk.call("tk", "windowingsystem") else "Helvetica"
        self.fonts = {
            "header": tkfont.Font(family=self.default_font_family, size=16, weight="bold"),
            "status": tkfont.Font(family=self.default_font_family, size=11, weight="bold"),
            "label": tkfont.Font(family=self.default_font_family, size=10),
            "button": tkfont.Font(family=self.default_font_family, size=10, weight="normal"),
            "entry": tkfont.Font(family=self.default_font_family, size=10),
            "info": tkfont.Font(family=self.default_font_family, size=9),
            "info_value": tkfont.Font(family=self.default_font_family, size=9, weight="bold"),
            "move_info": tkfont.Font(family="Consolas", size=9) 
        }
        self.setup_styles() 
        self.board_size = 3; self.k_to_win = 3
        self.current_player = PLAYER_HUMAN
        self.game_board = []; self.buttons = []
        self.game_over = False
        self.calculation_manager_thread = None
        self.progress_queue = thread_queue.Queue() 
        self.result_queue = thread_queue.Queue()
        self.root.attributes("-fullscreen", True); self.is_fullscreen = True 
        self.nodes_explored_var = tk.StringVar(value="0"); self.actual_depth_var = tk.StringVar(value="0")
        self.ai_eval_var = tk.StringVar(value="N/A"); self.status_var = tk.StringVar(value="Set parameters and start game.")
        self.time_taken_var = tk.StringVar(value="N/A"); self.top_moves_var = tk.StringVar(value="Top Move Considerations:\n(After calculation)")
        self.nps_var = tk.StringVar(value="N/A"); self.progress_percent_var = tk.StringVar(value="0.0%")
        self.estimated_time_left_var = tk.StringVar(value="Estimating..."); self.hint_suggestion_var = tk.StringVar(value="") 
        self.setup_ui_layout(); self.root.bind("<F11>", self.toggle_fullscreen); self.root.bind("<Escape>", self.on_closing) 

    def setup_styles(self):
        self.style = ttk.Style()
        available_themes = self.style.theme_names(); current_os = self.root.tk.call("tk", "windowingsystem")
        
        # Attempt to choose a theme that generally has better default contrast or is more customizable
        if 'clam' in available_themes: self.style.theme_use('clam')
        elif current_os == "win32" and 'vista' in available_themes: self.style.theme_use('vista')
        elif current_os == "aqua" and 'aqua' in available_themes: self.style.theme_use('aqua')
        else: self.style.theme_use(available_themes[0] if available_themes else 'default')
        
        print(f"Using ttk theme: {self.style.theme_use()}")

        # General styling
        self.style.configure(".", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"])
        self.style.configure("TFrame", background=COLOR_ROOT_BG)
        self.style.configure("Content.TFrame", background=COLOR_FRAME_BG)
        
        self.style.configure("TLabel", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"], padding=2)
        self.style.configure("Header.TLabel", font=self.fonts["header"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_FRAME_BG)
        self.style.configure("Status.TLabel", font=self.fonts["status"], foreground=COLOR_TEXT_PRIMARY)
        self.style.configure("Info.TLabel", font=self.fonts["info"], foreground=COLOR_TEXT_SECONDARY)
        self.style.configure("InfoValue.TLabel", font=self.fonts["info_value"], foreground=COLOR_TEXT_PRIMARY)
        self.style.configure("HintSuggest.TLabel", font=self.fonts["info_value"], foreground=COLOR_HINT_TEXT, background=COLOR_HINT_SUGGEST_BG, padding=4)
        self.style.configure("MoveInfo.TLabel", font=self.fonts["move_info"], background=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, padding=5, borderwidth=1, relief="solid", bordercolor="#CCCCCC")
        
        # Base TButton - some themes might ignore direct foreground/background configure
        self.style.configure("TButton", 
                             font=self.fonts["button"], 
                             padding=(6, 4), 
                             borderwidth=1) 
                             # relief="raised" # Let theme decide relief for base TButton
                             # foreground=COLOR_TEXT_PRIMARY # Let theme decide base text color
                             # background="#E0E0E0" # Let theme decide base background
        self.style.map("TButton", # Minimal mapping for base TButton
            relief=[('pressed', 'sunken'), ('!pressed', 'raised')], # Common relief mapping
            foreground=[('disabled', COLOR_TEXT_SECONDARY)], # Ensure disabled text is visible
            background=[('disabled', "#E0E0E0")]
        )

        # Accent Button (e.g., Start buttons) - FORCE foreground and background
        self.style.configure("Accent.TButton", 
                             font=self.fonts["button"],
                             padding=(6,4),
                             borderwidth=1,
                             relief="raised", # Explicitly set relief
                             background=COLOR_ACCENT_PRIMARY, # Base background
                             foreground=COLOR_TEXT_ON_ACCENT) # Base foreground
        self.style.map("Accent.TButton",
            background=[('pressed', COLOR_ACCENT_SECONDARY), 
                        ('active', COLOR_ACCENT_SECONDARY), # Darker on active too
                        ('!disabled', COLOR_ACCENT_PRIMARY)], # Normal state background
            foreground=[('pressed', COLOR_TEXT_ON_ACCENT), 
                        ('active', COLOR_TEXT_ON_ACCENT), 
                        ('!disabled', COLOR_TEXT_ON_ACCENT), # Ensure text color for normal state
                        ('disabled', COLOR_TEXT_SECONDARY)]  # Visible disabled text
        )

        # Hint Button - FORCE foreground and background
        self.style.configure("Hint.TButton", 
                             font=self.fonts["button"],
                             padding=(6,4),
                             borderwidth=1,
                             relief="raised", # Explicitly set relief
                             background=COLOR_HINT_BG, 
                             foreground=COLOR_HINT_TEXT)
        self.style.map("Hint.TButton", 
            background=[('active', "#FFD700"), # Slightly different active for yellow
                        ('pressed', "#EAA600"), 
                        ('!disabled', COLOR_HINT_BG)], 
            foreground=[('active', COLOR_HINT_TEXT), 
                        ('pressed', COLOR_HINT_TEXT), 
                        ('!disabled', COLOR_HINT_TEXT), # Ensure text color for normal state
                        ('disabled', COLOR_TEXT_SECONDARY)]
        )

        self.style.configure("TLabelframe", background=COLOR_ROOT_BG, bordercolor=COLOR_ACCENT_SECONDARY, padding=6)
        self.style.configure("TLabelframe.Label", font=self.fonts["label"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_ROOT_BG)
        self.style.configure("TEntry", font=self.fonts["entry"], padding=3, fieldbackground=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, bordercolor="#CCCCCC")
        self.style.configure("Horizontal.TProgressbar", thickness=10, background=COLOR_ACCENT_PRIMARY, troughcolor=COLOR_FRAME_BG, bordercolor=COLOR_ACCENT_SECONDARY)

    def toggle_fullscreen(self, event=None): # Same
        self.is_fullscreen = not self.is_fullscreen; self.root.attributes("-fullscreen", self.is_fullscreen)
        if self.buttons: self.root.after(50, self.create_board_ui_buttons) 

    def setup_ui_layout(self): # Same
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL, style="TPanedwindow")
        self.paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_pane = ttk.Frame(self.paned_window, padding=(10,5), style="Content.TFrame") 
        left_pane.grid_columnconfigure(0, weight=1); left_pane.grid_rowconfigure(0, weight=0); left_pane.grid_rowconfigure(1, weight=0); left_pane.grid_rowconfigure(2, weight=1) 
        self.paned_window.add(left_pane, weight=30) 
        self.board_outer_frame = ttk.Frame(self.paned_window, padding=10, style="TFrame") 
        self.board_outer_frame.grid_rowconfigure(0, weight=1); self.board_outer_frame.grid_columnconfigure(0, weight=1)
        self.paned_window.add(self.board_outer_frame, weight=70) 
        header_label = ttk.Label(left_pane, text=f"Tic-Tac-Toe ({NUM_AI_PROCESSES} Proc.)", style="Header.TLabel"); header_label.configure(background=COLOR_FRAME_BG) 
        header_label.grid(row=0, column=0, pady=(0,15), sticky="w")
        control_frame = ttk.Labelframe(left_pane, text="Game Settings", padding=10); control_frame.configure(style="Content.TLabelframe") 
        control_frame.grid(row=1, column=0, sticky="new", pady=(0,10)); control_frame.grid_columnconfigure(1, weight=1) 
        lbl_board_size = ttk.Label(control_frame, text="Board Size (N x N):"); lbl_board_size.configure(background=COLOR_FRAME_BG)
        lbl_board_size.grid(row=0, column=0, padx=5, pady=3, sticky="w") 
        self.size_entry = ttk.Entry(control_frame, width=5); self.size_entry.insert(0, str(self.board_size))
        self.size_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        lbl_k_row = ttk.Label(control_frame, text="K-in-a-row:"); lbl_k_row.configure(background=COLOR_FRAME_BG)
        lbl_k_row.grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.k_entry = ttk.Entry(control_frame, width=5); self.k_entry.insert(0, str(self.k_to_win))
        self.k_entry.grid(row=1, column=1, padx=5, pady=3, sticky="ew")
        button_frame = ttk.Frame(control_frame, style="Content.TFrame") 
        button_frame.grid(row=2, column=0, columnspan=2, pady=(8,0), sticky="ew") 
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        self.start_human_button = ttk.Button(button_frame, text="▶ You Start", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=True))
        self.start_human_button.grid(row=0, column=0, padx=(0,3), pady=2, sticky="ew")
        self.start_ai_button = ttk.Button(button_frame, text="🤖 AI Starts", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=False))
        self.start_ai_button.grid(row=0, column=1, padx=(3,0), pady=2, sticky="ew")
        self.hint_button = ttk.Button(button_frame, text="💡 Suggest Move", style="Hint.TButton", command=self.get_human_hint, state=tk.DISABLED)
        self.hint_button.grid(row=1, column=0, columnspan=2, pady=(4,0), sticky="ew") 
        status_vis_frame = ttk.Labelframe(left_pane, text="Game & AI Insights", padding=10); status_vis_frame.configure(style="Content.TLabelframe")
        status_vis_frame.grid(row=2, column=0, sticky="nsew", pady=(0,0)); status_vis_frame.grid_columnconfigure(0, weight=1); status_vis_frame.grid_columnconfigure(1, weight=0) 
        self.status_label = ttk.Label(status_vis_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=350) 
        self.status_label.configure(background=COLOR_FRAME_BG); self.status_label.grid(row=0, column=0, columnspan=2, pady=(0,6), sticky="new") 
        progress_frame = ttk.Frame(status_vis_frame, style="Content.TFrame")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,6)) 
        progress_frame.columnconfigure(0, weight=1) 
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_percent_label = ttk.Label(progress_frame, textvariable=self.progress_percent_var, style="Info.TLabel"); self.progress_percent_label.configure(background=COLOR_FRAME_BG)
        stats_grid = ttk.Frame(status_vis_frame, style="Content.TFrame")
        stats_grid.grid(row=2, column=0, columnspan=2, sticky="new", pady=(0,6)); stats_grid.columnconfigure(1, weight=1) 
        row_idx = 0
        for label_text, var, unit in [ ("Move Score:", self.ai_eval_var, ""), ("Time Taken:", self.time_taken_var, "s"), ("Nodes Explored:", self.nodes_explored_var, ""), ("Nodes/Sec (NPS):", self.nps_var, ""), ("Max Depth Searched:", self.actual_depth_var, ""), ("Est. Time Left:", self.estimated_time_left_var, "s")]: 
            lbl = ttk.Label(stats_grid, text=label_text, style="Info.TLabel"); lbl.configure(background=COLOR_FRAME_BG); lbl.grid(row=row_idx, column=0, sticky="w", padx=2, pady=1) 
            val_lbl = ttk.Label(stats_grid, textvariable=var, style="InfoValue.TLabel"); val_lbl.configure(background=COLOR_FRAME_BG); val_lbl.grid(row=row_idx, column=1, sticky="w", padx=2, pady=1)
            row_idx += 1
        score_info_lbl = ttk.Label(stats_grid, text="(Score: >0 X wins, <0 O wins, 0 Draw)", style="Info.TLabel", foreground="#6c757d"); score_info_lbl.configure(background=COLOR_FRAME_BG)
        score_info_lbl.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=2, pady=1)
        self.hint_suggestion_label = ttk.Label(status_vis_frame, textvariable=self.hint_suggestion_var, style="HintSuggest.TLabel", wraplength=350, anchor="center")
        self.hint_suggestion_label.grid(row=row_idx + 1, column=0, columnspan=2, sticky="new", pady=4) 
        self.top_moves_label = ttk.Label(status_vis_frame, textvariable=self.top_moves_var, style="MoveInfo.TLabel", anchor="nw") 
        self.top_moves_label.grid(row=row_idx + 2, column=0, columnspan=2, pady=(8,0), sticky="nsew"); status_vis_frame.grid_rowconfigure(row_idx + 2, weight=1) 
        self.board_frame = ttk.Frame(self.board_outer_frame, style="TFrame"); self.board_frame.grid(row=0, column=0, sticky="") 
        self.status_var.set("Adjust N (2-4 rec.) & K, then Start."); self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self, event=None): # Same
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            if messagebox.askokcancel("Calculation Running", "Calculation in progress. Close anyway?"): self.root.destroy()
            else: return 
        else: self.root.destroy()

    def start_new_game(self, human_starts=True): # Same
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): messagebox.showwarning("Busy", "Calculation in progress."); return
        try:
            n = int(self.size_entry.get()); k = int(self.k_entry.get())
            if not (2 <= n <= 7): messagebox.showerror("Error", "Board N: 2-7."); return
            if n > 4 : 
                 warning_key = f"_warned_slow_n{n}_mp" 
                 if not hasattr(self, warning_key):
                    if not messagebox.askokcancel("Perf. Warning", f"N={n} ({NUM_AI_PROCESSES} proc) can be slow.\nContinue?"): return
                    setattr(self, warning_key, True)
            if not (2 <= k <= n): messagebox.showerror("Error", f"K: 2-N."); return
            self.board_size = n; self.k_to_win = k
        except ValueError: messagebox.showerror("Error", "Invalid N or K."); return
        self.game_board = [[EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = PLAYER_HUMAN if human_starts else PLAYER_AI
        self.game_over = False; minimax.cache_clear(); self.hint_suggestion_var.set("") 
        initial_status = f"{n}x{n}, {k}-in-a-row. {'Your turn (X)' if human_starts else 'AI turn (O)'}."
        self.status_var.set(initial_status)
        self.ai_eval_var.set("N/A"); self.nodes_explored_var.set("0"); self.actual_depth_var.set("0")
        self.time_taken_var.set("N/A"); self.nps_var.set("N/A")
        self.estimated_time_left_var.set("N/A"); self.progress_percent_var.set("0.0%")
        self.top_moves_var.set("Top Move Considerations:\n(After calculation)")
        if self.progress_bar.master.winfo_ismapped(): self.progress_bar.master.grid_forget()
        self.create_board_ui_buttons(); self.update_hint_button_state()
        if not human_starts: self.root.after(100, self.trigger_ai_or_hint_calculation, PLAYER_AI) 

    def update_hint_button_state(self): # Same
        state = tk.NORMAL if not self.game_over and self.current_player == PLAYER_HUMAN and \
           (not self.calculation_manager_thread or not self.calculation_manager_thread.is_alive()) else tk.DISABLED
        self.hint_button.config(state=state)
            
    def get_human_hint(self): # Same
        if self.game_over or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): messagebox.showinfo("Busy", "Calculation in progress."); return
        self.hint_suggestion_var.set("Calculating your best move..."); self.trigger_ai_or_hint_calculation(PLAYER_HUMAN) 

    def trigger_ai_or_hint_calculation(self, player_to_calculate_for): # Same
        is_hint = (player_to_calculate_for == PLAYER_HUMAN)
        if is_hint: self.status_var.set("Calculating hint for Human (X)...")
        else: self.status_var.set("AI (O) is thinking (using processes)...")
        self.top_moves_var.set("Top Move Considerations:\nCalculating...")
        self.disable_board_buttons(); self.hint_button.config(state=tk.DISABLED) 
        self.progress_bar.master.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,5)) 
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,5)); self.progress_percent_label.grid(row=0, column=1, sticky="w", padx=(0,0))
        self.progress_bar["value"] = 0; self.progress_percent_var.set("0.0%")
        self.estimated_time_left_var.set("Estimating..."); self.root.config(cursor="watch"); self.root.update_idletasks()
        while not self.progress_queue.empty(): self.progress_queue.get_nowait()
        while not self.result_queue.empty(): self.result_queue.get_nowait()
        self.calculation_manager_thread = threading.Thread(target=find_best_move_generic_manager, args=(self.game_board, self.k_to_win, self.progress_queue, self.result_queue, player_to_calculate_for), daemon=True)
        self.ai_start_time = time.monotonic(); self._current_calculation_for_player = player_to_calculate_for 
        self.calculation_manager_thread.start(); self.check_ai_progress()

    def create_board_ui_buttons(self): # Same
        for widget in self.board_frame.winfo_children(): widget.destroy()
        self.buttons = []
        self.root.update_idletasks() 
        available_width = self.board_outer_frame.winfo_width(); available_height = self.board_outer_frame.winfo_height()
        if available_width < 50 or available_height < 50: available_width = max(self.root.winfo_height() * 0.55, 250); available_height = max(self.root.winfo_height() * 0.55, 250)
        cell_dim = min(available_width // self.board_size, available_height // self.board_size)
        cell_dim = max(35, cell_dim - 4) 
        btn_font_size = max(10, int(cell_dim * 0.35)) 
        board_button_font = tkfont.Font(family=self.default_font_family, size=btn_font_size, weight="bold")
        for r in range(self.board_size):
            row_buttons = []
            self.board_frame.grid_rowconfigure(r, weight=1, minsize=cell_dim)
            for c in range(self.board_size):
                self.board_frame.grid_columnconfigure(c, weight=1, minsize=cell_dim)
                button = tk.Button(self.board_frame, text=EMPTY_CELL, font=board_button_font, relief="flat", borderwidth=1, bg=COLOR_FRAME_BG, activebackground="#DDDDDD", fg=COLOR_TEXT_PRIMARY, command=lambda r_idx=r, c_idx=c: self.handle_cell_click(r_idx, c_idx))
                button.grid(row=r, column=c, sticky="nsew", padx=1, pady=1) 
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board_button_states()

    def handle_cell_click(self, r, c): # Same
        if self.game_over or self.game_board[r][c] != EMPTY_CELL or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): messagebox.showinfo("Wait", "Calculation in progress."); return
        self.hint_suggestion_var.set(""); self.make_move(r, c, PLAYER_HUMAN)
        if self.check_game_status(): return
        self.current_player = PLAYER_AI; self.update_hint_button_state()
        self.trigger_ai_or_hint_calculation(PLAYER_AI) 

    def check_ai_progress(self): # Same
        update_interval_ms = 30 
        try:
            progress_update = self.progress_queue.get_nowait()
            if progress_update.get('type') == 'start':
                self.progress_bar["maximum"] = progress_update['total']
                self.progress_percent_var.set("0.0%"); self.estimated_time_left_var.set("Estimating...")
                self.nodes_explored_var.set("0"); self.actual_depth_var.set("0") 
                self.ai_eval_var.set("N/A"); self.nps_var.set("0")
            elif progress_update.get('type') == 'progress':
                current_val = progress_update['current']; self.progress_bar["value"] = current_val
                max_val = self.progress_bar["maximum"]
                if max_val > 0: percent = (current_val / max_val) * 100
                else: percent = 100.0 if current_val > 0 else 0.0
                self.progress_percent_var.set(f"{percent:.1f}%") 
                est_left = progress_update.get('estimated_left', float('inf'))
                if est_left != float('inf') and est_left >= 0: self.estimated_time_left_var.set(f"{est_left:.1f}") 
                elif current_val == max_val : self.estimated_time_left_var.set("0.0")
                else: self.estimated_time_left_var.set("Calculating...")
                if 'current_nodes' in progress_update: self.nodes_explored_var.set(f"{progress_update['current_nodes']:,}")
                if 'current_max_depth' in progress_update: self.actual_depth_var.set(f"{progress_update['current_max_depth']}")
                if 'current_best_score' in progress_update:
                    score_val = progress_update['current_best_score']
                    if score_val not in (math.inf, -math.inf): self.ai_eval_var.set(f"{score_val:.0f}")
                    else: self.ai_eval_var.set("...") 
                if 'current_nps' in progress_update: self.nps_var.set(f"{progress_update['current_nps']:,.0f}")
                if 'time_elapsed' in progress_update: self.time_taken_var.set(f"{progress_update['time_elapsed']:.3f}")
        except thread_queue.Empty: pass 
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            self.root.after(update_interval_ms, self.check_ai_progress) 
        elif self.calculation_manager_thread: 
            self.handle_calculation_result(); self.calculation_manager_thread = None 

    def handle_calculation_result(self): # Same
        self.root.config(cursor=""); self.progress_bar.master.grid_forget()
        time_taken_calc = time.monotonic() - self.ai_start_time 
        was_hint_calculation = (self._current_calculation_for_player == PLAYER_HUMAN)
        try:
            ai_result_data = self.result_queue.get_nowait() 
            best_move, best_score = ai_result_data['best_move_data']
            top_moves_list = ai_result_data['top_moves_list']
            total_nodes = ai_result_data['total_nodes'] 
            max_search_depth = ai_result_data['max_search_depth'] 
        except thread_queue.Empty: 
            self.status_var.set("Error: No result."); self.enable_board_buttons(); self.update_hint_button_state()
            return
        self.time_taken_var.set(f"{time_taken_calc:.3f}"); self.nodes_explored_var.set(f"{total_nodes:,}")
        nps_val = total_nodes / time_taken_calc if time_taken_calc > 0.0001 else 0 
        self.nps_var.set(f"{nps_val:,.0f}"); self.actual_depth_var.set(f"{max_search_depth}")
        self.ai_eval_var.set(f"{best_score:.0f}"); self.estimated_time_left_var.set("Done.") 
        perspective = "Human (X)" if was_hint_calculation else "AI (O)"
        top_moves_text = f"Top {perspective} Considerations (Move: Score | Nodes | Depth):\n"
        if top_moves_list:
            for item in top_moves_list: 
                move_str = f"({item['move'][0]},{item['move'][1]})"; score_str = f"{item['score']:.0f}"
                nodes_str = f"N:{item['nodes']:,}"; depth_str = f"D:{item['depth']}"
                top_moves_text += f"  {move_str:<7}: {score_str:<5} | {nodes_str:<12}| {depth_str}\n" 
        else: top_moves_text += "  (N/A)\n"
        self.top_moves_var.set(top_moves_text.strip())
        if was_hint_calculation:
            self.status_var.set(f"Hint for X: Best is {best_move} (Score: {best_score:.0f}).")
            self.hint_suggestion_var.set(f"Suggested for X: {best_move} (Score: {best_score:.0f})")
            if best_move and self.buttons[best_move[0]][best_move[1]]['state'] == tk.NORMAL:
                self.buttons[best_move[0]][best_move[1]].config(bg=COLOR_HINT_SUGGEST_BG, relief="raised")
                self.root.after(3000, lambda r=best_move[0], c=best_move[1]: self.clear_hint_highlight(r,c))
            self.enable_board_buttons(); self.update_hint_button_state()
        else: 
            self.hint_suggestion_var.set("") 
            if best_move:
                self.make_move(best_move[0], best_move[1], PLAYER_AI)
                if self.check_game_status(): self.enable_board_buttons(); self.update_hint_button_state(); return
                self.current_player = PLAYER_HUMAN; self.status_var.set("Your turn (X).")
            else: 
                 if is_board_full(self.game_board) and not get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win) and not get_winning_line(self.game_board, PLAYER_AI, self.k_to_win): self.status_var.set("It's a Draw!")
                 else: self.status_var.set("AI error.") 
                 self.game_over = True
            self.enable_board_buttons(); self.update_hint_button_state()

    def clear_hint_highlight(self, r, c): # Same
        if self.game_board[r][c] == EMPTY_CELL and self.buttons[r][c]['bg'] == COLOR_HINT_SUGGEST_BG : 
            self.buttons[r][c].config(bg=COLOR_FRAME_BG, relief="flat")

    def make_move(self, r, c, player): # Same
        self.game_board[r][c] = player; btn = self.buttons[r][c]
        btn.config(text=player, state=tk.DISABLED, relief=tk.SUNKEN) 
        player_color_bg = COLOR_HUMAN_MOVE_BG if player == PLAYER_HUMAN else COLOR_AI_MOVE_BG
        player_fg_color = COLOR_ACCENT_SECONDARY if player == PLAYER_HUMAN else COLOR_DANGER
        btn.config(disabledforeground=player_fg_color, background=player_color_bg) 

    def check_game_status(self): # Same
        human_wins_line = get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win)
        ai_wins_line = get_winning_line(self.game_board, PLAYER_AI, self.k_to_win)
        changed_game_over_state = False
        if not self.game_over: 
            if human_wins_line or ai_wins_line or is_board_full(self.game_board): changed_game_over_state = True
        if human_wins_line: self.status_var.set("You (X) Win!"); self.game_over = True; self.highlight_winning_line(human_wins_line)
        elif ai_wins_line: self.status_var.set("AI (O) Wins!"); self.game_over = True; self.highlight_winning_line(ai_wins_line)
        elif is_board_full(self.game_board): self.status_var.set("It's a Draw!"); self.game_over = True
        if changed_game_over_state or self.game_over: self.disable_board_buttons(); self.update_hint_button_state()
        return self.game_over

    def update_board_button_states(self): # Same
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                text = self.game_board[r_idx][c_idx]; button = self.buttons[r_idx][c_idx] 
                current_bg = button.cget('bg'); is_hint_highlighted = (current_bg == COLOR_HINT_SUGGEST_BG)
                if text == EMPTY_CELL:
                    if not is_hint_highlighted : button.config(text=text, bg=COLOR_FRAME_BG, relief="flat", state=tk.NORMAL, fg=COLOR_TEXT_PRIMARY)
                    else: button.config(text=text, state=tk.NORMAL, fg=COLOR_TEXT_PRIMARY) 
                else: 
                    button.config(text=text, state=tk.DISABLED, relief="sunken")
                    player_color_bg = COLOR_HUMAN_MOVE_BG if text == PLAYER_HUMAN else COLOR_AI_MOVE_BG
                    player_fg_color = COLOR_ACCENT_SECONDARY if text == PLAYER_HUMAN else COLOR_DANGER
                    button.config(disabledforeground=player_fg_color, background=player_color_bg)

    def disable_board_buttons(self): # Same
        for r_buttons in self.buttons:
            for button in r_buttons:
                if button['state'] == tk.NORMAL: button.config(state=tk.DISABLED)

    def enable_board_buttons(self): # Same
        if self.game_over: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): return 
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.game_board[r][c] == EMPTY_CELL: self.buttons[r][c].config(state=tk.NORMAL)

    def highlight_winning_line(self, winning_cells): # Same
        for r,c in winning_cells: self.buttons[r][c].config(background=COLOR_WIN_HIGHLIGHT, relief=tk.GROOVE) 

# --- Main Execution ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = TicTacToeGUI(main_root)
    main_root.mainloop()