import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import math
import functools
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
threads_reserved = 3

try:
    cpu_cores = os.cpu_count()
    NUM_AI_PROCESSES = max(1, (cpu_cores - threads_reserved if cpu_cores and cpu_cores > threads_reserved else 1))
except (ImportError, AttributeError, TypeError, NotImplementedError):
    NUM_AI_PROCESSES = 1

COLOR_ROOT_BG = "#EAEAEA"
COLOR_FRAME_BG = "#FFFFFF"
COLOR_TEXT_PRIMARY = "#212529"
COLOR_TEXT_SECONDARY = "#495057"
COLOR_ACCENT_PRIMARY = "#007BFF"
COLOR_ACCENT_SECONDARY = "#0056b3"
COLOR_HINT = "#FFC107"
COLOR_HINT_BG = "#FFC107"
COLOR_HINT_TEXT = "#212529"
COLOR_HINT_SUGGEST_BG = "#FFF3CD"
COLOR_SUCCESS = "#28A745"
COLOR_DANGER = "#DC3545"
COLOR_HUMAN_MOVE_BG = "#E0EFFF"
COLOR_AI_MOVE_BG = "#FFE0E0"
COLOR_WIN_HIGHLIGHT = "#B8F5B8"
COLOR_TEXT_ON_ACCENT = "#FFFFFF"


# --- Symmetry Helper Functions ---
def _rotate_board_tuple(board_tuple):
    """Rotates a board tuple 90 degrees clockwise."""
    n = len(board_tuple)
    new_board_list = [[EMPTY_CELL for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            new_board_list[c][n - 1 - r] = board_tuple[r][c]
    return tuple(map(tuple, new_board_list))

def _reflect_board_tuple(board_tuple):
    """Reflects a board tuple horizontally."""
    return tuple(row[::-1] for row in board_tuple)

@functools.lru_cache(maxsize=1024) # Cache canonical forms for small boards
def get_canonical_board_tuple(board_tuple):
    """
    Generates all 8 symmetries of a board and returns the lexicographically smallest one.
    """
    symmetries = []
    current_board = board_tuple
    for _ in range(4):  # 4 rotations
        symmetries.append(current_board)
        symmetries.append(_reflect_board_tuple(current_board))
        current_board = _rotate_board_tuple(current_board)
    return min(symmetries)


# --- Game Logic ---
def check_win_for_eval(board, player, k_to_win): # ... (same)
    n = len(board)
    # Check rows
    for r in range(n):
        for c in range(n - k_to_win + 1):
            if all(board[r][c+i] == player for i in range(k_to_win)): return True
    # Check columns
    for c in range(n):
        for r in range(n - k_to_win + 1):
            if all(board[r+i][c] == player for i in range(k_to_win)): return True
    # Check diagonal (top-left to bottom-right)
    for r in range(n - k_to_win + 1):
        for c in range(n - k_to_win + 1):
            if all(board[r+i][c+i] == player for i in range(k_to_win)): return True
    # Check anti-diagonal (top-right to bottom-left)
    for r in range(n - k_to_win + 1):
        for c in range(k_to_win - 1, n):
            if all(board[r+i][c-i] == player for i in range(k_to_win)): return True
    return False

def get_winning_line(board, player, k_to_win): # ... (same)
    n = len(board)
    # Rows
    for r_idx in range(n):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx][c_idx+i] == player for i in range(k_to_win)):
                return [(r_idx, c_idx+i) for i in range(k_to_win)]
    # Columns
    for c_idx in range(n):
        for r_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx] == player for i in range(k_to_win)):
                return [(r_idx+i, c_idx) for i in range(k_to_win)]
    # Diagonal (TL to BR)
    for r_idx in range(n - k_to_win + 1):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx+i] == player for i in range(k_to_win)):
                return [(r_idx+i, c_idx+i) for i in range(k_to_win)]
    # Anti-diagonal (TR to BL)
    for r_idx in range(n - k_to_win + 1):
        for c_idx in range(k_to_win - 1, n):
            if all(board[r_idx+i][c_idx-i] == player for i in range(k_to_win)):
                return [(r_idx+i, c_idx-i) for i in range(k_to_win)]
    return []

def is_board_full(board): # ... (same)
    return all(cell != EMPTY_CELL for row in board for cell in row)

def get_available_moves(board): # OPTIMIZED
    """
    Gets available moves, sorted by Manhattan distance from the center.
    This helps alpha-beta pruning by exploring potentially stronger moves first.
    """
    n = len(board)
    center_r, center_c = (n - 1) / 2.0, (n - 1) / 2.0

    empty_cells = []
    for r in range(n):
        for c in range(n):
            if board[r][c] == EMPTY_CELL:
                empty_cells.append((r, c))

    # Sort by Manhattan distance to center, then by (r, c) as a consistent tie-breaker
    empty_cells.sort(key=lambda move: (abs(move[0] - center_r) + abs(move[1] - center_c), move[0], move[1]))
    return empty_cells

# --- Minimax AI (Modified for Symmetry) ---
@functools.lru_cache(maxsize=200000)
def _minimax_recursive_logic(canonical_board_tuple_arg, depth, is_maximizing_for_x, k_to_win, alpha, beta):
    """
    Recursive part of Minimax. Expects canonical_board_tuple_arg to be canonical.
    It will canonicalize board states for children before recursive calls.
    """
    nodes_here = 1
    max_depth_here = depth # Tracks the maximum depth reached from this node in this call

    # Board is already canonical, convert to mutable list of lists for local operations
    board = [list(row) for row in canonical_board_tuple_arg]

    if check_win_for_eval(board, PLAYER_HUMAN, k_to_win):
        return (1000 - depth, nodes_here, max_depth_here)
    if check_win_for_eval(board, PLAYER_AI, k_to_win):
        return (-1000 + depth, nodes_here, max_depth_here)
    if is_board_full(board):
        return (0, nodes_here, max_depth_here)

    available_moves_list = get_available_moves(board) # Moves are for the canonical board

    if is_maximizing_for_x:
        max_eval = -math.inf
        for r_move, c_move in available_moves_list:
            board[r_move][c_move] = PLAYER_HUMAN
            board_tuple_next = tuple(map(tuple, board))
            # Canonicalize the *next* board state before recursive call
            canonical_board_tuple_next = get_canonical_board_tuple(board_tuple_next)

            eval_score, child_nodes, child_max_depth = _minimax_recursive_logic(
                canonical_board_tuple_next, depth + 1, False, k_to_win, alpha, beta
            )
            nodes_here += child_nodes
            max_depth_here = max(max_depth_here, child_max_depth)
            board[r_move][c_move] = EMPTY_CELL # Backtrack

            if eval_score > max_eval:
                max_eval = eval_score
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return (max_eval, nodes_here, max_depth_here)
    else: # Minimizing for O
        min_eval = math.inf
        for r_move, c_move in available_moves_list:
            board[r_move][c_move] = PLAYER_AI
            board_tuple_next = tuple(map(tuple, board))
            canonical_board_tuple_next = get_canonical_board_tuple(board_tuple_next)

            eval_score, child_nodes, child_max_depth = _minimax_recursive_logic(
                canonical_board_tuple_next, depth + 1, True, k_to_win, alpha, beta
            )
            nodes_here += child_nodes
            max_depth_here = max(max_depth_here, child_max_depth)
            board[r_move][c_move] = EMPTY_CELL # Backtrack

            if eval_score < min_eval:
                min_eval = eval_score
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return (min_eval, nodes_here, max_depth_here)

def minimax(board_tuple_orig, depth, is_maximizing_for_x, k_to_win, alpha, beta):
    """
    Public interface for Minimax. Handles initial canonicalization.
    """
    canonical_initial_board = get_canonical_board_tuple(board_tuple_orig)
    return _minimax_recursive_logic(canonical_initial_board, depth, is_maximizing_for_x, k_to_win, alpha, beta)

# --- Coordinate Conversion ---
def to_algebraic(row, col, board_size):
    """Converts (row, col) 0-indexed from top-left to algebraic notation (e.g., a1, b3).
       'a1' is bottom-left.
    """
    if not (0 <= row < board_size and 0 <= col < board_size):
        return "Invalid"
    # Column: 0 -> 'a', 1 -> 'b', ...
    # Row: board_size-1 -> '1' (bottom row), 0 -> board_size (top row)
    file_char = chr(ord('a') + col)
    rank_char = str(board_size - row)
    return f"{file_char}{rank_char}"

def from_algebraic(alg_notation, board_size):
    """Converts algebraic notation (e.g., a1, b3) to (row, col) 0-indexed from top-left.
       'a1' is bottom-left.
    """
    alg_notation = alg_notation.lower().strip()
    if not (2 <= len(alg_notation) <= 3): # e.g. a1, aa1 (for larger boards)
        return None # Invalid format

    file_part = ""
    rank_part = ""
    for char in alg_notation:
        if 'a' <= char <= 'z':
            file_part += char
        elif '0' <= char <= '9':
            rank_part += char
        else:
            return None # Invalid char

    if not file_part or not rank_part:
        return None # Missing file or rank

    try:
        col = 0
        for i, char_code in enumerate(reversed([ord(c) - ord('a') for c in file_part])):
            col += (char_code + (i * 26)) # Handles 'aa', 'ab' etc. for cols > 25
            if i > 0: col += 1 # Adjust for multi-char file notation

        rank_num = int(rank_part)
        row = board_size - rank_num

        if not (0 <= row < board_size and 0 <= col < board_size):
            return None # Out of bounds
        return (row, col)
    except ValueError:
        return None # Invalid number for rank

def evaluate_single_move_for_player_process(board_state_tuple_arg, move_arg, k_to_win_arg, current_player_token, is_human_player_turn_for_hint): # ... (same)
    r, c = move_arg
    temp_board = [list(row) for row in board_state_tuple_arg]
    temp_board[r][c] = current_player_token
    next_turn_is_human_x = (current_player_token == PLAYER_AI)

    # The minimax call here will use the new version which handles canonicalization
    score, nodes_in_branch, depth_in_branch = minimax(
        tuple(map(tuple, temp_board)), 0, next_turn_is_human_x, k_to_win_arg, -math.inf, math.inf
    )
    return {'move': move_arg, 'score': score, 'nodes': nodes_in_branch, 'depth': depth_in_branch}

def find_best_move_generic_manager(board_config, k_to_win_config, progress_q, result_q, player_token_for_move): # ... (same)
    global NUM_AI_PROCESSES
    aggregated_nodes_explored = 0; aggregated_max_depth = 0
    is_maximizing_search = (player_token_for_move == PLAYER_HUMAN) # True if human (X) is making the move (we want to maximize X's score)
                                                                  # False if AI (O) is making the move (we want to minimize X's score / maximize O's score)
    current_best_score_so_far = -math.inf if is_maximizing_search else math.inf
    best_move_so_far = None
    board_tuple_config = tuple(map(tuple, board_config))
    
    # Use the optimized get_available_moves here too for consistency, though sorting for the first level
    # has less impact than in deeper minimax calls. It's good for ensuring move evaluation order is somewhat predictable.
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

                if is_maximizing_search: # Human's turn (X), seeking highest score
                    if current_move_score > current_best_score_so_far:
                        current_best_score_so_far = current_move_score
                        best_move_so_far = move_eval_data['move']
                else: # AI's turn (O), seeking lowest score (for X)
                    if current_move_score < current_best_score_so_far:
                        current_best_score_so_far = current_move_score
                        best_move_so_far = move_eval_data['move']
                
                # Fallback if no best move selected yet (e.g., all scores are -inf or +inf initially)
                if best_move_so_far is None:
                    best_move_so_far = move_eval_data['move']
                    current_best_score_so_far = current_move_score

            except Exception as exc:
                err_move = future_to_move_eval[future]
                print(f'Move evaluation for {err_move} generated an exception in process: {exc}') # Consider logging or other error handling
            
            num_completed += 1
            time_elapsed_so_far = time.monotonic() - start_time_total_search
            avg_time_per_move = time_elapsed_so_far / num_completed if num_completed > 0 else 0
            moves_remaining = len(available_moves_config) - num_completed
            estimated_time_left = avg_time_per_move * moves_remaining if avg_time_per_move > 0 else float('inf')
            current_nps = aggregated_nodes_explored / time_elapsed_so_far if time_elapsed_so_far > 0.0001 else 0
            progress_q.put({
                'current': num_completed, 'time_elapsed': time_elapsed_so_far,
                'estimated_left': estimated_time_left, 'type': 'progress',
                'current_nodes': aggregated_nodes_explored, 'current_max_depth': aggregated_max_depth,
                'current_best_score': current_best_score_so_far, 'current_nps': current_nps
            })

    if best_move_so_far is None and available_moves_config: # Should ideally not happen if list is not empty
        best_move_so_far = available_moves_config[0]
        # Try to find score for this fallback move
        fallback_eval = next((em for em in evaluated_moves_details if em['move'] == best_move_so_far), None)
        if fallback_eval:
            current_best_score_so_far = fallback_eval['score']
        else: # Should not happen if evaluated_moves_details is populated
             current_best_score_so_far = -math.inf if is_maximizing_search else math.inf


    # Sort for display, reverse based on whose perspective (max for X, min for O means max score for O player)
    # For hint (human X): higher score is better.
    # For AI (O): lower score (from X's perspective) is better for O. So sort ascending for O.
    evaluated_moves_details.sort(key=lambda x: x['score'], reverse=is_maximizing_search)

    result_q.put({
        'best_move_data': (best_move_so_far, current_best_score_so_far),
        'top_moves_list': evaluated_moves_details[:5], # Top 5 moves
        'total_nodes': aggregated_nodes_explored,
        'max_search_depth': aggregated_max_depth
    })


# --- Tkinter GUI (Largely Unchanged) ---
class TicTacToeGUI:
    def __init__(self, root_window):
        self.root = root_window
        print(f"Initializing GUI. Using {NUM_AI_PROCESSES} AI worker processes for calculations.")
        
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

    def setup_styles(self): # ... (same)
        self.style = ttk.Style()
        available_themes = self.style.theme_names(); current_os = self.root.tk.call("tk", "windowingsystem")
        
        if 'clam' in available_themes: self.style.theme_use('clam')
        elif current_os == "win32" and 'vista' in available_themes: self.style.theme_use('vista')
        elif current_os == "aqua" and 'aqua' in available_themes: self.style.theme_use('aqua')
        else: self.style.theme_use(available_themes[0] if available_themes else 'default')
        
        # print(f"Using ttk theme: {self.style.theme_use()}") # Optional: for debugging theme

        self.style.configure(".", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"])
        self.style.configure("TFrame", background=COLOR_ROOT_BG)
        self.style.configure("Content.TFrame", background=COLOR_FRAME_BG)
        
        self.style.configure("TLabel", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"], padding=2)
        self.style.configure("Header.TLabel", font=self.fonts["header"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_FRAME_BG)
        self.style.configure("Status.TLabel", font=self.fonts["status"], foreground=COLOR_TEXT_PRIMARY) # Ensure this background is COLOR_FRAME_BG if on white panel
        self.style.configure("Info.TLabel", font=self.fonts["info"], foreground=COLOR_TEXT_SECONDARY)
        self.style.configure("InfoValue.TLabel", font=self.fonts["info_value"], foreground=COLOR_TEXT_PRIMARY)
        self.style.configure("HintSuggest.TLabel", font=self.fonts["info_value"], foreground=COLOR_HINT_TEXT, background=COLOR_HINT_SUGGEST_BG, padding=4)
        self.style.configure("MoveInfo.TLabel", font=self.fonts["move_info"], background=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, padding=5, borderwidth=1, relief="solid", bordercolor="#CCCCCC")
        
        self.style.configure("TButton", 
                             font=self.fonts["button"], 
                             padding=(6, 4), 
                             borderwidth=1) 
        self.style.map("TButton",
            relief=[('pressed', 'sunken'), ('!pressed', 'raised')],
            foreground=[('disabled', COLOR_TEXT_SECONDARY)],
            background=[('disabled', "#E0E0E0")]
        )

        self.style.configure("Accent.TButton", 
                             font=self.fonts["button"],
                             padding=(6,4),
                             borderwidth=1,
                             relief="raised",
                             background=COLOR_ACCENT_PRIMARY,
                             foreground=COLOR_TEXT_ON_ACCENT)
        self.style.map("Accent.TButton",
            background=[('pressed', COLOR_ACCENT_SECONDARY), 
                        ('active', COLOR_ACCENT_SECONDARY),
                        ('!disabled', COLOR_ACCENT_PRIMARY)],
            foreground=[('pressed', COLOR_TEXT_ON_ACCENT), 
                        ('active', COLOR_TEXT_ON_ACCENT), 
                        ('!disabled', COLOR_TEXT_ON_ACCENT),
                        ('disabled', COLOR_TEXT_SECONDARY)]
        )

        self.style.configure("Hint.TButton", 
                             font=self.fonts["button"],
                             padding=(6,4),
                             borderwidth=1,
                             relief="raised",
                             background=COLOR_HINT_BG, 
                             foreground=COLOR_HINT_TEXT)
        self.style.map("Hint.TButton", 
            background=[('active', "#FFD700"), 
                        ('pressed', "#EAA600"), 
                        ('!disabled', COLOR_HINT_BG)], 
            foreground=[('active', COLOR_HINT_TEXT), 
                        ('pressed', COLOR_HINT_TEXT), 
                        ('!disabled', COLOR_HINT_TEXT),
                        ('disabled', COLOR_TEXT_SECONDARY)]
        )

        self.style.configure("TLabelframe", background=COLOR_ROOT_BG, bordercolor=COLOR_ACCENT_SECONDARY, padding=6)
        self.style.configure("TLabelframe.Label", font=self.fonts["label"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_ROOT_BG)
        self.style.configure("TEntry", font=self.fonts["entry"], padding=3, fieldbackground=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, bordercolor="#CCCCCC")
        self.style.configure("Horizontal.TProgressbar", thickness=10, background=COLOR_ACCENT_PRIMARY, troughcolor=COLOR_FRAME_BG, bordercolor=COLOR_ACCENT_SECONDARY)

    def toggle_fullscreen(self, event=None): # ... (same)
        self.is_fullscreen = not self.is_fullscreen; self.root.attributes("-fullscreen", self.is_fullscreen)
        if self.buttons: self.root.after(50, self.create_board_ui_buttons)

    def setup_ui_layout(self): # ... (same)
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL, style="TPanedwindow")
        self.paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        left_pane = ttk.Frame(self.paned_window, padding=(10,5), style="Content.TFrame")
        left_pane.grid_columnconfigure(0, weight=1); left_pane.grid_rowconfigure(0, weight=0); left_pane.grid_rowconfigure(1, weight=0); left_pane.grid_rowconfigure(2, weight=1)
        self.paned_window.add(left_pane, weight=30) # Adjust weight as needed, e.g. 30 for 30%
        
        self.board_outer_frame = ttk.Frame(self.paned_window, padding=10, style="TFrame") # Use TFrame for consistent root_bg
        self.board_outer_frame.grid_rowconfigure(0, weight=1); self.board_outer_frame.grid_columnconfigure(0, weight=1)
        self.paned_window.add(self.board_outer_frame, weight=70) # Adjust weight, e.g. 70 for 70%

        # --- Left Pane Content ---
        header_label = ttk.Label(left_pane, text=f"Tic-Tac-Toe ({NUM_AI_PROCESSES} Proc.)", style="Header.TLabel"); header_label.configure(background=COLOR_FRAME_BG) # Match Content.TFrame
        header_label.grid(row=0, column=0, pady=(0,15), sticky="w")

        control_frame = ttk.Labelframe(left_pane, text="Game Settings", padding=10, style="Content.TLabelframe") # Style this to use Content.TFrame's bg
        control_frame.grid(row=1, column=0, sticky="new", pady=(0,10)); control_frame.grid_columnconfigure(1, weight=1)
        # Configure children of control_frame to use COLOR_FRAME_BG
        lbl_board_size = ttk.Label(control_frame, text="Board Size (N x N):", background=COLOR_FRAME_BG);
        lbl_board_size.grid(row=0, column=0, padx=5, pady=3, sticky="w")
        self.size_entry = ttk.Entry(control_frame, width=5); self.size_entry.insert(0, str(self.board_size))
        self.size_entry.grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        lbl_k_row = ttk.Label(control_frame, text="K-in-a-row:", background=COLOR_FRAME_BG);
        lbl_k_row.grid(row=1, column=0, padx=5, pady=3, sticky="w")
        self.k_entry = ttk.Entry(control_frame, width=5); self.k_entry.insert(0, str(self.k_to_win))
        self.k_entry.grid(row=1, column=1, padx=5, pady=3, sticky="ew")

        button_frame = ttk.Frame(control_frame, style="Content.TFrame") # Match Content.TFrame
        button_frame.grid(row=2, column=0, columnspan=2, pady=(8,0), sticky="ew")
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        self.start_human_button = ttk.Button(button_frame, text="▶ You Start", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=True))
        self.start_human_button.grid(row=0, column=0, padx=(0,3), pady=2, sticky="ew")
        self.start_ai_button = ttk.Button(button_frame, text="🤖 AI Starts", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=False))
        self.start_ai_button.grid(row=0, column=1, padx=(3,0), pady=2, sticky="ew")
        self.hint_button = ttk.Button(button_frame, text="💡 Suggest Move", style="Hint.TButton", command=self.get_human_hint, state=tk.DISABLED)
        self.hint_button.grid(row=1, column=0, columnspan=2, pady=(4,0), sticky="ew")

        status_vis_frame = ttk.Labelframe(left_pane, text="Game & AI Insights", padding=10, style="Content.TLabelframe")
        status_vis_frame.grid(row=2, column=0, sticky="nsew", pady=(0,0)); status_vis_frame.grid_columnconfigure(0, weight=1); status_vis_frame.grid_columnconfigure(1, weight=0) # col 1 no weight
        
        self.status_label = ttk.Label(status_vis_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=350, background=COLOR_FRAME_BG)
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(0,6), sticky="new")
        
        progress_frame = ttk.Frame(status_vis_frame, style="Content.TFrame") # Match Content.TFrame
        progress_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,6))
        progress_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_percent_label = ttk.Label(progress_frame, textvariable=self.progress_percent_var, style="Info.TLabel", background=COLOR_FRAME_BG)

        stats_grid = ttk.Frame(status_vis_frame, style="Content.TFrame") # Match Content.TFrame
        stats_grid.grid(row=2, column=0, columnspan=2, sticky="new", pady=(0,6)); stats_grid.columnconfigure(1, weight=1)
        
        row_idx = 0
        for label_text, var, unit in [
            ("Move Score:", self.ai_eval_var, ""), ("Time Taken:", self.time_taken_var, "s"),
            ("Nodes Explored:", self.nodes_explored_var, ""), ("Nodes/Sec (NPS):", self.nps_var, ""),
            ("Max Depth Searched:", self.actual_depth_var, ""), ("Est. Time Left:", self.estimated_time_left_var, "s")
        ]:
            lbl = ttk.Label(stats_grid, text=label_text, style="Info.TLabel", background=COLOR_FRAME_BG); lbl.grid(row=row_idx, column=0, sticky="w", padx=2, pady=1)
            val_lbl = ttk.Label(stats_grid, textvariable=var, style="InfoValue.TLabel", background=COLOR_FRAME_BG); val_lbl.grid(row=row_idx, column=1, sticky="w", padx=2, pady=1)
            row_idx += 1
        
        score_info_lbl = ttk.Label(stats_grid, text="(Score: >0 X wins, <0 O wins, 0 Draw)", style="Info.TLabel", foreground="#6c757d", background=COLOR_FRAME_BG)
        score_info_lbl.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=2, pady=1)
        
        self.hint_suggestion_label = ttk.Label(status_vis_frame, textvariable=self.hint_suggestion_var, style="HintSuggest.TLabel", wraplength=350, anchor="center")
        self.hint_suggestion_label.grid(row=row_idx + 1, column=0, columnspan=2, sticky="new", pady=4)
        
        self.top_moves_label = ttk.Label(status_vis_frame, textvariable=self.top_moves_var, style="MoveInfo.TLabel", anchor="nw") # background from style
        self.top_moves_label.grid(row=row_idx + 2, column=0, columnspan=2, pady=(8,0), sticky="nsew"); status_vis_frame.grid_rowconfigure(row_idx + 2, weight=1)
        
        # --- Board Frame (Right Pane) ---
        self.board_frame = ttk.Frame(self.board_outer_frame, style="TFrame") # Centered in board_outer_frame
        self.board_frame.grid(row=0, column=0, sticky="") # Not sticky, to allow centering by parent's grid config

        self.status_var.set("Adjust N (2-7) & K, then Start."); self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.style.configure("Content.TLabelframe", background=COLOR_FRAME_BG)
        self.style.configure("Content.TLabelframe.Label", font=self.fonts["label"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_FRAME_BG)


    def on_closing(self, event=None):
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            if messagebox.askokcancel("Calculation Running", "Calculation in progress. Close anyway?"):
                # Potentially add logic to signal the thread/processes to terminate gracefully if possible
                self.root.destroy()
            else:
                return # Don't close
        else:
            self.root.destroy()

    def start_new_game(self, human_starts=True): # CORRECTED
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showwarning("Busy", "AI calculation is currently in progress. Please wait.")
            return
        try:
            n = int(self.size_entry.get()); k = int(self.k_entry.get())
            if not (2 <= n <= 7): # Max N=7 as larger can be extremely slow
                messagebox.showerror("Error", "Board size N must be between 2 and 7.")
                return
            if n > 4 : # Warn for N > 4
                 warning_key = f"_warned_slow_n{n}_mp{NUM_AI_PROCESSES}" # More specific key
                 if not hasattr(self, warning_key) or not getattr(self, warning_key):
                    if not messagebox.askokcancel("Performance Warning", f"Board size N={n} (with {NUM_AI_PROCESSES} process(es)) can be very slow, especially for the first few AI moves.\n\nConsider N=3 or N=4 for faster play.\n\nDo you want to continue?"):
                        return
                    setattr(self, warning_key, True) # Remember warning for this config

            if not (2 <= k <= n):
                messagebox.showerror("Error", f"K-in-a-row must be between 2 and N (board size).")
                return
            self.board_size = n; self.k_to_win = k
        except ValueError:
            messagebox.showerror("Error", "Invalid input for N or K. Please enter numbers.")
            return

        # Clear previous game caches
        _minimax_recursive_logic.cache_clear() # CORRECT: This is the function with lru_cache
        get_canonical_board_tuple.cache_clear() # CORRECT: This also has lru_cache

        self.game_board = [[EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = PLAYER_HUMAN if human_starts else PLAYER_AI
        self.game_over = False
        self.hint_suggestion_var.set("") # Clear previous hint

        initial_status = f"{self.board_size}x{self.board_size} game, {self.k_to_win}-in-a-row. "
        initial_status += "Your turn (X)." if human_starts else "AI's turn (O)."
        self.status_var.set(initial_status)

        # Reset AI stats
        self.ai_eval_var.set("N/A"); self.nodes_explored_var.set("0"); self.actual_depth_var.set("0")
        self.time_taken_var.set("N/A"); self.nps_var.set("N/A")
        self.estimated_time_left_var.set("N/A"); self.progress_percent_var.set("0.0%")
        self.top_moves_var.set("Top Move Considerations:\n(After calculation)")
        if self.progress_bar.master.winfo_ismapped(): # Hide progress bar elements
            self.progress_bar.master.grid_remove()


        self.create_board_ui_buttons() # Rebuilds buttons and board UI
        self.update_hint_button_state()

        if not human_starts:
            self.root.after(100, lambda: self.trigger_ai_or_hint_calculation(PLAYER_AI)) # AI makes the first move

    def update_hint_button_state(self): # ... (same)
        # Hint button is enabled if: game not over, human's turn, and no calculation running
        is_calc_running = self.calculation_manager_thread and self.calculation_manager_thread.is_alive()
        if not self.game_over and self.current_player == PLAYER_HUMAN and not is_calc_running:
            self.hint_button.config(state=tk.NORMAL)
        else:
            self.hint_button.config(state=tk.DISABLED)
            
    def get_human_hint(self): # ... (same)
        if self.game_over or self.current_player != PLAYER_HUMAN:
            return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showinfo("Busy", "AI calculation is currently in progress. Please wait.")
            return
        self.hint_suggestion_var.set("Calculating your best move...")
        self.trigger_ai_or_hint_calculation(PLAYER_HUMAN) # Calculate for Human (X)

    def trigger_ai_or_hint_calculation(self, player_to_calculate_for): # ... (same)
        is_hint = (player_to_calculate_for == PLAYER_HUMAN)
        if is_hint:
            self.status_var.set("Calculating hint for Human (X)...")
        else:
            self.status_var.set("AI (O) is thinking (using processes)...")
        
        self.top_moves_var.set("Top Move Considerations:\nCalculating...")
        self.disable_board_buttons() # Disable board during calculation
        self.hint_button.config(state=tk.DISABLED) # Disable hint button during any calculation

        # Show progress bar
        self.progress_bar.master.grid() # Make the frame visible
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self.progress_percent_label.grid(row=0, column=1, sticky="w", padx=(0,0))
        self.progress_bar["value"] = 0
        self.progress_percent_var.set("0.0%")
        self.estimated_time_left_var.set("Estimating...")
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        # Clear queues
        while not self.progress_queue.empty(): self.progress_queue.get_nowait()
        while not self.result_queue.empty(): self.result_queue.get_nowait()

        # Start the calculation in a separate thread to keep UI responsive
        self.calculation_manager_thread = threading.Thread(
            target=find_best_move_generic_manager,
            args=(self.game_board, self.k_to_win, self.progress_queue, self.result_queue, player_to_calculate_for),
            daemon=True # Daemon thread exits when main program exits
        )
        self.ai_start_time = time.monotonic()
        self._current_calculation_for_player = player_to_calculate_for # Store who we are calculating for
        self.calculation_manager_thread.start()
        self.check_ai_progress() # Start polling for progress

    def create_board_ui_buttons(self): # ... (same)
        for widget in self.board_frame.winfo_children(): widget.destroy()
        self.buttons = []
        self.root.update_idletasks() # Ensure dimensions are updated

        available_width = self.board_outer_frame.winfo_width()
        available_height = self.board_outer_frame.winfo_height()
        
        # Fallback if dimensions are too small (e.g. during init)
        if available_width < 50 or available_height < 50:
             available_width = max(self.root.winfo_height() * 0.55, 250) # Approx 55% of root height or 250px
             available_height = max(self.root.winfo_height() * 0.55, 250)

        cell_dim = min(available_width // self.board_size, available_height // self.board_size)
        cell_dim = max(35, cell_dim - 4) # Min size 35px, with 2px padding (total 4)
        
        btn_font_size = max(10, int(cell_dim * 0.35)) # Adjust font size based on cell size
        board_button_font = tkfont.Font(family=self.default_font_family, size=btn_font_size, weight="bold")

        for r in range(self.board_size):
            row_buttons = []
            self.board_frame.grid_rowconfigure(r, weight=1, minsize=cell_dim)
            for c in range(self.board_size):
                self.board_frame.grid_columnconfigure(c, weight=1, minsize=cell_dim)
                button = tk.Button(
                    self.board_frame,
                    text=EMPTY_CELL, # Will be updated by update_board_button_states
                    font=board_button_font,
                    relief="flat",
                    borderwidth=1,
                    bg=COLOR_FRAME_BG, # Default background
                    activebackground="#DDDDDD", # Lighter grey for active
                    fg=COLOR_TEXT_PRIMARY,
                    command=lambda r_idx=r, c_idx=c: self.handle_cell_click(r_idx, c_idx)
                )
                button.grid(row=r, column=c, sticky="nsew", padx=1, pady=1) # Small gap between buttons
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board_button_states() # Apply initial game state to buttons

    def handle_cell_click(self, r, c): # ... (same)
        if self.game_over or self.game_board[r][c] != EMPTY_CELL or self.current_player != PLAYER_HUMAN:
            return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showinfo("Wait", "AI calculation is currently in progress. Please wait.")
            return

        self.hint_suggestion_var.set("") # Clear any hint suggestion
        self.make_move(r, c, PLAYER_HUMAN)

        if self.check_game_status(): # Check if human's move ended the game
            return

        self.current_player = PLAYER_AI
        self.update_hint_button_state() # Disable hint button as it's AI's turn
        self.trigger_ai_or_hint_calculation(PLAYER_AI) # AI makes its move

    def check_ai_progress(self): # ... (same)
        update_interval_ms = 30 # How often to check the queue (e.g., ~30 FPS updates)
        try:
            progress_update = self.progress_queue.get_nowait() # Non-blocking get
            if progress_update.get('type') == 'start':
                self.progress_bar["maximum"] = progress_update['total']
                self.progress_percent_var.set("0.0%")
                self.estimated_time_left_var.set("Estimating...")
                # Reset stats that accumulate per calculation
                self.nodes_explored_var.set("0")
                self.actual_depth_var.set("0")
                self.ai_eval_var.set("N/A") # Or "Calculating..."
                self.nps_var.set("0")

            elif progress_update.get('type') == 'progress':
                current_val = progress_update['current']
                self.progress_bar["value"] = current_val
                max_val = self.progress_bar["maximum"]
                
                if max_val > 0: percent = (current_val / max_val) * 100
                else: percent = 100.0 if current_val > 0 else 0.0 # Handle max_val=0 case
                self.progress_percent_var.set(f"{percent:.1f}%")
                
                est_left = progress_update.get('estimated_left', float('inf'))
                if est_left != float('inf') and est_left >= 0:
                    self.estimated_time_left_var.set(f"{est_left:.1f}")
                elif current_val == max_val : # If done
                    self.estimated_time_left_var.set("0.0")
                else: # Still calculating initial estimates
                    self.estimated_time_left_var.set("Calculating...")

                if 'current_nodes' in progress_update: self.nodes_explored_var.set(f"{progress_update['current_nodes']:,}")
                if 'current_max_depth' in progress_update: self.actual_depth_var.set(f"{progress_update['current_max_depth']}")
                if 'current_best_score' in progress_update:
                    score_val = progress_update['current_best_score']
                    if score_val not in (math.inf, -math.inf): self.ai_eval_var.set(f"{score_val:.0f}")
                    else: self.ai_eval_var.set("...") # Placeholder for inf scores
                if 'current_nps' in progress_update: self.nps_var.set(f"{progress_update['current_nps']:,.0f}")
                if 'time_elapsed' in progress_update: self.time_taken_var.set(f"{progress_update['time_elapsed']:.3f}")

        except thread_queue.Empty:
            pass # No update yet, continue

        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            self.root.after(update_interval_ms, self.check_ai_progress) # Schedule next check
        elif self.calculation_manager_thread: # Thread finished but not None yet
            self.handle_calculation_result()
            self.calculation_manager_thread = None # Mark as handled

    def handle_calculation_result(self): # MODIFIED
        self.root.config(cursor="") # Reset cursor
        if self.progress_bar.master.winfo_ismapped():
            self.progress_bar.master.grid_remove() # Hide progress bar elements

        time_taken_calc = time.monotonic() - self.ai_start_time
        was_hint_calculation = (self._current_calculation_for_player == PLAYER_HUMAN)

        try:
            ai_result_data = self.result_queue.get_nowait() # Should have a result
            best_move_tuple, best_score = ai_result_data['best_move_data']
            top_moves_list = ai_result_data['top_moves_list']
            total_nodes = ai_result_data['total_nodes']
            max_search_depth = ai_result_data['max_search_depth']
        except thread_queue.Empty:
            self.status_var.set("Error: No result from AI calculation.")
            self.enable_board_buttons()
            self.update_hint_button_state()
            return

        # Update final stats
        self.time_taken_var.set(f"{time_taken_calc:.3f}")
        self.nodes_explored_var.set(f"{total_nodes:,}")
        nps_val = total_nodes / time_taken_calc if time_taken_calc > 0.0001 else 0
        self.nps_var.set(f"{nps_val:,.0f}")
        self.actual_depth_var.set(f"{max_search_depth}")
        if best_score not in (math.inf, -math.inf): self.ai_eval_var.set(f"{best_score:.0f}")
        else: self.ai_eval_var.set("Win/Loss")
        self.estimated_time_left_var.set("Done.")

        # Convert best_move_tuple to algebraic for display
        best_move_alg = to_algebraic(best_move_tuple[0], best_move_tuple[1], self.board_size) if best_move_tuple else "N/A"

        perspective = "Human (X)" if was_hint_calculation else "AI (O)"
        top_moves_text = f"Top {perspective} Moves (Move: Score | Nodes | Depth):\n" # Updated title
        if top_moves_list:
            for item in top_moves_list:
                move_tuple = item['move']
                # Convert move_tuple to algebraic for display
                move_alg = to_algebraic(move_tuple[0], move_tuple[1], self.board_size)
                
                score_str = f"{item['score']:.0f}"
                nodes_str = f"N:{item['nodes']:,}"
                depth_str = f"D:{item['depth']}"
                # Use a consistent width for algebraic notation, e.g., 4 chars ("aa10")
                top_moves_text += f"  {move_alg:<4}: {score_str:<5} | {nodes_str:<12}| {depth_str}\n"
        else:
            top_moves_text += "  (N/A)\n"
        self.top_moves_var.set(top_moves_text.strip())

        if was_hint_calculation:
            if best_move_tuple:
                self.status_var.set(f"Hint for X: Best is {best_move_alg} (Score: {best_score:.0f}). Your turn.")
                self.hint_suggestion_var.set(f"Suggested for X: {best_move_alg} (Score: {best_score:.0f})")
                # Highlight the suggested cell if it's empty
                if self.game_board[best_move_tuple[0]][best_move_tuple[1]] == EMPTY_CELL and \
                   self.buttons[best_move_tuple[0]][best_move_tuple[1]]['state'] != tk.DISABLED:
                    self.buttons[best_move_tuple[0]][best_move_tuple[1]].config(bg=COLOR_HINT_SUGGEST_BG, relief="raised")
                    self.root.after(3000, lambda r=best_move_tuple[0], c=best_move_tuple[1]: self.clear_hint_highlight(r,c))
            else:
                 self.status_var.set("Hint: No moves available or game ended. Your turn.")
                 self.hint_suggestion_var.set("No suggestion.")
            self.enable_board_buttons()
            self.update_hint_button_state()
        else: # AI's turn calculation result
            self.hint_suggestion_var.set("")
            if best_move_tuple:
                self.make_move(best_move_tuple[0], best_move_tuple[1], PLAYER_AI) # make_move still uses (r,c)
                if self.check_game_status():
                    self.enable_board_buttons()
                    self.update_hint_button_state()
                    return
                self.current_player = PLAYER_HUMAN
                # Update status with algebraic notation for AI's move
                ai_moved_to_alg = to_algebraic(best_move_tuple[0], best_move_tuple[1], self.board_size)
                self.status_var.set(f"AI (O) moved to {ai_moved_to_alg}. Your turn (X).")
            else:
                 if is_board_full(self.game_board) and \
                    not get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win) and \
                    not get_winning_line(self.game_board, PLAYER_AI, self.k_to_win):
                     self.status_var.set("It's a Draw!")
                 else:
                     self.status_var.set("AI error or no moves. Game Over?")
                 self.game_over = True
            
            self.enable_board_buttons()
            self.update_hint_button_state()


    def clear_hint_highlight(self, r, c): # ... (same)
        # Only clear if the cell is still empty and has the hint background
        if self.game_board[r][c] == EMPTY_CELL and \
           self.buttons[r][c]['bg'] == COLOR_HINT_SUGGEST_BG : # Check actual color
            self.buttons[r][c].config(bg=COLOR_FRAME_BG, relief="flat")

    def make_move(self, r, c, player): # ... (same)
        self.game_board[r][c] = player
        btn = self.buttons[r][c]
        btn.config(text=player, state=tk.DISABLED, relief=tk.SUNKEN) # Mark as taken
        
        player_color_bg = COLOR_HUMAN_MOVE_BG if player == PLAYER_HUMAN else COLOR_AI_MOVE_BG
        player_fg_color = COLOR_ACCENT_SECONDARY if player == PLAYER_HUMAN else COLOR_DANGER
        btn.config(disabledforeground=player_fg_color, background=player_color_bg)


    def check_game_status(self): # ... (same)
        human_wins_line = get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win)
        ai_wins_line = get_winning_line(self.game_board, PLAYER_AI, self.k_to_win)
        
        changed_game_over_state = False # Track if game_over state changes in this call

        if not self.game_over: # Only update if game wasn't already over
            if human_wins_line:
                self.status_var.set("You (X) Win!"); self.game_over = True; changed_game_over_state = True
                self.highlight_winning_line(human_wins_line)
            elif ai_wins_line:
                self.status_var.set("AI (O) Wins!"); self.game_over = True; changed_game_over_state = True
                self.highlight_winning_line(ai_wins_line)
            elif is_board_full(self.game_board):
                self.status_var.set("It's a Draw!"); self.game_over = True; changed_game_over_state = True
        
        if changed_game_over_state or self.game_over: # If state changed to over, or was already over
            self.disable_board_buttons() # Ensure all buttons disabled on game over
            self.update_hint_button_state() # Hint button should be disabled

        return self.game_over

    def update_board_button_states(self): # ... (same)
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                text = self.game_board[r_idx][c_idx]
                button = self.buttons[r_idx][c_idx]
                
                current_bg = button.cget('bg') # Check current background for hint
                is_hint_highlighted = (current_bg == COLOR_HINT_SUGGEST_BG)

                if text == EMPTY_CELL:
                    if not is_hint_highlighted : # Don't overwrite hint highlight
                        button.config(text=text, bg=COLOR_FRAME_BG, relief="flat", state=tk.NORMAL, fg=COLOR_TEXT_PRIMARY)
                    else: # If it is hint highlighted, ensure it's normal state
                        button.config(text=text, state=tk.NORMAL, fg=COLOR_TEXT_PRIMARY) # Keep hint bg
                else: # Cell is taken
                    button.config(text=text, state=tk.DISABLED, relief="sunken")
                    player_color_bg = COLOR_HUMAN_MOVE_BG if text == PLAYER_HUMAN else COLOR_AI_MOVE_BG
                    player_fg_color = COLOR_ACCENT_SECONDARY if text == PLAYER_HUMAN else COLOR_DANGER
                    button.config(disabledforeground=player_fg_color, background=player_color_bg)

    def disable_board_buttons(self): # ... (same)
        for r_buttons in self.buttons:
            for button in r_buttons:
                if button['state'] == tk.NORMAL: # Only disable if it's currently normal
                    button.config(state=tk.DISABLED)

    def enable_board_buttons(self): # ... (same)
        if self.game_over: return # Don't enable if game is over
        
        # Don't enable if AI is thinking
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): return 
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.game_board[r][c] == EMPTY_CELL:
                    # Check if it was a hint highlight, preserve it if so, otherwise reset
                    if self.buttons[r][c]['bg'] != COLOR_HINT_SUGGEST_BG:
                         self.buttons[r][c].config(state=tk.NORMAL, bg=COLOR_FRAME_BG, relief="flat")
                    else: # Keep hint bg, just enable
                         self.buttons[r][c].config(state=tk.NORMAL, relief="raised")


    def highlight_winning_line(self, winning_cells): # ... (same)
        for r,c in winning_cells:
            self.buttons[r][c].config(background=COLOR_WIN_HIGHLIGHT, relief=tk.GROOVE) # GROOVE makes it pop

# --- Main Execution ---
if __name__ == "__main__":
    # This is crucial for ProcessPoolExecutor on Windows and macOS with 'spawn' or 'forkserver' start methods
    # It should be outside the if __name__ == "__main__": block for modules that might be imported,
    # but for a script that's only run directly, it's fine here.
    # However, standard practice is to ensure it's at the top level if your functions are defined globally.
    # For Tkinter + multiprocessing, this setup is generally robust.
    main_root = tk.Tk()
    app = TicTacToeGUI(main_root)
    main_root.mainloop()