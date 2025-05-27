import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import math
import functools
import time
import threading
import queue as thread_queue

# --- Constants ---
PLAYER_HUMAN = 'X'
PLAYER_AI = 'O'
EMPTY_CELL = ' '

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

# --- Transposition Table Constants ---
TT_EXACT = 0
TT_LOWER_BOUND = 1
TT_UPPER_BOUND = 2

# --- Global Transposition Table ---
transposition_table = {}
tt_hits = 0 
tt_stores = 0


# --- Symmetry Helper Functions ---
def _rotate_board_tuple(board_tuple):
    n = len(board_tuple)
    new_board_list = [[EMPTY_CELL for _ in range(n)] for _ in range(n)]
    for r_idx in range(n):
        for c_idx in range(n):
            new_board_list[c_idx][n - 1 - r_idx] = board_tuple[r_idx][c_idx]
    return tuple(map(tuple, new_board_list))

def _reflect_board_tuple(board_tuple):
    return tuple(row[::-1] for row in board_tuple)

@functools.lru_cache(maxsize=2048)
def get_canonical_board_tuple(board_tuple):
    symmetries = []
    current_board = board_tuple
    for _ in range(4):
        symmetries.append(current_board)
        symmetries.append(_reflect_board_tuple(current_board))
        current_board = _rotate_board_tuple(current_board)
    return min(symmetries)

# --- Coordinate Conversion ---
def to_algebraic(row, col, board_size):
    if not (0 <= row < board_size and 0 <= col < board_size): return "Invalid"
    file_char = chr(ord('a') + col)
    rank_char = str(board_size - row)
    return f"{file_char}{rank_char}"

# --- Game Logic ---
def check_win_for_eval(board, player, k_to_win):
    n = len(board)
    for r in range(n):
        for c in range(n - k_to_win + 1):
            if all(board[r][c+i] == player for i in range(k_to_win)): return True
    for c_idx in range(n):
        for r_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx] == player for i in range(k_to_win)): return True
    for r in range(n - k_to_win + 1):
        for c in range(n - k_to_win + 1):
            if all(board[r+i][c+i] == player for i in range(k_to_win)): return True
    for r in range(n - k_to_win + 1):
        for c in range(k_to_win - 1, n):
            if all(board[r+i][c-i] == player for i in range(k_to_win)): return True
    return False

def get_winning_line(board, player, k_to_win):
    n = len(board)
    for r_idx in range(n):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx][c_idx+i] == player for i in range(k_to_win)):
                return [(r_idx, c_idx+i) for i in range(k_to_win)]
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

def is_board_full(board):
    return all(cell != EMPTY_CELL for row in board for cell in row)

def get_available_moves(board): # board is list of lists
    n = len(board)
    center_r, center_c = (n - 1) / 2.0, (n - 1) / 2.0
    empty_cells = []
    for r in range(n):
        for c in range(n):
            if board[r][c] == EMPTY_CELL:
                empty_cells.append((r, c))
    empty_cells.sort(key=lambda move: (abs(move[0] - center_r) + abs(move[1] - center_c), move[0], move[1]))
    return empty_cells

# --- Early Draw Detection Helpers ---
@functools.lru_cache(maxsize=8192)
def is_line_still_possible(line_segment_tuple, player, k_to_win):
    opponent = PLAYER_AI if player == PLAYER_HUMAN else PLAYER_HUMAN
    if opponent in line_segment_tuple: return False
    player_pieces = line_segment_tuple.count(player)
    empty_pieces = line_segment_tuple.count(EMPTY_CELL)
    return (player_pieces + empty_pieces) >= k_to_win

@functools.lru_cache(maxsize=4096)
def can_player_still_win(board_tuple, player, k_to_win):
    board = [list(row) for row in board_tuple]
    n = len(board)
    for r in range(n):
        for c in range(n - k_to_win + 1):
            segment = [board[r][c+i] for i in range(k_to_win)]
            if is_line_still_possible(tuple(segment), player, k_to_win): return True
    for c_col in range(n):
        for r_row in range(n - k_to_win + 1):
            segment = [board[r_row+i][c_col] for i in range(k_to_win)]
            if is_line_still_possible(tuple(segment), player, k_to_win): return True
    for r in range(n - k_to_win + 1):
        for c in range(n - k_to_win + 1):
            segment = [board[r+i][c+i] for i in range(k_to_win)]
            if is_line_still_possible(tuple(segment), player, k_to_win): return True
    for r in range(n - k_to_win + 1):
        for c in range(k_to_win - 1, n):
            segment = [board[r+i][c-i] for i in range(k_to_win)]
            if is_line_still_possible(tuple(segment), player, k_to_win): return True
    return False

@functools.lru_cache(maxsize=2048)
def is_unwinnable_for_either(board_tuple, k_to_win):
    if not can_player_still_win(board_tuple, PLAYER_HUMAN, k_to_win) and \
       not can_player_still_win(board_tuple, PLAYER_AI, k_to_win):
        return True
    return False

# --- Forced Win Check Helpers ---
@functools.lru_cache(maxsize=16384)
def _get_immediate_winning_moves(board_tuple, player, k_to_win):
    board = [list(row) for row in board_tuple]
    winning_moves = []
    temp_available_moves = []
    for r_try_idx in range(len(board)):
        for c_try_idx in range(len(board[0])):
            if board[r_try_idx][c_try_idx] == EMPTY_CELL:
                temp_available_moves.append((r_try_idx, c_try_idx))
    for r_try, c_try in temp_available_moves:
        board[r_try][c_try] = player
        if check_win_for_eval(board, player, k_to_win):
            winning_moves.append((r_try, c_try))
        board[r_try][c_try] = EMPTY_CELL
    return winning_moves

@functools.lru_cache(maxsize=32768)
def can_force_win_on_next_player_turn(board_tuple, player_to_move, opponent, k_to_win):
    original_player_moves = get_available_moves([list(row) for row in board_tuple])
    for r_m1, c_m1 in original_player_moves:
        board_after_m1 = [list(row) for row in board_tuple]
        board_after_m1[r_m1][c_m1] = player_to_move
        board_after_m1_tuple = tuple(map(tuple, board_after_m1))
        if check_win_for_eval(board_after_m1, player_to_move, k_to_win): continue
        opponent_available_moves_after_m1 = get_available_moves(board_after_m1)
        if not opponent_available_moves_after_m1:
            if _get_immediate_winning_moves(board_after_m1_tuple, player_to_move, k_to_win): return True
            else: continue
        m1_is_forcing_so_far = True
        for r_mopp, c_mopp in opponent_available_moves_after_m1:
            board_after_mopp = [list(row) for row in board_after_m1_tuple]
            board_after_mopp[r_mopp][c_mopp] = opponent
            board_after_mopp_tuple = tuple(map(tuple, board_after_mopp))
            if check_win_for_eval(board_after_mopp, opponent, k_to_win): m1_is_forcing_so_far = False; break
            if not _get_immediate_winning_moves(board_after_mopp_tuple, player_to_move, k_to_win): m1_is_forcing_so_far = False; break
        if m1_is_forcing_so_far: return True
    return False

# --- Minimax AI (Corrected Order for Depth Limit & TT Store for Terminals) ---
def _minimax_recursive_logic(canonical_board_tuple_arg, depth, is_maximizing_for_x, k_to_win, alpha, beta,
                             current_max_search_depth_for_this_call):
    global tt_hits, tt_stores
    nodes_here = 1; max_depth_here = depth 
    original_alpha = alpha

    remaining_depth_tt = current_max_search_depth_for_this_call - depth
    if canonical_board_tuple_arg in transposition_table:
        entry = transposition_table[canonical_board_tuple_arg]
        if entry['depth_searched'] >= remaining_depth_tt:
            tt_hits += 1
            if entry['flag'] == TT_EXACT: return (entry['score'], nodes_here, max_depth_here)
            elif entry['flag'] == TT_LOWER_BOUND: alpha = max(alpha, entry['score'])
            elif entry['flag'] == TT_UPPER_BOUND: beta = min(beta, entry['score'])
            if alpha >= beta: return (entry['score'], nodes_here, max_depth_here)

    board = [list(row) for row in canonical_board_tuple_arg]
    
    score_terminal = None
    if check_win_for_eval(board, PLAYER_HUMAN, k_to_win): score_terminal = 1000 - depth
    elif check_win_for_eval(board, PLAYER_AI, k_to_win): score_terminal = -1000 + depth
    elif is_board_full(board): score_terminal = 0
    elif is_unwinnable_for_either(canonical_board_tuple_arg, k_to_win): score_terminal = 0
    
    if score_terminal is not None:
        transposition_table[canonical_board_tuple_arg] = {
            'score': score_terminal, 'depth_searched': 99, 
            'flag': TT_EXACT, 'best_move': None 
        }
        tt_stores +=1
        return (score_terminal, nodes_here, max_depth_here)
    
    if depth >= current_max_search_depth_for_this_call: 
        return (0, nodes_here, max_depth_here)

    if (depth + 2) <= current_max_search_depth_for_this_call:
        score_forced = 0; is_forced = False
        if is_maximizing_for_x:
            if can_force_win_on_next_player_turn(canonical_board_tuple_arg, PLAYER_HUMAN, PLAYER_AI, k_to_win):
                score_forced = 1000 - (depth + 2); is_forced = True
        else:
            if can_force_win_on_next_player_turn(canonical_board_tuple_arg, PLAYER_AI, PLAYER_HUMAN, k_to_win):
                score_forced = -1000 + (depth + 2); is_forced = True
        if is_forced: return (score_forced, nodes_here, max_depth_here)

    available_moves_list = get_available_moves(board)
    if not available_moves_list: return (0, nodes_here, max_depth_here)
    
    tt_best_move = None
    if canonical_board_tuple_arg in transposition_table:
        entry = transposition_table[canonical_board_tuple_arg]
        if entry.get('best_move') is not None:
            bm_r, bm_c = entry['best_move']
            if board[bm_r][bm_c] == EMPTY_CELL and entry['best_move'] in available_moves_list:
                tt_best_move = entry['best_move']
                available_moves_list.remove(tt_best_move)
                available_moves_list.insert(0, tt_best_move)

    best_move_this_node = None
    if is_maximizing_for_x:
        max_eval = -math.inf
        for r_move, c_move in available_moves_list:
            board[r_move][c_move] = PLAYER_HUMAN
            board_tuple_next = tuple(tuple(inner_list) for inner_list in board)
            canonical_board_tuple_next = get_canonical_board_tuple(board_tuple_next)
            eval_score, child_nodes, child_max_depth = _minimax_recursive_logic(
                canonical_board_tuple_next, depth + 1, False, k_to_win, alpha, beta, current_max_search_depth_for_this_call)
            nodes_here += child_nodes; max_depth_here = max(max_depth_here, child_max_depth)
            board[r_move][c_move] = EMPTY_CELL
            if eval_score > max_eval:
                max_eval = eval_score; best_move_this_node = (r_move, c_move)
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        flag = TT_EXACT
        if max_eval <= original_alpha: flag = TT_UPPER_BOUND
        elif max_eval >= beta: flag = TT_LOWER_BOUND
        current_tt_entry = transposition_table.get(canonical_board_tuple_arg)
        if current_tt_entry is None or remaining_depth_tt >= current_tt_entry['depth_searched']:
            transposition_table[canonical_board_tuple_arg] = {
                'score': max_eval, 'depth_searched': remaining_depth_tt, 
                'flag': flag, 'best_move': best_move_this_node}
            tt_stores += 1
        return (max_eval, nodes_here, max_depth_here)
    else: # Minimizing for O
        min_eval = math.inf
        for r_move, c_move in available_moves_list:
            board[r_move][c_move] = PLAYER_AI
            board_tuple_next = tuple(tuple(inner_list) for inner_list in board)
            canonical_board_tuple_next = get_canonical_board_tuple(board_tuple_next)
            eval_score, child_nodes, child_max_depth = _minimax_recursive_logic(
                canonical_board_tuple_next, depth + 1, True, k_to_win, alpha, beta, current_max_search_depth_for_this_call)
            nodes_here += child_nodes; max_depth_here = max(max_depth_here, child_max_depth)
            board[r_move][c_move] = EMPTY_CELL
            if eval_score < min_eval:
                min_eval = eval_score; best_move_this_node = (r_move, c_move)
            beta = min(beta, eval_score)
            if beta <= alpha: break
        flag = TT_EXACT
        if min_eval <= original_alpha: flag = TT_UPPER_BOUND
        elif min_eval >= beta: flag = TT_LOWER_BOUND
        current_tt_entry = transposition_table.get(canonical_board_tuple_arg)
        if current_tt_entry is None or remaining_depth_tt >= current_tt_entry['depth_searched']:
            transposition_table[canonical_board_tuple_arg] = {
                'score': min_eval, 'depth_searched': remaining_depth_tt, 
                'flag': flag, 'best_move': best_move_this_node}
            tt_stores +=1
        return (min_eval, nodes_here, max_depth_here)

def minimax_iterative(board_tuple_orig, game_depth_of_board, is_turn_of_maximizer, k_to_win, alpha, beta,
                      search_ply_limit_from_here):
    canonical_initial_board = get_canonical_board_tuple(board_tuple_orig)
    score, nodes, max_rec_depth = _minimax_recursive_logic(
        canonical_initial_board, 0, is_turn_of_maximizer, k_to_win, alpha, beta, search_ply_limit_from_here)
    return score, nodes, max_rec_depth

def find_best_move_iterative_deepening(board_config, k_to_win_config, progress_q, result_q, player_token_for_move):
    global transposition_table, tt_hits, tt_stores
    transposition_table.clear(); tt_hits = 0; tt_stores = 0
    start_total_time = time.monotonic()
    is_maximizing_search = (player_token_for_move == PLAYER_HUMAN)
    board_tuple_config = tuple(map(tuple, board_config))
    
    current_game_depth_abs = sum(row.count(PLAYER_HUMAN) + row.count(PLAYER_AI) for row in board_config)
    max_remaining_plies = (len(board_config) * len(board_config[0])) - current_game_depth_abs
    
    if max_remaining_plies == 0:
        result_q.put({'best_move_data': (None, 0), 'top_moves_list': [], 'total_nodes': 0, 'max_search_depth': current_game_depth_abs}); return

    initial_available_moves = get_available_moves([list(row) for row in board_tuple_config])
    if not initial_available_moves:
        result_q.put({'best_move_data': (None, 0), 'top_moves_list': [], 'total_nodes': 0, 'max_search_depth': current_game_depth_abs}); return

    opponent_token = PLAYER_AI if player_token_for_move == PLAYER_HUMAN else PLAYER_HUMAN
    immediate_wins_player = _get_immediate_winning_moves(board_tuple_config, player_token_for_move, k_to_win_config)
    if immediate_wins_player:
        best_move = immediate_wins_player[0]
        score = (1000 - 0) if is_maximizing_search else (-1000 + 0)
        result_q.put({'best_move_data': (best_move, score),
                      'top_moves_list': [{'move': best_move, 'score': score, 'nodes': 1, 'depth': 0, 'actual_eval_depth': 0}],
                      'total_nodes': 1, 'max_search_depth': current_game_depth_abs}); return
    
    opponent_immediate_wins = _get_immediate_winning_moves(board_tuple_config, opponent_token, k_to_win_config)
    if opponent_immediate_wins and len(opponent_immediate_wins) == 1:
        must_block_move = opponent_immediate_wins[0]
        if must_block_move in initial_available_moves: 
            initial_available_moves = [must_block_move]

    overall_best_move_info = {'move': None, 'score': -math.inf if is_maximizing_search else math.inf}
    move_scores_from_last_iter = {move: (-math.inf if is_maximizing_search else math.inf) for move in initial_available_moves}
    accumulated_total_nodes = 0
    max_depth_reached_in_any_minimax_absolute = current_game_depth_abs 
    last_iter_full_eval_details = []

    for current_iddfs_ply_limit in range(1, max_remaining_plies + 1):
        progress_q.put({'total_root_moves_this_iter': len(initial_available_moves),
                         'current_depth_iter': current_iddfs_ply_limit, 'type': 'start_iter',
                         'max_total_depth_iters': max_remaining_plies })
        if current_iddfs_ply_limit > 1:
            initial_available_moves.sort(key=lambda m: move_scores_from_last_iter.get(m, (-math.inf if is_maximizing_search else math.inf)),
                                         reverse=is_maximizing_search)
        current_iter_best_move_candidate = None
        current_iter_best_score_candidate = -math.inf if is_maximizing_search else math.inf
        temp_evaluated_this_iter = []
        alpha_iddfs_root = -math.inf; beta_iddfs_root = math.inf

        for idx, move_to_eval in enumerate(initial_available_moves):
            r, c = move_to_eval
            temp_board = [list(row) for row in board_tuple_config]
            temp_board[r][c] = player_token_for_move
            board_tuple_after_move = tuple(tuple(inner_list) for inner_list in temp_board)
            next_player_is_human_x = (player_token_for_move == PLAYER_AI)
            minimax_search_ply_limit_for_children = current_iddfs_ply_limit - 1
            if minimax_search_ply_limit_for_children < 0: minimax_search_ply_limit_for_children = 0
            
            eval_score, nodes_in_branch, actual_depth_this_branch_relative = minimax_iterative(
                board_tuple_after_move, current_game_depth_abs + 1, next_player_is_human_x, 
                k_to_win_config, -math.inf, math.inf, minimax_search_ply_limit_for_children)
            
            accumulated_total_nodes += nodes_in_branch
            max_depth_reached_in_any_minimax_absolute = max(max_depth_reached_in_any_minimax_absolute, 
                                                            current_game_depth_abs + 1 + actual_depth_this_branch_relative)
            move_scores_from_last_iter[move_to_eval] = eval_score
            current_move_details = {'move': move_to_eval, 'score': eval_score, 'nodes': nodes_in_branch,
                                    'depth': current_iddfs_ply_limit, 
                                    'actual_eval_depth': actual_depth_this_branch_relative + 1}
            temp_evaluated_this_iter.append(current_move_details)

            if is_maximizing_search:
                if eval_score > current_iter_best_score_candidate:
                    current_iter_best_score_candidate = eval_score; current_iter_best_move_candidate = move_to_eval
                alpha_iddfs_root = max(alpha_iddfs_root, eval_score) 
            else:
                if eval_score < current_iter_best_score_candidate:
                    current_iter_best_score_candidate = eval_score; current_iter_best_move_candidate = move_to_eval
                beta_iddfs_root = min(beta_iddfs_root, eval_score)
            
            nps_so_far = accumulated_total_nodes / (time.monotonic() - start_total_time if (time.monotonic() - start_total_time) > 0.001 else 1)
            progress_q.put({'type': 'progress', 'current_nodes': accumulated_total_nodes,
                             'current_max_depth': max_depth_reached_in_any_minimax_absolute,
                             'current_best_score': current_iter_best_score_candidate if current_iter_best_move_candidate else "...",
                             'current_nps': nps_so_far, 'time_elapsed': time.monotonic() - start_total_time,
                             'current_depth_iter': current_iddfs_ply_limit,
                             'root_moves_done_this_iter': idx + 1,
                             'total_root_moves_this_iter': len(initial_available_moves),
                             'max_total_depth_iters': max_remaining_plies })
        
        last_iter_full_eval_details = temp_evaluated_this_iter
        if current_iter_best_move_candidate is not None:
            overall_best_move_info = {'move': current_iter_best_move_candidate, 'score': current_iter_best_score_candidate}
            if is_maximizing_search and current_iter_best_score_candidate >= (1000 - current_iddfs_ply_limit): break 
            if not is_maximizing_search and current_iter_best_score_candidate <= (-1000 + current_iddfs_ply_limit): break 
        if abs(overall_best_move_info['score']) > 990 : break

    if overall_best_move_info['move'] is None and initial_available_moves:
        overall_best_move_info['move'] = initial_available_moves[0]
        overall_best_move_info['score'] = move_scores_from_last_iter.get(initial_available_moves[0], (-math.inf if is_maximizing_search else math.inf))

    if last_iter_full_eval_details:
        last_iter_full_eval_details.sort(key=lambda x: x['score'], reverse=is_maximizing_search)
        final_top_moves_list = last_iter_full_eval_details[:5]
    else:
        if overall_best_move_info['move']:
             final_top_moves_list = [{'move': overall_best_move_info['move'], 'score': overall_best_move_info['score'], 'nodes': 1, 'depth':1, 'actual_eval_depth':1}]
        else: final_top_moves_list = []

    result_q.put({
        'best_move_data': (overall_best_move_info['move'], overall_best_move_info['score']),
        'top_moves_list': final_top_moves_list,
        'total_nodes': accumulated_total_nodes,
        'max_search_depth': max_depth_reached_in_any_minimax_absolute
    })

# --- Tkinter GUI Class ---
class TicTacToeGUI:
    def __init__(self, root_window):
        self.root = root_window
        print(f"Initializing GUI. AI: Single-threaded IDDFS with TT.")
        
        self.root.title(f"Tic-Tac-Toe AI (IDDFS+TT)") 
        self.root.configure(bg=COLOR_ROOT_BG)
        try: self.default_font_family = tkfont.nametofont("TkDefaultFont").actual()["family"]
        except (KeyError, AttributeError, tk.TclError): self.default_font_family = "Segoe UI" if "win" in self.root.tk.call("tk", "windowingsystem") else "Helvetica"
        
        self.fonts = { 
            "header": tkfont.Font(family=self.default_font_family, size=18, weight="bold"),
            "status": tkfont.Font(family=self.default_font_family, size=12, weight="bold"),
            "label": tkfont.Font(family=self.default_font_family, size=10),
            "button": tkfont.Font(family=self.default_font_family, size=10, weight="bold"),
            "entry": tkfont.Font(family=self.default_font_family, size=10),
            "info_header": tkfont.Font(family=self.default_font_family, size=11, weight="bold"),
            "info": tkfont.Font(family=self.default_font_family, size=9),
            "info_value": tkfont.Font(family=self.default_font_family, size=9, weight="bold"),
            "move_info": tkfont.Font(family="Consolas", size=9) 
        }
        self.setup_styles()
        self.board_size = 4; self.k_to_win = 4 
        self.current_player = PLAYER_HUMAN
        self.game_board = [] 
        self.buttons = []    
        self.game_over = True 
        
        self.calculation_manager_thread = None
        self.progress_queue = thread_queue.Queue()
        self.result_queue = thread_queue.Queue()
        self.root.attributes("-fullscreen", True); self.is_fullscreen = True
        
        self.nodes_explored_var = tk.StringVar(value="0")
        self.actual_depth_var = tk.StringVar(value="0") 
        self.target_depth_var = tk.StringVar(value="0") 
        self.ai_eval_var = tk.StringVar(value="N/A")
        self.status_var = tk.StringVar(value="Loading game...") 
        self.time_taken_var = tk.StringVar(value="N/A")
        self.top_moves_var = tk.StringVar(value="Top Move Considerations:\n(After calculation)")
        self.nps_var = tk.StringVar(value="N/A")
        self.progress_percent_var = tk.StringVar(value="Calculating...")
        self.detailed_progress_var = tk.StringVar(value="")
        self.hint_suggestion_var = tk.StringVar(value="")

        self.setup_ui_layout()
        
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.on_silent_close)
        self.root.protocol("WM_DELETE_WINDOW", self.on_silent_close)

        self.root.after_idle(lambda: self.start_new_game(human_starts=True))


    def setup_styles(self):
        self.style = ttk.Style()
        available_themes = self.style.theme_names(); current_os = self.root.tk.call("tk", "windowingsystem")
        if 'clam' in available_themes: self.style.theme_use('clam')
        elif current_os == "win32" and 'vista' in available_themes: self.style.theme_use('vista')
        elif current_os == "aqua" and 'aqua' in available_themes: self.style.theme_use('aqua')
        else: self.style.theme_use(available_themes[0] if available_themes else 'default')
        
        self.style.configure(".", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"])
        self.style.configure("TFrame", background=COLOR_ROOT_BG)
        self.style.configure("Content.TFrame", background=COLOR_FRAME_BG)
        
        self.style.configure("TLabel", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"], padding=2)
        self.style.configure("Header.TLabel", font=self.fonts["header"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_FRAME_BG)
        self.style.configure("Status.TLabel", font=self.fonts["status"], foreground=COLOR_TEXT_PRIMARY, background=COLOR_FRAME_BG) 
        self.style.configure("Info.TLabel", font=self.fonts["info"], foreground=COLOR_TEXT_SECONDARY, background=COLOR_FRAME_BG)
        self.style.configure("InfoValue.TLabel", font=self.fonts["info_value"], foreground=COLOR_TEXT_PRIMARY, background=COLOR_FRAME_BG)
        self.style.configure("DetailedProgress.TLabel", font=self.fonts["info"], foreground=COLOR_TEXT_SECONDARY, background=COLOR_FRAME_BG, anchor=tk.CENTER) 
        self.style.configure("HintSuggest.TLabel", font=self.fonts["info_value"], foreground=COLOR_HINT_TEXT, background=COLOR_HINT_SUGGEST_BG, padding=5, anchor=tk.CENTER, borderwidth=1, relief="solid", bordercolor=COLOR_HINT)
        self.style.configure("MoveInfo.TLabel", font=self.fonts["move_info"], background=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, padding=5, borderwidth=1, relief="groove", bordercolor="#CCCCCC")
        
        self.style.configure("TButton", font=self.fonts["button"], padding=(8, 5), borderwidth=1)
        self.style.map("TButton", relief=[('pressed', 'sunken'), ('!pressed', 'raised')], foreground=[('disabled', COLOR_TEXT_SECONDARY)], background=[('disabled', "#E0E0E0")])
        self.style.configure("Accent.TButton", font=self.fonts["button"],padding=(8,5),borderwidth=1,relief="raised",background=COLOR_ACCENT_PRIMARY,foreground=COLOR_TEXT_ON_ACCENT)
        self.style.map("Accent.TButton", background=[('pressed', COLOR_ACCENT_SECONDARY), ('active', COLOR_ACCENT_SECONDARY), ('!disabled', COLOR_ACCENT_PRIMARY)], foreground=[('pressed', COLOR_TEXT_ON_ACCENT), ('active', COLOR_TEXT_ON_ACCENT), ('!disabled', COLOR_TEXT_ON_ACCENT), ('disabled', COLOR_TEXT_SECONDARY)])
        self.style.configure("Hint.TButton", font=self.fonts["button"],padding=(8,5),borderwidth=1,relief="raised",background=COLOR_HINT_BG, foreground=COLOR_HINT_TEXT)
        self.style.map("Hint.TButton", background=[('active', "#FFD700"), ('pressed', "#EAA600"), ('!disabled', COLOR_HINT_BG)], foreground=[('active', COLOR_HINT_TEXT), ('pressed', COLOR_HINT_TEXT), ('!disabled', COLOR_HINT_TEXT), ('disabled', COLOR_TEXT_SECONDARY)])
        
        self.style.configure("TLabelframe", background=COLOR_ROOT_BG, bordercolor=COLOR_ACCENT_SECONDARY, padding=8)
        self.style.configure("TLabelframe.Label", font=self.fonts["info_header"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_ROOT_BG) 
        self.style.configure("Content.TLabelframe", background=COLOR_FRAME_BG, bordercolor=COLOR_ACCENT_PRIMARY, padding=8) 
        self.style.configure("Content.TLabelframe.Label", font=self.fonts["info_header"], foreground=COLOR_ACCENT_PRIMARY, background=COLOR_FRAME_BG)
        
        self.style.configure("TEntry", font=self.fonts["entry"], padding=3, fieldbackground=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, bordercolor="#CCCCCC")
        self.style.configure("Horizontal.TProgressbar", thickness=12, background=COLOR_ACCENT_PRIMARY, troughcolor="#DDDDDD", bordercolor=COLOR_ACCENT_SECONDARY)

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        if self.buttons: self.root.after(50, self.create_board_ui_buttons)

    def setup_ui_layout(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # --- Left Pane (Controls & Info) ---
        left_pane_container = ttk.Frame(self.paned_window, padding=(5,0))
        self.paned_window.add(left_pane_container, weight=1) # Give it some weight, user can adjust

        left_pane_content = ttk.Frame(left_pane_container, style="Content.TFrame", relief="sunken", borderwidth=1, padding=10)
        left_pane_content.pack(fill=tk.BOTH, expand=True)
        # ... (All the content of left_pane_content: header, control_frame, status_vis_frame, etc. remains IDENTICAL to your last fully working code)
        left_pane_content.grid_columnconfigure(0, weight=1)
        left_pane_content.grid_rowconfigure(0, weight=0); left_pane_content.grid_rowconfigure(1, weight=0) 
        left_pane_content.grid_rowconfigure(2, weight=1) 
        
        header_label = ttk.Label(left_pane_content, text="Tic-Tac-Toe AI", style="Header.TLabel", anchor=tk.CENTER)
        header_label.grid(row=0, column=0, pady=(5,15), sticky="ew")
        
        control_frame = ttk.Labelframe(left_pane_content, text="Game Settings", padding=(10,5), style="Content.TLabelframe")
        control_frame.grid(row=1, column=0, sticky="new", pady=(0,10), padx=0) # Reduced padx
        control_frame.grid_columnconfigure(1, weight=1) 
        
        entry_padx = (5,2); entry_pady = (2,2); label_padx = (5,0) 
        lbl_board_size = ttk.Label(control_frame, text="Board Size:", style="Info.TLabel") 
        lbl_board_size.grid(row=0, column=0, padx=label_padx, pady=entry_pady, sticky="w")
        self.size_entry = ttk.Entry(control_frame, width=4, font=self.fonts["entry"]) 
        self.size_entry.insert(0, str(self.board_size))
        self.size_entry.grid(row=0, column=1, padx=entry_padx, pady=entry_pady, sticky="ew")
        
        lbl_k_row = ttk.Label(control_frame, text="K-in-a-row:", style="Info.TLabel")
        lbl_k_row.grid(row=1, column=0, padx=label_padx, pady=entry_pady, sticky="w")
        self.k_entry = ttk.Entry(control_frame, width=4, font=self.fonts["entry"]) 
        self.k_entry.insert(0, str(self.k_to_win))
        self.k_entry.grid(row=1, column=1, padx=entry_padx, pady=entry_pady, sticky="ew")
        
        button_frame = ttk.Frame(control_frame, style="Content.TFrame", padding=(0,5)) 
        button_frame.grid(row=2, column=0, columnspan=2, pady=(8,0), sticky="ew")
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        btn_padx = 1; btn_pady = 1 
        self.start_human_button = ttk.Button(button_frame, text="▶ You Start", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=True))
        self.start_human_button.grid(row=0, column=0, padx=(0,btn_padx), pady=btn_pady, sticky="ew")
        self.start_ai_button = ttk.Button(button_frame, text="🤖 AI Starts", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=False))
        self.start_ai_button.grid(row=0, column=1, padx=(btn_padx,0), pady=btn_pady, sticky="ew")
        self.hint_button = ttk.Button(button_frame, text="💡 Suggest", style="Hint.TButton", command=self.get_human_hint, state=tk.DISABLED) 
        self.hint_button.grid(row=1, column=0, columnspan=2, pady=(btn_pady+2,0), sticky="ew")

        status_vis_frame = ttk.Labelframe(left_pane_content, text="AI Insights", style="Content.TLabelframe") 
        status_vis_frame.grid(row=2, column=0, sticky="nsew", pady=(5,0), padx=0) # Reduced padx
        status_vis_frame.grid_columnconfigure(0, weight=1)
        status_vis_frame.grid_rowconfigure(5, weight=1) 

        self.status_label = ttk.Label(status_vis_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=280, anchor=tk.W, justify=tk.LEFT) 
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(0,8), sticky="new")
        
        progress_bar_frame = ttk.Frame(status_vis_frame, style="Content.TFrame")
        progress_bar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,1))
        progress_bar_frame.columnconfigure(0, weight=1) 
        progress_bar_frame.columnconfigure(1, weight=0) 
        self.progress_bar = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=100, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,3))
        self.progress_percent_label = ttk.Label(progress_bar_frame, textvariable=self.progress_percent_var, style="Info.TLabel", anchor=tk.W)
        self.progress_percent_label.grid(row=0, column=1, sticky="w")

        self.detailed_progress_label = ttk.Label(status_vis_frame, textvariable=self.detailed_progress_var, style="DetailedProgress.TLabel", anchor=tk.W)
        self.detailed_progress_label.grid(row=2, column=0, columnspan=2, pady=(0,5), sticky="new", padx=2) # Reduced pady
        
        stats_grid = ttk.Frame(status_vis_frame, style="Content.TFrame")
        stats_grid.grid(row=3, column=0, columnspan=2, sticky="new", pady=(0,5)) # Reduced pady
        stats_grid.columnconfigure(0, weight=0, minsize=130) 
        stats_grid.columnconfigure(1, weight=1)    

        row_idx_stats = 0
        stats_to_display = [("Move Score:", self.ai_eval_var), ("Time Taken (s):", self.time_taken_var),
            ("Nodes Explored:", self.nodes_explored_var), ("Nodes/Sec (NPS):", self.nps_var),
            ("Game Max Moves:", self.actual_depth_var), ("IDDFS Target:", self.target_depth_var)] 
        for label_text, var in stats_to_display:
            lbl = ttk.Label(stats_grid, text=label_text, style="Info.TLabel")
            lbl.grid(row=row_idx_stats, column=0, sticky="w", padx=(2,5), pady=1) 
            val_lbl = ttk.Label(stats_grid, textvariable=var, style="InfoValue.TLabel")
            val_lbl.grid(row=row_idx_stats, column=1, sticky="w", padx=2, pady=1)
            row_idx_stats += 1
        
        score_info_lbl = ttk.Label(stats_grid, text="(Scr: >0 X, <0 O, 0 Draw)", style="Info.TLabel", foreground="#6c757d") 
        score_info_lbl.grid(row=row_idx_stats, column=0, columnspan=2, sticky="w", padx=2, pady=(1,3))
        
        self.hint_suggestion_label = ttk.Label(status_vis_frame, textvariable=self.hint_suggestion_var, style="HintSuggest.TLabel", wraplength=280) 
        self.hint_suggestion_label.grid(row=row_idx_stats + 1, column=0, columnspan=2, pady=3, sticky="new")
        
        self.top_moves_label = ttk.Label(status_vis_frame, textvariable=self.top_moves_var, style="MoveInfo.TLabel", anchor="nw", wraplength=280) 
        self.top_moves_label.grid(row=row_idx_stats + 2, column=0, columnspan=2, pady=(3,0), sticky="nsew")


        # --- Right Pane (Board Area) ---
        # This outer frame will be added to the paned window.
        # It will try to expand and will center the board_frame within it.
        self.board_outer_frame = ttk.Frame(self.paned_window, style="TFrame", padding=10) # Padding around board area
        self.paned_window.add(self.board_outer_frame, weight=2) # Give it more weight to expand

        # Configure board_outer_frame to center its child (board_frame)
        self.board_outer_frame.grid_rowconfigure(0, weight=1)
        self.board_outer_frame.grid_columnconfigure(0, weight=1)
        
        # This frame holds the actual buttons. It will be sized by create_board_ui_buttons
        # to be square and then centered by board_outer_frame's grid config.
        self.board_frame = ttk.Frame(self.board_outer_frame, style="TFrame")
        self.board_frame.grid(row=0, column=0, sticky="") # NOT "nsew", so it can be centered

        self.status_var.set("Adjust N, K, then Start.")

    def on_silent_close(self, event=None):
        self.root.destroy()

    def start_new_game(self, human_starts=True):
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showwarning("Busy", "AI calculation is currently in progress."); return
        try:
            n = int(self.size_entry.get()); k = int(self.k_entry.get())
            if not (2 <= n <= 7): messagebox.showerror("Error", "Board size N must be between 2 and 7."); return
            if not (2 <= k <= n): messagebox.showerror("Error", f"K-in-a-row must be between 2 and N."); return
            self.board_size = n; self.k_to_win = k
        except ValueError: messagebox.showerror("Error", "Invalid input for N or K."); return

        global transposition_table 
        transposition_table = {} 
        get_canonical_board_tuple.cache_clear()
        _get_immediate_winning_moves.cache_clear(); can_force_win_on_next_player_turn.cache_clear()
        is_line_still_possible.cache_clear(); can_player_still_win.cache_clear(); is_unwinnable_for_either.cache_clear()

        self.game_board = [[EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = PLAYER_HUMAN if human_starts else PLAYER_AI
        self.game_over = False; self.hint_suggestion_var.set("")
        initial_status = f"{self.board_size}x{self.board_size}, {self.k_to_win}-in-a-row. "
        initial_status += "Your turn (X)." if human_starts else "AI's turn (O)."
        self.status_var.set(initial_status)
        self.ai_eval_var.set("N/A"); self.nodes_explored_var.set("0")
        self.actual_depth_var.set("0"); self.target_depth_var.set("0")
        self.time_taken_var.set("N/A"); self.nps_var.set("N/A")
        self.progress_percent_var.set("Calculating..."); self.detailed_progress_var.set("")
        self.top_moves_var.set("Top Move Considerations:\n(After calculation)")
        
        if self.progress_bar.master.winfo_ismapped(): self.progress_bar.master.grid_remove()
        if self.detailed_progress_label.winfo_ismapped(): self.detailed_progress_label.grid_remove()

        self.create_board_ui_buttons(); self.update_hint_button_state()
        if not human_starts:
            self.root.after(100, lambda: self.trigger_ai_or_hint_calculation(PLAYER_AI))

    def update_hint_button_state(self):
        is_calc_running = self.calculation_manager_thread and self.calculation_manager_thread.is_alive()
        self.hint_button.config(state=tk.NORMAL if not self.game_over and self.current_player == PLAYER_HUMAN and not is_calc_running else tk.DISABLED)
            
    def get_human_hint(self):
        if self.game_over or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showinfo("Busy", "AI calculation is currently in progress."); return
        self.hint_suggestion_var.set("Calculating your best move...")
        self.trigger_ai_or_hint_calculation(PLAYER_HUMAN)

    def trigger_ai_or_hint_calculation(self, player_to_calculate_for):
        is_hint = (player_to_calculate_for == PLAYER_HUMAN)
        self.status_var.set("Calculating hint for Human (X) using IDDFS..." if is_hint else "AI (O) is thinking (IDDFS)...")
        self.top_moves_var.set("Top Move Considerations:\nCalculating...")
        self.disable_board_buttons(); self.hint_button.config(state=tk.DISABLED)
        
        self.progress_bar.master.grid()
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self.progress_percent_label.grid(row=0, column=1, sticky="w")
        self.detailed_progress_label.grid()

        self.progress_bar["value"] = 0
        self.progress_percent_var.set("Starting...") 
        self.detailed_progress_var.set("Initializing search...")
        
        current_board_depth = self._get_current_game_depth_for_ui()
        iddfs_target_depth = (self.board_size * self.board_size) - current_board_depth
        self.target_depth_var.set(str(iddfs_target_depth if iddfs_target_depth > 0 else 0))


        self.root.config(cursor="watch"); self.root.update_idletasks()
        while not self.progress_queue.empty(): self.progress_queue.get_nowait()
        while not self.result_queue.empty(): self.result_queue.get_nowait()
        self.calculation_manager_thread = threading.Thread(
            target=find_best_move_iterative_deepening,
            args=(self.game_board, self.k_to_win, self.progress_queue, self.result_queue, player_to_calculate_for),
            daemon=True )
        self.ai_start_time = time.monotonic()
        self._current_calculation_for_player = player_to_calculate_for
        self.calculation_manager_thread.start(); self.check_ai_progress()

    def create_board_ui_buttons(self, attempt=1):
        for widget in self.board_frame.winfo_children(): widget.destroy()
        self.buttons = []
        
        self.root.update_idletasks() 
        
        # Get dimensions from board_outer_frame, which is managed by PanedWindow
        container_width = self.board_outer_frame.winfo_width()
        container_height = self.board_outer_frame.winfo_height()
        
        if (container_width < 50 or container_height < 50) and attempt < 10:
            self.root.after(100, lambda: self.create_board_ui_buttons(attempt + 1))
            return

        if container_width < 50 or container_height < 50: # Fallback if still too small
            min_dim_root = min(self.root.winfo_width(), self.root.winfo_height()) * 0.6 
            container_width = max(min_dim_root, 200) 
            container_height = max(min_dim_root, 200)

        # Determine the size of the square board based on the smaller dimension of the container
        board_dimension = min(container_width, container_height) - 10 # -10 for some padding within outer frame
        board_dimension = max(self.board_size * 35, board_dimension) # Ensure a minimum size per cell

        cell_dim = board_dimension // self.board_size
        
        # Set the width and height of board_frame to be square
        self.board_frame.config(width=board_dimension, height=board_dimension)
        
        btn_font_size = max(10, int(cell_dim * 0.50))
        board_button_font = tkfont.Font(family=self.default_font_family, size=btn_font_size, weight="bold")

        for i in range(self.board_size):
            self.board_frame.grid_rowconfigure(i, weight=1, minsize=cell_dim)
            self.board_frame.grid_columnconfigure(i, weight=1, minsize=cell_dim)

        for r in range(self.board_size):
            row_buttons = []
            for c_idx in range(self.board_size):
                button = tk.Button(self.board_frame, text=EMPTY_CELL, font=board_button_font, 
                                   relief="flat", borderwidth=1, 
                                   bg=COLOR_FRAME_BG, activebackground="#DDDDDD", 
                                   fg=COLOR_TEXT_PRIMARY, 
                                   command=lambda r_idx=r, c_idx_btn=c_idx: self.handle_cell_click(r_idx, c_idx_btn))
                button.grid(row=r, column=c_idx, sticky="nsew", padx=1, pady=1)
                row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board_button_states()

    def handle_cell_click(self, r, c):
        if self.game_over or self.game_board[r][c] != EMPTY_CELL or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showinfo("Wait", "AI calculation is currently in progress."); return
        self.hint_suggestion_var.set("")
        self.make_move(r, c, PLAYER_HUMAN)
        if self.check_game_status(): return
        self.current_player = PLAYER_AI
        self.update_hint_button_state()
        self.trigger_ai_or_hint_calculation(PLAYER_AI)
    
    def _get_current_game_depth_for_ui(self): # HELPER METHOD
        depth = 0
        if hasattr(self, 'game_board') and self.game_board and \
           hasattr(self, 'board_size') and self.board_size > 0 and \
           len(self.game_board) == self.board_size:
            for r in range(self.board_size):
                if len(self.game_board[r]) == self.board_size:
                    for c in range(self.board_size):
                        if self.game_board[r][c] != EMPTY_CELL:
                            depth += 1
                else: return sum(sum(1 for cell in row if cell != EMPTY_CELL) for row in self.game_board) 
        else: return 0 
        return depth

    def check_ai_progress(self):
        update_interval_ms = 100 
        current_runtime = 0.0 

        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            current_runtime = time.monotonic() - self.ai_start_time
            self.time_taken_var.set(f"{current_runtime:.2f}") 

        try:
            progress_update = self.progress_queue.get_nowait() 
            
            current_nodes_from_ai = progress_update.get('current_nodes')
            if current_nodes_from_ai is not None: 
                self.nodes_explored_var.set(f"{current_nodes_from_ai:,}")
                if current_runtime > 0.01: 
                    nps_val = current_nodes_from_ai / current_runtime
                    self.nps_var.set(f"{nps_val:,.0f}")
                elif current_nodes_from_ai > 0 : 
                    self.nps_var.set("High") 
                else: 
                    self.nps_var.set("...") 
            
            current_depth_iter_from_ai = progress_update.get('current_depth_iter', 0) 
            max_remaining_plies_for_overall_percent = progress_update.get('max_total_depth_iters', 
                                                                     (self.board_size * self.board_size) - self._get_current_game_depth_for_ui())

            if progress_update.get('type') == 'start_iter':
                self.progress_bar["maximum"] = progress_update.get('total_root_moves_this_iter', 1)
                self.progress_bar["value"] = 0 
                self.detailed_progress_var.set(f"Depth {current_depth_iter_from_ai} (Move 0/{self.progress_bar['maximum']})")
                
                overall_percent = (current_depth_iter_from_ai / max_remaining_plies_for_overall_percent) * 100 if max_remaining_plies_for_overall_percent > 0 else 0
                percent_str = f"{overall_percent:.0f}".strip() 
                self.progress_percent_var.set(f"Max {max_remaining_plies_for_overall_percent} plies ({percent_str}%)")


            elif progress_update.get('type') == 'progress':
                if 'current_max_depth' in progress_update: self.actual_depth_var.set(f"{progress_update['current_max_depth']}")
                if 'current_best_score' in progress_update:
                    score_val = progress_update['current_best_score']
                    self.ai_eval_var.set(f"{score_val:.0f}" if isinstance(score_val, (int, float)) and score_val not in (math.inf, -math.inf) else str(score_val))
                
                root_moves_done = progress_update.get('root_moves_done_this_iter',0)
                total_root_moves_for_bar = progress_update.get('total_root_moves_this_iter', self.progress_bar["maximum"])
                
                if total_root_moves_for_bar > 0 :
                    self.progress_bar["value"] = root_moves_done
                    self.progress_bar["maximum"] = total_root_moves_for_bar 
                    self.detailed_progress_var.set(f"Depth {current_depth_iter_from_ai} (Move {root_moves_done}/{total_root_moves_for_bar})")
                else: 
                     self.detailed_progress_var.set(f"Depth {current_depth_iter_from_ai} (Calculating...)")
                
                overall_percent = (current_depth_iter_from_ai / max_remaining_plies_for_overall_percent) * 100 if max_remaining_plies_for_overall_percent > 0 else 0
                percent_str = f"{overall_percent:.0f}".strip() 
                self.progress_percent_var.set(f"Target {max_remaining_plies_for_overall_percent} plies ({percent_str}%)")
        
        except thread_queue.Empty: pass 
        
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            self.root.after(update_interval_ms, self.check_ai_progress)
        elif self.calculation_manager_thread: 
            final_runtime = time.monotonic() - self.ai_start_time 
            self.time_taken_var.set(f"{final_runtime:.3f}") 

            try: 
                final_progress_update = self.progress_queue.get_nowait()
                final_nodes = final_progress_update.get('current_nodes')
                if final_nodes is not None:
                    self.nodes_explored_var.set(f"{final_nodes:,}")
                    if final_runtime > 0.01:
                        final_nps = final_nodes / final_runtime
                        self.nps_var.set(f"{final_nps:,.0f}")
                    else:
                        self.nps_var.set("High" if final_nodes > 0 else "0")
                if 'current_max_depth' in final_progress_update: self.actual_depth_var.set(f"{final_progress_update['current_max_depth']}")
                if 'current_best_score' in final_progress_update:
                    score_val = final_progress_update['current_best_score']
                    self.ai_eval_var.set(f"{score_val:.0f}" if isinstance(score_val, (int, float)) and score_val not in (math.inf, -math.inf) else str(score_val))

            except thread_queue.Empty: 
                # If no final message, calculate NPS with last known nodes_explored and current_runtime
                # Ensure nodes_explored_var contains a valid number
                nodes_str = self.nodes_explored_var.get().replace(',', '')
                if nodes_str.isdigit(): # Check if it's purely digits after removing comma
                    nodes_val = int(nodes_str)
                    if current_runtime > 0.01 and nodes_val > 0 :
                        final_nps = nodes_val / current_runtime
                        self.nps_var.set(f"{final_nps:,.0f}")
                    elif nodes_val > 0:
                        self.nps_var.set("High")
                    else: # nodes_val is 0 or not a number
                        self.nps_var.set("0" if nodes_str == "0" else "N/A")
                else: # nodes_explored_var was "N/A" or "Calculating..."
                    self.nps_var.set("N/A")


            self.handle_calculation_result(); self.calculation_manager_thread = None
            self.progress_percent_var.set("Done.")
            self.detailed_progress_var.set("")

    def handle_calculation_result(self):
        self.root.config(cursor="")
        if self.progress_bar.master.winfo_ismapped():
            self.progress_bar.master.grid_remove()
        if self.detailed_progress_label.winfo_ismapped():
            self.detailed_progress_label.grid_remove()
            
        time_taken_calc = time.monotonic() - self.ai_start_time 
        self.time_taken_var.set(f"{time_taken_calc:.3f}") 

        was_hint_calculation = (self._current_calculation_for_player == PLAYER_HUMAN)
        try:
            ai_result_data = self.result_queue.get_nowait()
            best_move_tuple, best_score = ai_result_data['best_move_data']
            top_moves_list = ai_result_data['top_moves_list']
            total_nodes = ai_result_data['total_nodes']
            max_search_depth = ai_result_data['max_search_depth']
        except thread_queue.Empty:
            self.status_var.set("Error: No result from AI calculation."); self.enable_board_buttons(); self.update_hint_button_state(); return
        
        self.nodes_explored_var.set(f"{total_nodes:,}")
        nps_val = total_nodes / time_taken_calc if time_taken_calc > 0.0001 else 0
        self.nps_var.set(f"{nps_val:,.0f}")
        self.actual_depth_var.set(f"{max_search_depth}")
        self.ai_eval_var.set(f"{best_score:.0f}" if best_score not in (math.inf, -math.inf) else "Win/Loss")
        best_move_alg = to_algebraic(best_move_tuple[0], best_move_tuple[1], self.board_size) if best_move_tuple else "N/A"
        perspective = "Human (X)" if was_hint_calculation else "AI (O)"
        top_moves_text = f"Top {perspective} Moves (Move: Score | Nodes | IterD | ActualD):\n"
        if top_moves_list:
            for item in top_moves_list:
                move_tuple = item['move']; move_alg = to_algebraic(move_tuple[0], move_tuple[1], self.board_size)
                score_str = f"{item['score']:.0f}"; nodes_str = f"N:{item.get('nodes',0):,}"
                iter_depth_str = f"ID:{item.get('depth',0)}"; actual_eval_depth_str = f"AD:{item.get('actual_eval_depth',0)}"
                top_moves_text += f"  {move_alg:<4}: {score_str:<5} | {nodes_str:<10}| {iter_depth_str:<5}| {actual_eval_depth_str}\n"
        else: top_moves_text += "  (N/A)\n"
        self.top_moves_var.set(top_moves_text.strip())
        if was_hint_calculation:
            if best_move_tuple:
                self.status_var.set(f"Hint for X: Best is {best_move_alg} (Score: {best_score:.0f}). Your turn.")
                self.hint_suggestion_var.set(f"Suggested for X: {best_move_alg} (Score: {best_score:.0f})")
                if self.game_board[best_move_tuple[0]][best_move_tuple[1]] == EMPTY_CELL and \
                   self.buttons[best_move_tuple[0]][best_move_tuple[1]]['state'] != tk.DISABLED:
                    self.buttons[best_move_tuple[0]][best_move_tuple[1]].config(bg=COLOR_HINT_SUGGEST_BG, relief="raised")
                    self.root.after(3000, lambda r=best_move_tuple[0], c=best_move_tuple[1]: self.clear_hint_highlight(r,c))
            else: self.status_var.set("Hint: No moves available or game ended."); self.hint_suggestion_var.set("No suggestion.")
            self.enable_board_buttons(); self.update_hint_button_state()
        else:
            self.hint_suggestion_var.set("")
            if best_move_tuple:
                self.make_move(best_move_tuple[0], best_move_tuple[1], PLAYER_AI)
                if self.check_game_status(): self.enable_board_buttons(); self.update_hint_button_state(); return
                self.current_player = PLAYER_HUMAN
                ai_moved_to_alg = to_algebraic(best_move_tuple[0], best_move_tuple[1], self.board_size)
                self.status_var.set(f"AI (O) moved to {ai_moved_to_alg}. Your turn (X).")
            else:
                 if is_board_full(self.game_board) and not get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win) and not get_winning_line(self.game_board, PLAYER_AI, self.k_to_win):
                     self.status_var.set("It's a Draw!")
                 else: self.status_var.set("AI error or no moves. Game Over?")
                 self.game_over = True
            self.enable_board_buttons(); self.update_hint_button_state()

    def clear_hint_highlight(self, r, c):
        if self.game_board[r][c] == EMPTY_CELL and self.buttons[r][c]['bg'] == COLOR_HINT_SUGGEST_BG :
            self.buttons[r][c].config(bg=COLOR_FRAME_BG, relief="flat")

    def make_move(self, r, c, player):
        self.game_board[r][c] = player; btn = self.buttons[r][c]
        btn.config(text=player, state=tk.DISABLED, relief=tk.SUNKEN)
        player_color_bg = COLOR_HUMAN_MOVE_BG if player == PLAYER_HUMAN else COLOR_AI_MOVE_BG
        player_fg_color = COLOR_ACCENT_SECONDARY if player == PLAYER_HUMAN else COLOR_DANGER
        btn.config(disabledforeground=player_fg_color, background=player_color_bg)

    def check_game_status(self):
        human_wins_line = get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win)
        ai_wins_line = get_winning_line(self.game_board, PLAYER_AI, self.k_to_win)
        board_is_actually_full = is_board_full(self.game_board)
        changed_game_over_state = False
        if not self.game_over:
            if human_wins_line: self.status_var.set("You (X) Win!"); self.game_over = True; changed_game_over_state = True; self.highlight_winning_line(human_wins_line)
            elif ai_wins_line: self.status_var.set("AI (O) Wins!"); self.game_over = True; changed_game_over_state = True; self.highlight_winning_line(ai_wins_line)
            elif board_is_actually_full: 
                self.status_var.set("It's a Draw (Board Full)!")
                self.game_over = True; changed_game_over_state = True
        if changed_game_over_state or self.game_over: self.disable_board_buttons(); self.update_hint_button_state()
        return self.game_over

    def update_board_button_states(self):
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

    def disable_board_buttons(self):
        for r_buttons in self.buttons:
            for button in r_buttons:
                if button['state'] == tk.NORMAL: button.config(state=tk.DISABLED)

    def enable_board_buttons(self):
        if self.game_over: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): return
        for r_idx in range(self.board_size):
            for c_idx in range(self.board_size):
                if self.game_board[r_idx][c_idx] == EMPTY_CELL:
                    if self.buttons[r_idx][c_idx]['bg'] != COLOR_HINT_SUGGEST_BG:
                         self.buttons[r_idx][c_idx].config(state=tk.NORMAL, bg=COLOR_FRAME_BG, relief="flat")
                    else: self.buttons[r_idx][c_idx].config(state=tk.NORMAL, relief="raised")

    def highlight_winning_line(self, winning_cells):
        for r,c in winning_cells: self.buttons[r][c].config(background=COLOR_WIN_HIGHLIGHT, relief=tk.GROOVE)

# --- Main Execution ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = TicTacToeGUI(main_root)
    main_root.mainloop()