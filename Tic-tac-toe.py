import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import math
import functools
import time
import threading
import queue as thread_queue
import random

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
# tt_hits and tt_stores are reset in find_best_move_iterative_deepening

# --- Zobrist Hashing Constants & Setup ---
ZOBRIST_TABLE = {}

def init_zobrist_for_board_size(current_board_size):
    global ZOBRIST_TABLE
    ZOBRIST_TABLE.clear()
    random.seed(42)
    for r_idx in range(current_board_size):
        for c_idx in range(current_board_size):
            ZOBRIST_TABLE[(r_idx, c_idx, PLAYER_HUMAN)] = random.getrandbits(64)
            ZOBRIST_TABLE[(r_idx, c_idx, PLAYER_AI)] = random.getrandbits(64)

@functools.lru_cache(maxsize=32768)
def compute_zobrist_hash_from_canonical_tuple(canonical_board_tuple):
    h = 0
    for r_idx, row in enumerate(canonical_board_tuple):
        for c_idx, piece in enumerate(row):
            if piece != EMPTY_CELL:
                key = (r_idx, c_idx, piece)
                h ^= ZOBRIST_TABLE[key]
    return h

# --- Symmetry Helper Functions ---
def _rotate_board_tuple(board_tuple): # Optimized version
    n = len(board_tuple)
    n_minus_1 = n - 1
    rotated_rows_list = []
    for r_new in range(n):
        # new_board[r_new][c_new] = board_tuple[n_minus_1 - c_new][r_new]
        row_elements = [board_tuple[n_minus_1 - c_new][r_new] for c_new in range(n)]
        rotated_rows_list.append(tuple(row_elements))
    return tuple(rotated_rows_list)

def _reflect_board_tuple(board_tuple):
    return tuple(row[::-1] for row in board_tuple)

@functools.lru_cache(maxsize=8192) 
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

# --- Game Logic (Core) ---
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

# Renamed get_available_moves_lean back to get_available_moves
def get_available_moves(board_list_mutable): # Removed unused player/k_to_win args
    n = len(board_list_mutable)
    center_coord_val = (n - 1) / 2.0 
    empty_cells = []
    for r_idx in range(n): 
        for c_idx in range(n): 
            if board_list_mutable[r_idx][c_idx] == EMPTY_CELL:
                empty_cells.append((r_idx, c_idx))
    
    empty_cells.sort(key=lambda move: (abs(move[0] - center_coord_val) + abs(move[1] - center_coord_val), move[0], move[1]))
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
    board_list = [list(row) for row in board_tuple]
    n = len(board_list)
    winning_moves = []
    for r_try in range(n):
        for c_try in range(n):
            if board_list[r_try][c_try] == EMPTY_CELL:
                board_list[r_try][c_try] = player
                if check_win_for_eval(board_list, player, k_to_win):
                    winning_moves.append((r_try, c_try))
                board_list[r_try][c_try] = EMPTY_CELL
    return winning_moves

@functools.lru_cache(maxsize=32768)
def can_force_win_on_next_player_turn(board_tuple, player_to_move, opponent, k_to_win):
    mutable_board_for_get_moves = [list(row) for row in board_tuple]
    original_player_moves = get_available_moves(mutable_board_for_get_moves) # Now uses the renamed function

    for r_m1, c_m1 in original_player_moves:
        board_after_m1 = [list(row) for row in board_tuple]
        board_after_m1[r_m1][c_m1] = player_to_move
        board_after_m1_tuple = tuple(map(tuple, board_after_m1))
        if check_win_for_eval(board_after_m1_tuple, player_to_move, k_to_win): continue

        mutable_board_after_m1_2 = [list(row) for row in board_after_m1_tuple]
        opponent_available_moves_after_m1 = get_available_moves(mutable_board_after_m1_2) 
        
        if not opponent_available_moves_after_m1:
            if _get_immediate_winning_moves(board_after_m1_tuple, player_to_move, k_to_win):
                return True
            else:
                continue

        m1_is_forcing_so_far = True
        for r_mopp, c_mopp in opponent_available_moves_after_m1:
            board_after_mopp = [list(row) for row in board_after_m1_tuple]
            board_after_mopp[r_mopp][c_mopp] = opponent
            board_after_mopp_tuple = tuple(map(tuple, board_after_mopp))

            if check_win_for_eval(board_after_mopp_tuple, opponent, k_to_win):
                m1_is_forcing_so_far = False; break 
            if not _get_immediate_winning_moves(board_after_mopp_tuple, player_to_move, k_to_win):
                m1_is_forcing_so_far = False; break
        
        if m1_is_forcing_so_far: return True
    return False

# --- Minimax AI (PVS Implemented) ---
def _minimax_recursive_logic(canonical_board_tuple_arg, depth, is_maximizing_for_x, k_to_win, alpha, beta,
                             current_max_search_depth_for_this_call):
    global tt_hits, tt_stores
    nodes_here = 1; max_depth_here = depth
    original_alpha = alpha 

    current_board_hash = compute_zobrist_hash_from_canonical_tuple(canonical_board_tuple_arg)
    remaining_depth_tt = current_max_search_depth_for_this_call - depth

    if current_board_hash in transposition_table:
        entry = transposition_table[current_board_hash]
        if entry['depth_searched'] >= remaining_depth_tt:
            tt_hits += 1
            if entry['flag'] == TT_EXACT: return (entry['score'], nodes_here, max_depth_here)
            elif entry['flag'] == TT_LOWER_BOUND: alpha = max(alpha, entry['score'])
            elif entry['flag'] == TT_UPPER_BOUND: beta = min(beta, entry['score'])
            if alpha >= beta: return (entry['score'], nodes_here, max_depth_here)

    board_list_mutable = [list(row) for row in canonical_board_tuple_arg]
    score_terminal = None
    if check_win_for_eval(canonical_board_tuple_arg, PLAYER_HUMAN, k_to_win): score_terminal = 1000 - depth
    elif check_win_for_eval(canonical_board_tuple_arg, PLAYER_AI, k_to_win): score_terminal = -1000 + depth
    elif is_board_full(canonical_board_tuple_arg): score_terminal = 0
    elif is_unwinnable_for_either(canonical_board_tuple_arg, k_to_win): score_terminal = 0

    if score_terminal is not None:
        transposition_table[current_board_hash] = {
            'score': score_terminal, 'depth_searched': 99,
            'flag': TT_EXACT, 'best_move': None
        }
        tt_stores +=1
        return (score_terminal, nodes_here, max_depth_here)

    if depth >= current_max_search_depth_for_this_call:
        return (0, nodes_here, max_depth_here)

    if (depth + 3) <= current_max_search_depth_for_this_call: 
        score_forced = 0; is_forced = False
        if is_maximizing_for_x:
            if can_force_win_on_next_player_turn(canonical_board_tuple_arg, PLAYER_HUMAN, PLAYER_AI, k_to_win):
                score_forced = 1000 - (depth + 2); is_forced = True 
        else:
            if can_force_win_on_next_player_turn(canonical_board_tuple_arg, PLAYER_AI, PLAYER_HUMAN, k_to_win):
                score_forced = -1000 + (depth + 2); is_forced = True
        if is_forced:
            return (score_forced, nodes_here, max_depth_here)

    available_moves_list = get_available_moves(board_list_mutable) # Using the renamed lean version
    
    if not available_moves_list: return (0, nodes_here, max_depth_here)

    if current_board_hash in transposition_table:
        entry = transposition_table[current_board_hash]
        if entry.get('best_move') is not None:
            bm_r_tt, bm_c_tt = entry['best_move'] 
            if board_list_mutable[bm_r_tt][bm_c_tt] == EMPTY_CELL and entry['best_move'] in available_moves_list:
                available_moves_list.remove(entry['best_move'])
                available_moves_list.insert(0, entry['best_move'])
    
    best_move_this_node = None
    eval_to_store = 0
    if is_maximizing_for_x:
        max_eval = -math.inf
        for i, (r_move, c_move) in enumerate(available_moves_list):
            board_list_mutable[r_move][c_move] = PLAYER_HUMAN
            board_tuple_next = tuple(map(tuple, board_list_mutable))
            canonical_board_tuple_next = get_canonical_board_tuple(board_tuple_next)
            
            current_eval_score = 0; child_nodes_iter = 0; child_max_depth_iter = 0
            if i == 0: 
                current_eval_score, child_nodes_iter, child_max_depth_iter = _minimax_recursive_logic(
                    canonical_board_tuple_next, depth + 1, False, k_to_win, alpha, beta, current_max_search_depth_for_this_call)
            else: 
                null_window_score, child_nodes_null, child_max_depth_null = _minimax_recursive_logic(
                    canonical_board_tuple_next, depth + 1, False, k_to_win, alpha, alpha + 1, current_max_search_depth_for_this_call)
                child_nodes_iter += child_nodes_null; child_max_depth_iter = max(child_max_depth_iter, child_max_depth_null)
                if null_window_score > alpha and null_window_score < beta: 
                    current_eval_score, child_nodes_re, child_max_depth_re = _minimax_recursive_logic(
                        canonical_board_tuple_next, depth + 1, False, k_to_win, alpha, beta, current_max_search_depth_for_this_call)
                    child_nodes_iter += child_nodes_re; child_max_depth_iter = max(child_max_depth_iter, child_max_depth_re)
                else:
                    current_eval_score = null_window_score
            
            nodes_here += child_nodes_iter; max_depth_here = max(max_depth_here, child_max_depth_iter)
            board_list_mutable[r_move][c_move] = EMPTY_CELL

            if current_eval_score > max_eval:
                max_eval = current_eval_score; best_move_this_node = (r_move, c_move)
            alpha = max(alpha, max_eval)
            if alpha >= beta: break
        eval_to_store = max_eval
    else: # Minimizing for O
        min_eval = math.inf
        for i, (r_move, c_move) in enumerate(available_moves_list):
            board_list_mutable[r_move][c_move] = PLAYER_AI
            board_tuple_next = tuple(map(tuple, board_list_mutable))
            canonical_board_tuple_next = get_canonical_board_tuple(board_tuple_next)

            current_eval_score = 0; child_nodes_iter = 0; child_max_depth_iter = 0
            if i == 0:
                current_eval_score, child_nodes_iter, child_max_depth_iter = _minimax_recursive_logic(
                    canonical_board_tuple_next, depth + 1, True, k_to_win, alpha, beta, current_max_search_depth_for_this_call)
            else:
                null_window_score, child_nodes_null, child_max_depth_null = _minimax_recursive_logic(
                    canonical_board_tuple_next, depth + 1, True, k_to_win, beta - 1, beta, current_max_search_depth_for_this_call)
                child_nodes_iter += child_nodes_null; child_max_depth_iter = max(child_max_depth_iter, child_max_depth_null)
                if null_window_score < beta and null_window_score > alpha: 
                    current_eval_score, child_nodes_re, child_max_depth_re = _minimax_recursive_logic(
                        canonical_board_tuple_next, depth + 1, True, k_to_win, alpha, beta, current_max_search_depth_for_this_call)
                    child_nodes_iter += child_nodes_re; child_max_depth_iter = max(child_max_depth_iter, child_max_depth_re)
                else:
                    current_eval_score = null_window_score
            
            nodes_here += child_nodes_iter; max_depth_here = max(max_depth_here, child_max_depth_iter)
            board_list_mutable[r_move][c_move] = EMPTY_CELL

            if current_eval_score < min_eval:
                min_eval = current_eval_score; best_move_this_node = (r_move, c_move)
            beta = min(beta, min_eval)
            if beta <= alpha: break
        eval_to_store = min_eval

    flag = TT_EXACT
    if eval_to_store <= original_alpha: flag = TT_UPPER_BOUND
    elif eval_to_store >= beta: flag = TT_LOWER_BOUND
    
    current_tt_entry = transposition_table.get(current_board_hash)
    if current_tt_entry is None or remaining_depth_tt >= current_tt_entry['depth_searched']:
        transposition_table[current_board_hash] = {
            'score': eval_to_store, 'depth_searched': remaining_depth_tt,
            'flag': flag, 'best_move': best_move_this_node
        }
        tt_stores += 1
    return (eval_to_store, nodes_here, max_depth_here)

def minimax_iterative(board_tuple_orig, game_depth_of_board, is_turn_of_maximizer, k_to_win, alpha, beta,
                      search_ply_limit_from_here):
    canonical_initial_board = get_canonical_board_tuple(board_tuple_orig)
    return _minimax_recursive_logic( 
        canonical_initial_board, 0, is_turn_of_maximizer, k_to_win, alpha, beta, search_ply_limit_from_here)

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

    temp_mutable_board_for_root_moves = [list(row) for row in board_tuple_config]
    initial_available_moves = get_available_moves(temp_mutable_board_for_root_moves) # Using the renamed lean version
    
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
        
        if current_iddfs_ply_limit > 1 and len(initial_available_moves) > 1 :
            initial_available_moves.sort(key=lambda m: move_scores_from_last_iter.get(m, (-math.inf if is_maximizing_search else math.inf)),
                                         reverse=is_maximizing_search)
        
        current_iter_best_move_candidate = None
        current_iter_best_score_candidate = -math.inf if is_maximizing_search else math.inf
        temp_evaluated_this_iter = []

        for idx, move_to_eval in enumerate(initial_available_moves):
            r_root, c_root = move_to_eval 
            temp_board_list_iter_root = [list(row) for row in board_tuple_config] 
            temp_board_list_iter_root[r_root][c_root] = player_token_for_move
            board_tuple_after_move_root = tuple(map(tuple, temp_board_list_iter_root)) 
            
            next_player_is_human_x = (player_token_for_move == PLAYER_AI)
            minimax_search_ply_limit_for_children = current_iddfs_ply_limit - 1
            if minimax_search_ply_limit_for_children < 0: minimax_search_ply_limit_for_children = 0

            eval_score, nodes_in_branch, actual_depth_this_branch_relative = minimax_iterative(
                board_tuple_after_move_root, current_game_depth_abs + 1, next_player_is_human_x,
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
            else:
                if eval_score < current_iter_best_score_candidate:
                    current_iter_best_score_candidate = eval_score; current_iter_best_move_candidate = move_to_eval
            
            current_overall_time = time.monotonic() - start_total_time
            nps_so_far = accumulated_total_nodes / current_overall_time if current_overall_time > 0.001 else accumulated_total_nodes * 1000.0

            progress_q.put({'type': 'progress', 'current_nodes': accumulated_total_nodes,
                             'current_max_depth': max_depth_reached_in_any_minimax_absolute,
                             'current_best_score': current_iter_best_score_candidate if current_iter_best_move_candidate else "...",
                             'current_nps': nps_so_far, 'time_elapsed': current_overall_time, 
                             'current_depth_iter': current_iddfs_ply_limit,
                             'root_moves_done_this_iter': idx + 1,
                             'total_root_moves_this_iter': len(initial_available_moves),
                             'max_total_depth_iters': max_remaining_plies })

        last_iter_full_eval_details = temp_evaluated_this_iter
        if current_iter_best_move_candidate is not None:
            overall_best_move_info = {'move': current_iter_best_move_candidate, 'score': current_iter_best_score_candidate}
            if is_maximizing_search and current_iter_best_score_candidate >= (1000 - (current_iddfs_ply_limit -1) ): 
                break
            if not is_maximizing_search and current_iter_best_score_candidate <= (-1000 + (current_iddfs_ply_limit -1) ):
                break
        if abs(overall_best_move_info['score']) > 990 and overall_best_move_info['move'] is not None : 
             break

    if overall_best_move_info['move'] is None and initial_available_moves:
        overall_best_move_info['move'] = initial_available_moves[0]
        overall_best_move_info['score'] = move_scores_from_last_iter.get(initial_available_moves[0], (-math.inf if is_maximizing_search else math.inf))

    final_top_moves_list = []
    if last_iter_full_eval_details:
        last_iter_full_eval_details.sort(key=lambda x: x['score'], reverse=is_maximizing_search)
        final_top_moves_list = last_iter_full_eval_details[:5]
    elif overall_best_move_info['move']: 
             final_top_moves_list = [{'move': overall_best_move_info['move'], 
                                      'score': overall_best_move_info['score'], 
                                      'nodes': 1, 'depth':1, 'actual_eval_depth':1}]
    
    result_q.put({
        'best_move_data': (overall_best_move_info['move'], overall_best_move_info['score']),
        'top_moves_list': final_top_moves_list,
        'total_nodes': accumulated_total_nodes,
        'max_search_depth': max_depth_reached_in_any_minimax_absolute
    })

# --- Tkinter GUI Class (No changes from your last working version) ---
class TicTacToeGUI:
    def __init__(self, root_window):
        self.root = root_window
        print(f"Initializing GUI. AI: IDDFS+TT (Zobrist, PVS, Ultra-Lean Ordering).") 
        self.root.title(f"Tic-Tac-Toe AI (IDDFS+TT+Zobrist+PVS+UltraLean)") 
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

        left_pane_container = ttk.Frame(self.paned_window, padding=(5,0))
        self.paned_window.add(left_pane_container, weight=1)

        left_pane_content = ttk.Frame(left_pane_container, style="Content.TFrame", relief="sunken", borderwidth=1, padding=10)
        left_pane_content.pack(fill=tk.BOTH, expand=True)
        left_pane_content.grid_columnconfigure(0, weight=1)
        left_pane_content.grid_rowconfigure(2, weight=1)

        header_label = ttk.Label(left_pane_content, text="Tic-Tac-Toe AI", style="Header.TLabel", anchor=tk.CENTER)
        header_label.grid(row=0, column=0, pady=(5,15), sticky="ew")

        control_frame = ttk.Labelframe(left_pane_content, text="Game Settings", padding=(10,5), style="Content.TLabelframe")
        control_frame.grid(row=1, column=0, sticky="new", pady=(0,10), padx=0)
        control_frame.grid_columnconfigure(1, weight=1)

        entry_padx = (5,2); entry_pady = (2,2); label_padx = (5,0)
        ttk.Label(control_frame, text="Board Size:", style="Info.TLabel").grid(row=0, column=0, padx=label_padx, pady=entry_pady, sticky="w")
        self.size_entry = ttk.Entry(control_frame, width=4, font=self.fonts["entry"])
        self.size_entry.insert(0, str(self.board_size)); self.size_entry.grid(row=0, column=1, padx=entry_padx, pady=entry_pady, sticky="ew")
        ttk.Label(control_frame, text="K-in-a-row:", style="Info.TLabel").grid(row=1, column=0, padx=label_padx, pady=entry_pady, sticky="w")
        self.k_entry = ttk.Entry(control_frame, width=4, font=self.fonts["entry"])
        self.k_entry.insert(0, str(self.k_to_win)); self.k_entry.grid(row=1, column=1, padx=entry_padx, pady=entry_pady, sticky="ew")

        button_frame = ttk.Frame(control_frame, style="Content.TFrame", padding=(0,5))
        button_frame.grid(row=2, column=0, columnspan=2, pady=(8,0), sticky="ew")
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        self.start_human_button = ttk.Button(button_frame, text="▶ You Start", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=True))
        self.start_human_button.grid(row=0, column=0, padx=(0,1), pady=1, sticky="ew")
        self.start_ai_button = ttk.Button(button_frame, text="🤖 AI Starts", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=False))
        self.start_ai_button.grid(row=0, column=1, padx=(1,0), pady=1, sticky="ew")
        self.hint_button = ttk.Button(button_frame, text="💡 Suggest", style="Hint.TButton", command=self.get_human_hint, state=tk.DISABLED)
        self.hint_button.grid(row=1, column=0, columnspan=2, pady=(3,0), sticky="ew")

        status_vis_frame = ttk.Labelframe(left_pane_content, text="AI Insights", style="Content.TLabelframe")
        status_vis_frame.grid(row=2, column=0, sticky="nsew", pady=(5,0), padx=0)
        status_vis_frame.grid_columnconfigure(0, weight=1); status_vis_frame.grid_rowconfigure(5, weight=1)

        self.status_label = ttk.Label(status_vis_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=280, anchor=tk.W, justify=tk.LEFT)
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(0,8), sticky="new")

        progress_bar_frame = ttk.Frame(status_vis_frame, style="Content.TFrame")
        progress_bar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,1))
        progress_bar_frame.columnconfigure(0, weight=1); progress_bar_frame.columnconfigure(1, weight=0)
        self.progress_bar = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=100, mode="determinate", style="Horizontal.TProgressbar")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,3))
        self.progress_percent_label = ttk.Label(progress_bar_frame, textvariable=self.progress_percent_var, style="Info.TLabel", anchor=tk.W)
        self.progress_percent_label.grid(row=0, column=1, sticky="w")
        self.detailed_progress_label = ttk.Label(status_vis_frame, textvariable=self.detailed_progress_var, style="DetailedProgress.TLabel", anchor=tk.W)
        self.detailed_progress_label.grid(row=2, column=0, columnspan=2, pady=(0,5), sticky="new", padx=2)

        stats_grid = ttk.Frame(status_vis_frame, style="Content.TFrame")
        stats_grid.grid(row=3, column=0, columnspan=2, sticky="new", pady=(0,5))
        stats_grid.columnconfigure(0, weight=0, minsize=140); stats_grid.columnconfigure(1, weight=1)
        row_idx_stats = 0
        stats_to_display = [("Move Score:", self.ai_eval_var), ("Time Taken (s):", self.time_taken_var),
            ("Nodes Explored:", self.nodes_explored_var), ("Nodes/Sec (NPS):", self.nps_var),
            ("Max Search Depth (ply):", self.actual_depth_var), ("IDDFS Target (ply):", self.target_depth_var)]
        for label_text, var in stats_to_display:
            ttk.Label(stats_grid, text=label_text, style="Info.TLabel").grid(row=row_idx_stats, column=0, sticky="w", padx=(2,5), pady=1)
            ttk.Label(stats_grid, textvariable=var, style="InfoValue.TLabel").grid(row=row_idx_stats, column=1, sticky="w", padx=2, pady=1)
            row_idx_stats += 1
        ttk.Label(stats_grid, text="(Scr: >0 X, <0 O, 0 Draw)", style="Info.TLabel", foreground="#6c757d").grid(row=row_idx_stats, column=0, columnspan=2, sticky="w", padx=2, pady=(1,3))
        
        self.hint_suggestion_label = ttk.Label(status_vis_frame, textvariable=self.hint_suggestion_var, style="HintSuggest.TLabel", wraplength=280)
        self.hint_suggestion_label.grid(row=4, column=0, columnspan=2, pady=3, sticky="new")
        self.top_moves_label = ttk.Label(status_vis_frame, textvariable=self.top_moves_var, style="MoveInfo.TLabel", anchor="nw", wraplength=280)
        self.top_moves_label.grid(row=5, column=0, columnspan=2, pady=(3,0), sticky="nsew")

        self.board_outer_frame = ttk.Frame(self.paned_window, style="TFrame", padding=10)
        self.paned_window.add(self.board_outer_frame, weight=2)
        self.board_outer_frame.grid_rowconfigure(0, weight=1); self.board_outer_frame.grid_columnconfigure(0, weight=1)
        self.board_frame = ttk.Frame(self.board_outer_frame, style="TFrame")
        self.board_frame.grid(row=0, column=0, sticky="")
        self.status_var.set("Adjust N, K, then Start.")

    def on_silent_close(self, event=None): self.root.destroy()

    def start_new_game(self, human_starts=True):
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showwarning("Busy", "AI calculation is currently in progress."); return
        try:
            n_val = int(self.size_entry.get()); k_val = int(self.k_entry.get())
            if not (2 <= n_val <= 7): messagebox.showerror("Error", "Board size N must be between 2 and 7."); return
            if not (2 <= k_val <= n_val): messagebox.showerror("Error", f"K-in-a-row must be between 2 and N (currently {n_val})."); return
            self.board_size = n_val; self.k_to_win = k_val
            init_zobrist_for_board_size(self.board_size)
            compute_zobrist_hash_from_canonical_tuple.cache_clear()
        except ValueError: messagebox.showerror("Error", "Invalid input for N or K."); return

        get_canonical_board_tuple.cache_clear(); _get_immediate_winning_moves.cache_clear()
        can_force_win_on_next_player_turn.cache_clear(); is_line_still_possible.cache_clear()
        can_player_still_win.cache_clear(); is_unwinnable_for_either.cache_clear()

        self.game_board = [[EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = PLAYER_HUMAN if human_starts else PLAYER_AI
        self.game_over = False; self.hint_suggestion_var.set("")
        self.status_var.set(f"{self.board_size}x{self.board_size}, {self.k_to_win}-in-a-row. {'Your turn (X).' if human_starts else 'AI`s turn (O).'}")
        self.ai_eval_var.set("N/A"); self.nodes_explored_var.set("0"); self.actual_depth_var.set("0")
        self.target_depth_var.set("0"); self.time_taken_var.set("N/A"); self.nps_var.set("N/A")
        self.progress_percent_var.set("Calculating..."); self.detailed_progress_var.set("")
        self.top_moves_var.set("Top Move Considerations:\n(After calculation)")
        if self.progress_bar.master.winfo_ismapped(): self.progress_bar.master.grid_remove()
        if self.detailed_progress_label.winfo_ismapped(): self.detailed_progress_label.grid_remove()
        self.create_board_ui_buttons(); self.update_hint_button_state()
        if not human_starts: self.root.after(100, lambda: self.trigger_ai_or_hint_calculation(PLAYER_AI))

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
        self.status_var.set("Calculating hint for Human (X)..." if player_to_calculate_for == PLAYER_HUMAN else "AI (O) is thinking...")
        self.top_moves_var.set("Top Move Considerations:\nCalculating...")
        self.disable_board_buttons(); self.hint_button.config(state=tk.DISABLED)
        self.progress_bar.master.grid(); self.detailed_progress_label.grid()
        self.progress_bar["value"] = 0; self.progress_percent_var.set("Starting...")
        self.detailed_progress_var.set("Initializing search...")
        current_board_depth = self._get_current_game_depth_for_ui()
        self.target_depth_var.set(str(max(0, (self.board_size**2) - current_board_depth)))
        self.root.config(cursor="watch"); self.root.update_idletasks()
        while not self.progress_queue.empty(): self.progress_queue.get_nowait()
        while not self.result_queue.empty(): self.result_queue.get_nowait()
        self.calculation_manager_thread = threading.Thread(target=find_best_move_iterative_deepening,
            args=(self.game_board, self.k_to_win, self.progress_queue, self.result_queue, player_to_calculate_for), daemon=True)
        self.ai_start_time = time.monotonic()
        self._current_calculation_for_player = player_to_calculate_for
        self.calculation_manager_thread.start(); self.check_ai_progress()

    def create_board_ui_buttons(self, attempt=1):
        for widget in self.board_frame.winfo_children(): widget.destroy()
        self.buttons = []
        self.root.update_idletasks()
        container_width = self.board_outer_frame.winfo_width(); container_height = self.board_outer_frame.winfo_height()
        if (container_width < 50 or container_height < 50) and attempt < 10:
            self.root.after(100, lambda: self.create_board_ui_buttons(attempt + 1)); return
        if container_width < 50 or container_height < 50:
            min_dim_root = min(self.root.winfo_width(), self.root.winfo_height()) * 0.6
            container_width = max(min_dim_root, 200); container_height = max(min_dim_root, 200)
        board_dimension = max(self.board_size * 35, min(container_width, container_height) - 10)
        cell_dim = board_dimension // self.board_size
        self.board_frame.config(width=board_dimension, height=board_dimension)
        btn_font_size = max(10, int(cell_dim * 0.45))
        board_button_font = tkfont.Font(family=self.default_font_family, size=btn_font_size, weight="bold")
        for i in range(self.board_size):
            self.board_frame.grid_rowconfigure(i, weight=1, minsize=cell_dim)
            self.board_frame.grid_columnconfigure(i, weight=1, minsize=cell_dim)
        for r in range(self.board_size):
            row_buttons = []
            for c_idx in range(self.board_size):
                button = tk.Button(self.board_frame, text=EMPTY_CELL, font=board_button_font, relief="flat", borderwidth=1,
                                   bg=COLOR_FRAME_BG, activebackground="#DDDDDD", fg=COLOR_TEXT_PRIMARY,
                                   command=lambda r_idx=r, c_idx_btn=c_idx: self.handle_cell_click(r_idx, c_idx_btn))
                button.grid(row=r, column=c_idx, sticky="nsew", padx=1, pady=1); row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board_button_states()

    def handle_cell_click(self, r, c):
        if self.game_over or self.game_board[r][c] != EMPTY_CELL or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            messagebox.showinfo("Wait", "AI calculation is currently in progress."); return
        self.hint_suggestion_var.set(""); self.clear_all_hint_highlights()
        self.make_move(r, c, PLAYER_HUMAN)
        if self.check_game_status(): return
        self.current_player = PLAYER_AI; self.update_hint_button_state()
        self.trigger_ai_or_hint_calculation(PLAYER_AI)

    def _get_current_game_depth_for_ui(self):
        depth = 0
        if hasattr(self, 'game_board') and self.game_board and hasattr(self, 'board_size') and self.board_size > 0 and len(self.game_board) == self.board_size:
            for r_idx in range(self.board_size):
                if len(self.game_board[r_idx]) == self.board_size:
                    for c_idx in range(self.board_size):
                        if self.game_board[r_idx][c_idx] != EMPTY_CELL: depth += 1
                else: return sum(sum(1 for cell in row if cell != EMPTY_CELL) for row in self.game_board)
        else: return 0
        return depth

    def check_ai_progress(self):
        update_interval_ms = 100; current_runtime = 0.0
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            current_runtime = time.monotonic() - self.ai_start_time
            self.time_taken_var.set(f"{current_runtime:.2f}")
        try:
            progress_update = self.progress_queue.get_nowait()
            current_nodes = progress_update.get('current_nodes')
            if current_nodes is not None:
                self.nodes_explored_var.set(f"{current_nodes:,}")
                self.nps_var.set(f"{(current_nodes / current_runtime):,.0f}" if current_runtime > 0.01 and current_nodes > 0 else ("High" if current_nodes > 0 else "..."))
            
            current_depth_iter = progress_update.get('current_depth_iter', 0)
            max_plies_target = max(1, progress_update.get('max_total_depth_iters', (self.board_size**2) - self._get_current_game_depth_for_ui()))
            root_done = progress_update.get('root_moves_done_this_iter', 0)
            total_root = max(1, progress_update.get('total_root_moves_this_iter', 1))
            
            frac_current_iter = root_done / total_root if progress_update.get('type') == 'progress' else 0
            eff_completed_depth = (current_depth_iter - 1) + frac_current_iter
            overall_perc = min(100.0, max(0.0, (eff_completed_depth / max_plies_target) * 100))

            if progress_update.get('type') == 'start_iter':
                self.progress_bar["maximum"] = total_root; self.progress_bar["value"] = 0
                self.detailed_progress_var.set(f"Depth {current_depth_iter} (Move 0/{total_root})")
            elif progress_update.get('type') == 'progress':
                if 'current_max_depth' in progress_update: self.actual_depth_var.set(f"{progress_update['current_max_depth']}")
                score_val = progress_update.get('current_best_score')
                if score_val is not None : self.ai_eval_var.set(f"{score_val:.0f}" if isinstance(score_val, (int, float)) and score_val not in (math.inf, -math.inf) else str(score_val))
                self.progress_bar["value"] = root_done; self.progress_bar["maximum"] = total_root
                self.detailed_progress_var.set(f"Depth {current_depth_iter} (Move {root_done}/{total_root})")
            self.progress_percent_var.set(f"Target {max_plies_target} plies ({overall_perc:.0f}%)")
        except thread_queue.Empty: pass

        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive():
            self.root.after(update_interval_ms, self.check_ai_progress)
        elif self.calculation_manager_thread:
            final_runtime = time.monotonic() - self.ai_start_time; self.time_taken_var.set(f"{final_runtime:.3f}")
            try: # Try to get a very last update
                last_prog = self.progress_queue.get_nowait()
                last_nodes = last_prog.get('current_nodes')
                if last_nodes is not None:
                    self.nodes_explored_var.set(f"{last_nodes:,}")
                    self.nps_var.set(f"{(last_nodes / final_runtime):,.0f}" if final_runtime > 0.01 and last_nodes > 0 else ("High" if last_nodes > 0 else "0"))
                if 'current_max_depth' in last_prog: self.actual_depth_var.set(f"{last_prog['current_max_depth']}")
                score_val = last_prog.get('current_best_score')
                if score_val is not None : self.ai_eval_var.set(f"{score_val:.0f}" if isinstance(score_val, (int, float)) and score_val not in (math.inf, -math.inf) else str(score_val))
            except thread_queue.Empty: # Update with existing values if no last message
                nodes_s = self.nodes_explored_var.get().replace(',', ''); nodes_v = int(nodes_s) if nodes_s.isdigit() else 0
                self.nps_var.set(f"{(nodes_v / final_runtime):,.0f}" if final_runtime > 0.01 and nodes_v > 0 else ("High" if nodes_v > 0 else "N/A"))
            self.handle_calculation_result(); self.calculation_manager_thread = None
            self.progress_percent_var.set(f"Target {self.target_depth_var.get()} plies (100%)" if self.target_depth_var.get() != "0" else "Done.")
            self.detailed_progress_var.set("Search complete.")

    def handle_calculation_result(self):
        self.root.config(cursor="");
        if self.progress_bar.master.winfo_ismapped(): self.progress_bar.master.grid_remove()
        if self.detailed_progress_label.winfo_ismapped(): self.detailed_progress_label.grid_remove()
        time_taken_calc = time.monotonic() - self.ai_start_time; self.time_taken_var.set(f"{time_taken_calc:.3f}")
        was_hint = (self._current_calculation_for_player == PLAYER_HUMAN)
        try:
            res = self.result_queue.get_nowait()
            move, score = res['best_move_data']; top_moves = res['top_moves_list']
            nodes = res['total_nodes']; max_depth = res['max_search_depth']
        except thread_queue.Empty:
            self.status_var.set("Error: No result from AI."); self.enable_board_buttons(); self.update_hint_button_state(); return
        
        self.nodes_explored_var.set(f"{nodes:,}")
        self.nps_var.set(f"{(nodes / time_taken_calc):,.0f}" if time_taken_calc > 1e-4 else "High")
        self.actual_depth_var.set(f"{max_depth}")
        self.ai_eval_var.set(f"{score:.0f}" if score not in (math.inf, -math.inf) else "Win/Loss")
        move_alg = to_algebraic(move[0], move[1], self.board_size) if move else "N/A"
        
        top_text = f"Top {'Human (X)' if was_hint else 'AI (O)'} Moves (Mv: Scr | N | ID | AD):\n"
        if top_moves:
            for item in top_moves:
                m_alg = to_algebraic(item['move'][0], item['move'][1], self.board_size)
                s_str = f"{item['score']:.0f}"; n_str = f"N:{item.get('nodes',0):,}"
                id_str = f"ID:{item.get('depth',0)}"; ad_str = f"AD:{item.get('actual_eval_depth',0)}"
                top_text += f"  {m_alg:<4}: {s_str:<5} | {n_str:<10}| {id_str:<5}| {ad_str}\n"
        else: top_text += "  (N/A)\n"
        self.top_moves_var.set(top_text.strip())

        if was_hint:
            if move:
                self.status_var.set(f"Hint for X: Best is {move_alg} (Score: {score:.0f}). Your turn.")
                self.hint_suggestion_var.set(f"Suggested for X: {move_alg} (Score: {score:.0f})")
                if self.game_board[move[0]][move[1]] == EMPTY_CELL and self.buttons[move[0]][move[1]]['state']!=tk.DISABLED:
                    self.buttons[move[0]][move[1]].config(bg=COLOR_HINT_SUGGEST_BG, relief="raised")
            else: self.status_var.set("Hint: No moves or game ended."); self.hint_suggestion_var.set("No suggestion.")
        else: # AI's turn
            self.hint_suggestion_var.set(""); self.clear_all_hint_highlights()
            if move:
                self.make_move(move[0], move[1], PLAYER_AI)
                if self.check_game_status(): self.enable_board_buttons(); self.update_hint_button_state(); return
                self.current_player = PLAYER_HUMAN
                self.status_var.set(f"AI (O) moved to {move_alg}. Your turn (X).")
            else:
                self.status_var.set("It's a Draw!" if is_board_full(self.game_board) else "AI error/no moves. Game Over?")
                self.game_over = True
        self.enable_board_buttons(); self.update_hint_button_state()

    def clear_all_hint_highlights(self):
        if not self.buttons: return
        for r in range(self.board_size):
            if r < len(self.buttons) and self.buttons[r]:
                for c in range(self.board_size):
                    if c < len(self.buttons[r]) and self.buttons[r][c] and self.game_board[r][c] == EMPTY_CELL and \
                       self.buttons[r][c].cget('bg') == COLOR_HINT_SUGGEST_BG:
                        self.buttons[r][c].config(bg=COLOR_FRAME_BG, relief="flat")

    def make_move(self, r, c, player):
        self.game_board[r][c] = player; btn = self.buttons[r][c]
        btn.config(text=player, state=tk.DISABLED, relief=tk.SUNKEN,
                   disabledforeground=(COLOR_ACCENT_SECONDARY if player == PLAYER_HUMAN else COLOR_DANGER),
                   background=(COLOR_HUMAN_MOVE_BG if player == PLAYER_HUMAN else COLOR_AI_MOVE_BG))

    def check_game_status(self):
        human_line = get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win)
        ai_line = get_winning_line(self.game_board, PLAYER_AI, self.k_to_win)
        full = is_board_full(self.game_board)
        over_changed = False
        if not self.game_over:
            if human_line: self.status_var.set("You (X) Win!"); self.game_over=True; over_changed=True; self.highlight_winning_line(human_line)
            elif ai_line: self.status_var.set("AI (O) Wins!"); self.game_over=True; over_changed=True; self.highlight_winning_line(ai_line)
            elif full: self.status_var.set("It's a Draw!"); self.game_over=True; over_changed=True
        if over_changed or self.game_over: self.disable_board_buttons(); self.update_hint_button_state()
        return self.game_over

    def update_board_button_states(self): # Simplified
        if not self.buttons: return
        for r, row_btns in enumerate(self.buttons):
            for c, button in enumerate(row_btns):
                if not button: continue
                text = self.game_board[r][c]
                is_hinted = button.cget('bg') == COLOR_HINT_SUGGEST_BG
                if text == EMPTY_CELL:
                    button.config(text=text, state=tk.NORMAL, fg=COLOR_TEXT_PRIMARY,
                                  bg=COLOR_HINT_SUGGEST_BG if is_hinted else COLOR_FRAME_BG,
                                  relief="raised" if is_hinted else "flat")
                else:
                    button.config(text=text, state=tk.DISABLED, relief="sunken",
                                  disabledforeground=(COLOR_ACCENT_SECONDARY if text == PLAYER_HUMAN else COLOR_DANGER),
                                  background=(COLOR_HUMAN_MOVE_BG if text == PLAYER_HUMAN else COLOR_AI_MOVE_BG))

    def disable_board_buttons(self):
        if not self.buttons: return
        for row_btns in self.buttons:
            for button in row_btns:
                if button and button['state'] == tk.NORMAL: button.config(state=tk.DISABLED)

    def enable_board_buttons(self):
        if self.game_over or (self.calculation_manager_thread and self.calculation_manager_thread.is_alive()) or not self.buttons: return
        self.update_board_button_states() 

    def highlight_winning_line(self, winning_cells):
        if not self.buttons: return
        for r,c in winning_cells:
             if r < len(self.buttons) and self.buttons[r] and c < len(self.buttons[r]) and self.buttons[r][c]:
                self.buttons[r][c].config(background=COLOR_WIN_HIGHLIGHT, relief=tk.GROOVE)

# --- Main Execution ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = TicTacToeGUI(main_root)
    main_root.mainloop()