import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import math
import functools
import time
import threading
import queue as thread_queue
import random

# --- Constants ---
PLAYER_HUMAN = 'X' # Corresponds to player_id 0 for bitboards
PLAYER_AI = 'O'    # Corresponds to player_id 1 for bitboards
EMPTY_CELL = ' '
PLAYER_IDS = {PLAYER_HUMAN: 0, PLAYER_AI: 1}

COLOR_ROOT_BG = "#EAEAEA" # UI Colors (unchanged)
COLOR_FRAME_BG = "#FFFFFF"
COLOR_TEXT_PRIMARY = "#212529"
COLOR_TEXT_SECONDARY = "#495057"
COLOR_ACCENT_PRIMARY = "#007BFF"
COLOR_ACCENT_SECONDARY = "#0056b3"
COLOR_HINT = "#FFC107"
COLOR_HINT_BG = "#FFC107"
COLOR_HINT_TEXT = "#212529"
COLOR_HINT_SUGGEST_BG = COLOR_HINT_BG
COLOR_SUCCESS = "#28A745"
COLOR_DANGER = "#DC3545"
COLOR_HUMAN_MOVE_BG = "#E0EFFF"
COLOR_AI_MOVE_BG = "#FFE0E0"
COLOR_WIN_HIGHLIGHT = "#B8F5B8"
COLOR_TEXT_ON_ACCENT = "#FFFFFF"
COLOR_HINT_PROGRESS_BAR = COLOR_HINT

# --- Transposition Table Constants ---
TT_EXACT = 0
TT_LOWER_BOUND = 1
TT_UPPER_BOUND = 2

# --- Global Transposition Table (for AI search) ---
transposition_table = {}
tt_hits = 0
tt_stores = 0

# --- Bitboard Zobrist Hashing ---
ZOBRIST_BITBOARD_TABLE = {}

# --- Bitboard Winning Masks Cache ---
WINNING_MASKS_CACHE = {}
POTENTIAL_LINE_MASKS_CACHE = {}

# --- Coordinate Conversion ---
def to_algebraic(row, col, board_size):
    if not (0 <= row < board_size and 0 <= col < board_size): return "Invalid"
    file_char = chr(ord('a') + col)
    rank_char = str(board_size - row)
    return f"{file_char}{rank_char}"

# --- Bitboard Utilities ---
@functools.lru_cache(maxsize=128)
def cell_to_bit_index(r, c, board_size):
    return r * board_size + c

@functools.lru_cache(maxsize=128)
def bit_index_to_cell(bit_index, board_size):
    return divmod(bit_index, board_size)

def list_to_bitboards(list_board, board_size):
    player_x_bb = 0
    player_o_bb = 0
    for r in range(board_size):
        for c in range(board_size):
            bit_idx = cell_to_bit_index(r, c, board_size)
            if list_board[r][c] == PLAYER_HUMAN:
                player_x_bb |= (1 << bit_idx)
            elif list_board[r][c] == PLAYER_AI:
                player_o_bb |= (1 << bit_idx)
    return player_x_bb, player_o_bb

def init_zobrist_bitboard(board_size):
    global ZOBRIST_BITBOARD_TABLE
    ZOBRIST_BITBOARD_TABLE.clear()
    random.seed(42)
    num_cells = board_size * board_size
    for i in range(num_cells):
        ZOBRIST_BITBOARD_TABLE[(i, PLAYER_IDS[PLAYER_HUMAN])] = random.getrandbits(64)
        ZOBRIST_BITBOARD_TABLE[(i, PLAYER_IDS[PLAYER_AI])] = random.getrandbits(64)

# --- Winning Mask Generation for Bitboards ---
def _generate_masks_for_k(board_size, k_to_win):
    masks = []
    # Horizontal
    for r in range(board_size):
        for c in range(board_size - k_to_win + 1):
            mask = 0
            for i in range(k_to_win):
                mask |= (1 << cell_to_bit_index(r, c + i, board_size))
            masks.append(mask)
    # Vertical
    for c in range(board_size):
        for r in range(board_size - k_to_win + 1):
            mask = 0
            for i in range(k_to_win):
                mask |= (1 << cell_to_bit_index(r + i, c, board_size))
            masks.append(mask)
    # Diagonal \
    for r in range(board_size - k_to_win + 1):
        for c in range(board_size - k_to_win + 1):
            mask = 0
            for i in range(k_to_win):
                mask |= (1 << cell_to_bit_index(r + i, c + i, board_size))
            masks.append(mask)
    # Diagonal /
    for r in range(board_size - k_to_win + 1):
        for c in range(k_to_win - 1, board_size):
            mask = 0
            for i in range(k_to_win):
                mask |= (1 << cell_to_bit_index(r + i, c - i, board_size))
            masks.append(mask)
    return list(set(masks))

def get_win_masks(board_size, k_to_win):
    if (board_size, k_to_win) not in WINNING_MASKS_CACHE:
        WINNING_MASKS_CACHE[(board_size, k_to_win)] = _generate_masks_for_k(board_size, k_to_win)
    return WINNING_MASKS_CACHE[(board_size, k_to_win)]

def get_potential_line_masks(board_size, k_to_win):
    if (board_size, k_to_win) not in POTENTIAL_LINE_MASKS_CACHE:
        POTENTIAL_LINE_MASKS_CACHE[(board_size, k_to_win)] = _generate_masks_for_k(board_size, k_to_win)
    return POTENTIAL_LINE_MASKS_CACHE[(board_size, k_to_win)]


# --- Game Logic (Core - GUI related or Bitboard based) ---
def get_winning_line(board, player, k_to_win):
    n = len(board)
    # Horizontal
    for r_idx in range(n):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx][c_idx+i] == player for i in range(k_to_win)):
                return [(r_idx, c_idx+i) for i in range(k_to_win)]
    # Vertical
    for c_idx in range(n):
        for r_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx] == player for i in range(k_to_win)): return [(r_idx+i, c_idx) for i in range(k_to_win)]
    # Diagonal (top-left to bottom-right)
    for r_idx in range(n - k_to_win + 1):
        for c_idx in range(n - k_to_win + 1):
            if all(board[r_idx+i][c_idx+i] == player for i in range(k_to_win)): return [(r_idx+i, c_idx+i) for i in range(k_to_win)]
    # Diagonal (top-right to bottom-left)
    for r_idx in range(n - k_to_win + 1):
        for c_idx in range(k_to_win - 1, n):
            if all(board[r_idx+i][c_idx-i] == player for i in range(k_to_win)): return [(r_idx+i, c_idx-i) for i in range(k_to_win)]
    return []

@functools.lru_cache(maxsize=16384)
def check_win_bb(player_bb, board_size, k_to_win):
    win_masks = get_win_masks(board_size, k_to_win)
    for mask in win_masks:
        if (player_bb & mask) == mask:
            return True
    return False

@functools.lru_cache(maxsize=8192)
def is_board_full_bb(player_x_bb, player_o_bb, board_size):
    num_cells = board_size * board_size
    full_mask = (1 << num_cells) - 1
    return (player_x_bb | player_o_bb) == full_mask

# --- Zobrist Hash for Bitboards (Canonical) ---
@functools.lru_cache(maxsize=32768)
def compute_zobrist_hash_from_canonical_bbs(canonical_x_bb, canonical_o_bb):
    h = 0
    bb = canonical_x_bb; idx = 0
    while bb > 0:
        if bb & 1: h ^= ZOBRIST_BITBOARD_TABLE[(idx, PLAYER_IDS[PLAYER_HUMAN])]
        bb >>= 1; idx += 1
    bb = canonical_o_bb; idx = 0
    while bb > 0:
        if bb & 1: h ^= ZOBRIST_BITBOARD_TABLE[(idx, PLAYER_IDS[PLAYER_AI])]
        bb >>= 1; idx += 1
    return h

# --- Pure Bitboard Transformations ---
def _rotate_bitboard_90_cw(bb, board_size):
    new_bb = 0
    temp_bb = bb
    while temp_bb > 0:
        lsb_mask = temp_bb & -temp_bb
        idx = lsb_mask.bit_length() - 1
        r, c = bit_index_to_cell(idx, board_size)
        new_idx = cell_to_bit_index(c, board_size - 1 - r, board_size)
        new_bb |= (1 << new_idx)
        temp_bb ^= lsb_mask
    return new_bb

def _reflect_bitboard_horizontal(bb, board_size):
    new_bb = 0
    temp_bb = bb
    while temp_bb > 0:
        lsb_mask = temp_bb & -temp_bb
        idx = lsb_mask.bit_length() - 1
        r, c = bit_index_to_cell(idx, board_size)
        new_idx = cell_to_bit_index(r, board_size - 1 - c, board_size)
        new_bb |= (1 << new_idx)
        temp_bb ^= lsb_mask
    return new_bb

# --- Symmetry for Bitboards (using pure bitboard ops) ---
@functools.lru_cache(maxsize=8192)
def get_canonical_bitboards(player_x_bb, player_o_bb, board_size):
    symmetries = []
    current_x_bb = player_x_bb
    current_o_bb = player_o_bb
    for _ in range(4):
        symmetries.append((current_x_bb, current_o_bb))
        reflected_x_bb = _reflect_bitboard_horizontal(current_x_bb, board_size)
        reflected_o_bb = _reflect_bitboard_horizontal(current_o_bb, board_size)
        symmetries.append((reflected_x_bb, reflected_o_bb))
        if _ < 3:
            current_x_bb = _rotate_bitboard_90_cw(current_x_bb, board_size)
            current_o_bb = _rotate_bitboard_90_cw(current_o_bb, board_size)
    return min(symmetries)

# --- AI Helper Functions (Bitboard based) ---
def get_available_moves_bb(player_x_bb, player_o_bb, board_size, center_sort=True):
    occupied_bb = player_x_bb | player_o_bb; num_cells = board_size * board_size
    available_moves_indices = []
    for bit_idx in range(num_cells):
        if not (occupied_bb & (1 << bit_idx)): available_moves_indices.append(bit_idx)
    if center_sort:
        center_r, center_c = (board_size - 1) / 2.0, (board_size - 1) / 2.0
        def sort_key(bit_idx): r, c = bit_index_to_cell(bit_idx, board_size); return (abs(r - center_r) + abs(c - center_c), r, c)
        available_moves_indices.sort(key=sort_key)
    return available_moves_indices

# --- OPTIMIZED _get_immediate_winning_moves_bb ---
@functools.lru_cache(maxsize=16384)
def _get_immediate_winning_moves_bb(player_x_bb, player_o_bb, current_player_id, k_to_win, board_size):
    winning_moves_indices = []
    
    current_player_bb = player_x_bb if current_player_id == PLAYER_IDS[PLAYER_HUMAN] else player_o_bb
    opponent_player_bb = player_o_bb if current_player_id == PLAYER_IDS[PLAYER_HUMAN] else player_x_bb
    
    occupied_bb = player_x_bb | player_o_bb
    win_masks = get_win_masks(board_size, k_to_win) # Cached call

    for mask in win_masks:
        # 1. Check if the opponent already blocks this potential winning line
        if (mask & opponent_player_bb) != 0:
            continue 

        # 2. Find which pieces the current player still needs to place in this line to win
        #    These are bits in the win_mask that are not yet set in current_player_bb.
        needed_for_win_mask = mask & ~current_player_bb
        
        # 3. Check if exactly one piece is needed to complete the line
        #    (i.e., needed_for_win_mask has only one bit set - it's a power of 2)
        if needed_for_win_mask != 0 and (needed_for_win_mask & (needed_for_win_mask - 1)) == 0:
            move_bit = needed_for_win_mask # This is the single bit (piece) needed
            
            # 4. Crucially, check if the cell for this 'move_bit' is actually empty
            if (move_bit & occupied_bb) == 0: 
                move_idx = move_bit.bit_length() - 1 # Get the 0-based index of the bit
                # Avoid adding duplicates if a rare move completes multiple lines
                if move_idx not in winning_moves_indices: 
                    winning_moves_indices.append(move_idx)
                    
    return winning_moves_indices

@functools.lru_cache(maxsize=4096)
def is_unwinnable_for_either_bb(player_x_bb, player_o_bb, k_to_win, board_size):
    potential_lines = get_potential_line_masks(board_size, k_to_win)
    human_can_win = any((line_mask & player_o_bb) == 0 for line_mask in potential_lines)
    ai_can_win = any((line_mask & player_x_bb) == 0 for line_mask in potential_lines)
    return not human_can_win and not ai_can_win

@functools.lru_cache(maxsize=32768)
def can_force_win_on_next_player_turn_bb(player_x_bb, player_o_bb, current_player_id, k_to_win, board_size):
    opponent_player_id = 1 - current_player_id
    original_player_moves = get_available_moves_bb(player_x_bb, player_o_bb, board_size, center_sort=False)
    for m1_idx in original_player_moves:
        m1_bit = 1 << m1_idx
        next_x_bb_after_m1, next_o_bb_after_m1 = player_x_bb, player_o_bb
        if current_player_id == PLAYER_IDS[PLAYER_HUMAN]: next_x_bb_after_m1 |= m1_bit
        else: next_o_bb_after_m1 |= m1_bit
        current_player_bb_after_m1 = next_x_bb_after_m1 if current_player_id == PLAYER_IDS[PLAYER_HUMAN] else next_o_bb_after_m1
        if check_win_bb(current_player_bb_after_m1, board_size, k_to_win): continue # Current player wins with m1, not a "next turn" force
        
        opponent_moves_after_m1 = get_available_moves_bb(next_x_bb_after_m1, next_o_bb_after_m1, board_size, center_sort=False)
        if not opponent_moves_after_m1: # Opponent has no moves
            # If current player has an immediate win after m1 (e.g. m1 filled the board, making a line)
            if _get_immediate_winning_moves_bb(next_x_bb_after_m1, next_o_bb_after_m1, current_player_id, k_to_win, board_size): 
                return True # m1 forces a win as opponent has no response and current player can win
            else: 
                continue # Opponent has no moves, current player cannot win - m1 doesn't force
        
        m1_is_forcing = True
        for mopp_idx in opponent_moves_after_m1:
            mopp_bit = 1 << mopp_idx
            next_x_bb_after_mopp, next_o_bb_after_mopp = next_x_bb_after_m1, next_o_bb_after_m1
            if opponent_player_id == PLAYER_IDS[PLAYER_HUMAN]: next_x_bb_after_mopp |= mopp_bit
            else: next_o_bb_after_mopp |= mopp_bit
            
            opponent_bb_after_mopp = next_x_bb_after_mopp if opponent_player_id == PLAYER_IDS[PLAYER_HUMAN] else next_o_bb_after_mopp
            if check_win_bb(opponent_bb_after_mopp, board_size, k_to_win): # If opponent wins with their move
                m1_is_forcing = False; break # Then m1 was not a forcing move
            
            # If opponent doesn't win, current player must have an immediate win
            if not _get_immediate_winning_moves_bb(next_x_bb_after_mopp, next_o_bb_after_mopp, current_player_id, k_to_win, board_size):
                m1_is_forcing = False; break # Current player cannot immediately win, so m1 not forcing
        
        if m1_is_forcing: return True # m1 is forcing if all opponent replies lead to current player's win
    return False

# --- Minimax AI (Bitboard based) ---
def _minimax_recursive_logic_bb(canonical_x_bb, canonical_o_bb, depth, is_maximizing_for_x, k_to_win, board_size, alpha, beta, current_max_search_depth_for_this_call, current_zobrist_hash):
    global tt_hits, tt_stores
    nodes_here = 1; max_depth_here = depth; original_alpha = alpha
    remaining_depth_tt = current_max_search_depth_for_this_call - depth
    
    # Transposition Table Lookup
    if current_zobrist_hash in transposition_table:
        entry = transposition_table[current_zobrist_hash]
        if entry['depth_searched'] >= remaining_depth_tt:
            tt_hits += 1
            if entry['flag'] == TT_EXACT: return (entry['score'], nodes_here, max_depth_here)
            elif entry['flag'] == TT_LOWER_BOUND: alpha = max(alpha, entry['score'])
            elif entry['flag'] == TT_UPPER_BOUND: beta = min(beta, entry['score'])
            if alpha >= beta: return (entry['score'], nodes_here, max_depth_here)
            
    # Terminal State Checks
    score_terminal = None
    if check_win_bb(canonical_x_bb, board_size, k_to_win): score_terminal = 1000 - depth
    elif check_win_bb(canonical_o_bb, board_size, k_to_win): score_terminal = -1000 + depth
    elif is_board_full_bb(canonical_x_bb, canonical_o_bb, board_size): score_terminal = 0
    elif is_unwinnable_for_either_bb(canonical_x_bb, canonical_o_bb, k_to_win, board_size): score_terminal = 0
    
    if score_terminal is not None:
        transposition_table[current_zobrist_hash] = {'score': score_terminal, 'depth_searched': 99, 'flag': TT_EXACT, 'best_move_idx': None} # High depth for terminal
        tt_stores +=1; return (score_terminal, nodes_here, max_depth_here)
        
    # Max Depth Reached (Quiescence or limit)
    if depth >= current_max_search_depth_for_this_call: return (0, nodes_here, max_depth_here) # Neutral score at depth limit
    
    current_player_id = PLAYER_IDS[PLAYER_HUMAN] if is_maximizing_for_x else PLAYER_IDS[PLAYER_AI]
    
    # Check for Forced Win (2-ply tactical sequence)
    # Only check if enough depth is remaining to "see" the 2-ply sequence + current ply.
    if (depth + 3) <= current_max_search_depth_for_this_call: # (current_depth + P_move + O_move + P_win_move)
        score_forced = 0; is_forced = False
        if can_force_win_on_next_player_turn_bb(canonical_x_bb, canonical_o_bb, current_player_id, k_to_win, board_size):
            # Score reflects win depth: current ply + 1 (player's move) + 1 (opponent's forced reply)
            score_forced = (1000 - (depth + 2)) if is_maximizing_for_x else (-1000 + (depth + 2))
            is_forced = True
        if is_forced:
            transposition_table[current_zobrist_hash] = {'score': score_forced, 'depth_searched': 99, 'flag': TT_EXACT, 'best_move_idx': None}
            tt_stores +=1; return (score_forced, nodes_here, max_depth_here)
            
    available_moves_indices = get_available_moves_bb(canonical_x_bb, canonical_o_bb, board_size, center_sort=True)
    if not available_moves_indices: return (0, nodes_here, max_depth_here) # Draw if no moves but not terminal
    
    # Move Ordering: Use TT's best move first
    if current_zobrist_hash in transposition_table: # Re-check as alpha/beta might have changed
        entry = transposition_table[current_zobrist_hash]
        if entry.get('best_move_idx') is not None:
            best_move_idx_tt = entry['best_move_idx']
            if best_move_idx_tt in available_moves_indices: 
                available_moves_indices.remove(best_move_idx_tt)
                available_moves_indices.insert(0, best_move_idx_tt)
                
    best_move_idx_this_node = None; eval_to_store = 0
    
    # Principal Variation Search (PVS) / NegaScout
    if is_maximizing_for_x:
        max_eval = -math.inf
        for i, move_idx in enumerate(available_moves_indices):
            move_bit = 1 << move_idx; next_x_bb = canonical_x_bb | move_bit; next_o_bb = canonical_o_bb
            child_canonical_x_bb, child_canonical_o_bb = get_canonical_bitboards(next_x_bb, next_o_bb, board_size)
            child_hash = compute_zobrist_hash_from_canonical_bbs(child_canonical_x_bb, child_canonical_o_bb)
            
            current_eval_score = 0; child_nodes_iter = 0; child_max_depth_iter = 0
            if i == 0: # Full window search for the first move
                current_eval_score, child_nodes_iter, child_max_depth_iter = _minimax_recursive_logic_bb(child_canonical_x_bb, child_canonical_o_bb, depth + 1, False, k_to_win, board_size, alpha, beta, current_max_search_depth_for_this_call, child_hash)
            else: # Null window (scout) search for subsequent moves
                # Search with a tight (null) window around alpha
                null_window_score, child_nodes_null, child_max_depth_null = _minimax_recursive_logic_bb(child_canonical_x_bb, child_canonical_o_bb, depth + 1, False, k_to_win, board_size, alpha, alpha + 1, current_max_search_depth_for_this_call, child_hash)
                child_nodes_iter += child_nodes_null; child_max_depth_iter = max(child_max_depth_iter, child_max_depth_null)
                
                # If scout search failed high (score > alpha) and potentially beats beta, re-search with full window
                if null_window_score > alpha and null_window_score < beta: 
                    current_eval_score, child_nodes_re, child_max_depth_re = _minimax_recursive_logic_bb(child_canonical_x_bb, child_canonical_o_bb, depth + 1, False, k_to_win, board_size, alpha, beta, current_max_search_depth_for_this_call, child_hash)
                    child_nodes_iter += child_nodes_re # Accumulate nodes from re-search
                    child_max_depth_iter = max(child_max_depth_iter, child_max_depth_re)
                else:
                    current_eval_score = null_window_score # Use scout score if it didn't warrant a re-search
                    
            nodes_here += child_nodes_iter; max_depth_here = max(max_depth_here, child_max_depth_iter)
            
            if current_eval_score > max_eval: 
                max_eval = current_eval_score
                best_move_idx_this_node = move_idx
            alpha = max(alpha, max_eval)
            if alpha >= beta: break # Beta cutoff
        eval_to_store = max_eval
    else: # Minimizing player (AI 'O')
        min_eval = math.inf
        for i, move_idx in enumerate(available_moves_indices):
            move_bit = 1 << move_idx; next_x_bb = canonical_x_bb; next_o_bb = canonical_o_bb | move_bit
            child_canonical_x_bb, child_canonical_o_bb = get_canonical_bitboards(next_x_bb, next_o_bb, board_size)
            child_hash = compute_zobrist_hash_from_canonical_bbs(child_canonical_x_bb, child_canonical_o_bb)
            
            current_eval_score = 0; child_nodes_iter = 0; child_max_depth_iter = 0
            if i == 0: # Full window search
                current_eval_score, child_nodes_iter, child_max_depth_iter = _minimax_recursive_logic_bb(child_canonical_x_bb, child_canonical_o_bb, depth + 1, True, k_to_win, board_size, alpha, beta, current_max_search_depth_for_this_call, child_hash)
            else: # Null window (scout) search
                null_window_score, child_nodes_null, child_max_depth_null = _minimax_recursive_logic_bb(child_canonical_x_bb, child_canonical_o_bb, depth + 1, True, k_to_win, board_size, beta - 1, beta, current_max_search_depth_for_this_call, child_hash)
                child_nodes_iter += child_nodes_null; child_max_depth_iter = max(child_max_depth_iter, child_max_depth_null)
                
                if null_window_score < beta and null_window_score > alpha: # Re-search if necessary
                    current_eval_score, child_nodes_re, child_max_depth_re = _minimax_recursive_logic_bb(child_canonical_x_bb, child_canonical_o_bb, depth + 1, True, k_to_win, board_size, alpha, beta, current_max_search_depth_for_this_call, child_hash)
                    child_nodes_iter += child_nodes_re
                    child_max_depth_iter = max(child_max_depth_iter, child_max_depth_re)
                else:
                    current_eval_score = null_window_score
                    
            nodes_here += child_nodes_iter; max_depth_here = max(max_depth_here, child_max_depth_iter)
            
            if current_eval_score < min_eval: 
                min_eval = current_eval_score
                best_move_idx_this_node = move_idx
            beta = min(beta, min_eval)
            if beta <= alpha: break # Alpha cutoff
        eval_to_store = min_eval
        
    # Transposition Table Store
    flag = TT_EXACT
    if eval_to_store <= original_alpha: flag = TT_UPPER_BOUND # Failed low
    elif eval_to_store >= beta: flag = TT_LOWER_BOUND      # Failed high
    
    # Store if new entry or deeper search for this state
    current_tt_entry = transposition_table.get(current_zobrist_hash)
    if current_tt_entry is None or remaining_depth_tt >= current_tt_entry['depth_searched']:
        transposition_table[current_zobrist_hash] = {
            'score': eval_to_store, 
            'depth_searched': remaining_depth_tt, 
            'flag': flag, 
            'best_move_idx': best_move_idx_this_node
        }
        tt_stores += 1
        
    return (eval_to_store, nodes_here, max_depth_here)

def minimax_iterative_bb(player_x_bb, player_o_bb, is_turn_of_maximizer_x, k_to_win, board_size, search_ply_limit_from_here):
    canonical_x, canonical_o = get_canonical_bitboards(player_x_bb, player_o_bb, board_size)
    initial_hash = compute_zobrist_hash_from_canonical_bbs(canonical_x, canonical_o)
    return _minimax_recursive_logic_bb(canonical_x, canonical_o, 0, is_turn_of_maximizer_x, k_to_win, board_size, -math.inf, math.inf, search_ply_limit_from_here, initial_hash)

def find_best_move_iterative_deepening(initial_board_list, k_to_win_config, progress_q, result_q, player_token_for_move):
    global transposition_table, tt_hits, tt_stores
    transposition_table.clear(); tt_hits = 0; tt_stores = 0
    start_total_time = time.monotonic()
    current_board_size = len(initial_board_list)
    player_id_for_move = PLAYER_IDS[player_token_for_move]
    is_maximizing_search = (player_token_for_move == PLAYER_HUMAN)
    initial_x_bb, initial_o_bb = list_to_bitboards(initial_board_list, current_board_size)
    num_moves_made_abs = bin(initial_x_bb).count('1') + bin(initial_o_bb).count('1')
    max_remaining_plies = (current_board_size * current_board_size) - num_moves_made_abs
    if max_remaining_plies == 0:
        result_q.put({'best_move_data': (None, 0), 'top_moves_list': [], 'total_nodes': 0, 'achieved_relative_depth': 0}); return
    root_available_moves_indices = get_available_moves_bb(initial_x_bb, initial_o_bb, current_board_size, center_sort=True)
    if not root_available_moves_indices:
        result_q.put({'best_move_data': (None, 0), 'top_moves_list': [], 'total_nodes': 0, 'achieved_relative_depth': 0}); return
    
    # Check for immediate win for the current player
    immediate_wins_indices = _get_immediate_winning_moves_bb(initial_x_bb, initial_o_bb, player_id_for_move, k_to_win_config, current_board_size)
    if immediate_wins_indices:
        best_move_idx = immediate_wins_indices[0] # Take the first one
        best_move_rc = bit_index_to_cell(best_move_idx, current_board_size)
        score = (1000 - 0) if is_maximizing_search else (-1000 + 0) # Depth 0 for immediate win
        result_q.put({'best_move_data': (best_move_rc, score), 'top_moves_list': [{'move': best_move_rc, 'score': score, 'nodes': 1, 'depth': 0, 'actual_eval_depth': 0}], 'total_nodes': 1, 'achieved_relative_depth': 0}); return
        
    # Check if opponent has a single immediate winning move (must block)
    opponent_id = 1 - player_id_for_move
    opponent_immediate_wins_indices = _get_immediate_winning_moves_bb(initial_x_bb, initial_o_bb, opponent_id, k_to_win_config, current_board_size)
    if opponent_immediate_wins_indices and len(opponent_immediate_wins_indices) == 1:
        must_block_idx = opponent_immediate_wins_indices[0]
        if must_block_idx in root_available_moves_indices: # Ensure the block is a valid move
            root_available_moves_indices = [must_block_idx] # Only consider this blocking move
            
    overall_best_move_idx = None; overall_best_score = -math.inf if is_maximizing_search else math.inf
    move_scores_from_last_iter = {idx: (-math.inf if is_maximizing_search else math.inf) for idx in root_available_moves_indices}
    accumulated_total_nodes = 0; final_iddfs_ply_limit_for_best_move = 0
    last_iter_full_eval_details_rc = []

    for current_iddfs_ply_limit in range(1, max_remaining_plies + 1):
        progress_q.put({'total_root_moves_this_iter': len(root_available_moves_indices), 'current_depth_iter': current_iddfs_ply_limit, 'type': 'start_iter', 'max_total_depth_iters': max_remaining_plies })
        
        # Sort root moves based on scores from the previous iteration (move ordering)
        if current_iddfs_ply_limit > 1 and len(root_available_moves_indices) > 1:
            root_available_moves_indices.sort(key=lambda idx: move_scores_from_last_iter.get(idx, (-math.inf if is_maximizing_search else math.inf)), reverse=is_maximizing_search)
            
        current_iter_best_move_idx_candidate = None; current_iter_best_score_candidate = -math.inf if is_maximizing_search else math.inf
        temp_evaluated_this_iter_rc = []

        for i, root_move_idx in enumerate(root_available_moves_indices):
            move_rc_for_progress = bit_index_to_cell(root_move_idx, current_board_size)
            current_overall_time_before_eval = time.monotonic() - start_total_time
            nps_so_far_before_eval = accumulated_total_nodes / current_overall_time_before_eval if current_overall_time_before_eval > 0.001 else (accumulated_total_nodes * 1000.0 if accumulated_total_nodes > 0 else 0)
            progress_q.put({'type': 'progress_start_move_eval', 'current_nodes': accumulated_total_nodes, 'current_max_depth': num_moves_made_abs + current_iddfs_ply_limit, 'current_nps': nps_so_far_before_eval, 'time_elapsed': current_overall_time_before_eval, 'current_depth_iter': current_iddfs_ply_limit, 'root_moves_done_this_iter': i, 'total_root_moves_this_iter': len(root_available_moves_indices), 'max_total_depth_iters': max_remaining_plies, 'evaluating_move': move_rc_for_progress})
            
            root_move_bit = 1 << root_move_idx; child_x_bb, child_o_bb = initial_x_bb, initial_o_bb
            if player_id_for_move == PLAYER_IDS[PLAYER_HUMAN]: child_x_bb |= root_move_bit
            else: child_o_bb |= root_move_bit
            
            next_player_is_maximizer_x = (player_id_for_move == PLAYER_IDS[PLAYER_AI]) # Turn flips
            minimax_search_ply_limit_for_children = current_iddfs_ply_limit - 1 # Search one ply less deep
            
            eval_score, nodes_in_branch, actual_depth_this_branch_relative = minimax_iterative_bb(child_x_bb, child_o_bb, next_player_is_maximizer_x, k_to_win_config, current_board_size, minimax_search_ply_limit_for_children)
            
            accumulated_total_nodes += nodes_in_branch
            move_scores_from_last_iter[root_move_idx] = eval_score
            current_move_details_rc = {'move': bit_index_to_cell(root_move_idx, current_board_size), 'score': eval_score, 'nodes': nodes_in_branch, 'depth': current_iddfs_ply_limit, 'actual_eval_depth': actual_depth_this_branch_relative + 1}
            temp_evaluated_this_iter_rc.append(current_move_details_rc)
            
            if is_maximizing_search:
                if eval_score > current_iter_best_score_candidate: 
                    current_iter_best_score_candidate = eval_score
                    current_iter_best_move_idx_candidate = root_move_idx
            else: # Minimizing search
                if eval_score < current_iter_best_score_candidate: 
                    current_iter_best_score_candidate = eval_score
                    current_iter_best_move_idx_candidate = root_move_idx
                    
            current_overall_time = time.monotonic() - start_total_time
            nps_so_far = accumulated_total_nodes / current_overall_time if current_overall_time > 0.001 else (accumulated_total_nodes * 1000.0 if accumulated_total_nodes > 0 else 0)
            progress_q.put({'type': 'progress_end_move_eval', 'current_nodes': accumulated_total_nodes, 'current_max_depth': num_moves_made_abs + current_iddfs_ply_limit, 'current_best_score': current_iter_best_score_candidate if current_iter_best_move_idx_candidate is not None else "...", 'current_nps': nps_so_far, 'time_elapsed': current_overall_time, 'current_depth_iter': current_iddfs_ply_limit, 'root_moves_done_this_iter': i + 1, 'total_root_moves_this_iter': len(root_available_moves_indices), 'max_total_depth_iters': max_remaining_plies})
        
        last_iter_full_eval_details_rc = temp_evaluated_this_iter_rc
        if current_iter_best_move_idx_candidate is not None:
            overall_best_move_idx = current_iter_best_move_idx_candidate
            overall_best_score = current_iter_best_score_candidate
            final_iddfs_ply_limit_for_best_move = current_iddfs_ply_limit
            
            # Early exit if a mate is found (score reflects win/loss at this depth)
            # The score for a win at depth `d` is `1000 - d`. Here, `current_iddfs_ply_limit - 1` is the depth of the child node.
            if (is_maximizing_search and current_iter_best_score_candidate >= (1000 - (current_iddfs_ply_limit -1))) or \
               (not is_maximizing_search and current_iter_best_score_candidate <= (-1000 + (current_iddfs_ply_limit -1))):
                break 
        
        # Another early exit condition if a strong win/loss is found (close to max score)
        if abs(overall_best_score) > 990 and overall_best_move_idx is not None: 
            break

    if overall_best_move_idx is None and root_available_moves_indices:
        overall_best_move_idx = root_available_moves_indices[0] # Fallback to first available move if no clear best
        overall_best_score = move_scores_from_last_iter.get(overall_best_move_idx, (-math.inf if is_maximizing_search else math.inf))
        found_fallback_details = next((item for item in last_iter_full_eval_details_rc if item['move'] == bit_index_to_cell(overall_best_move_idx, current_board_size)), None)
        if found_fallback_details: 
            final_iddfs_ply_limit_for_best_move = found_fallback_details.get('depth', 1)
        else: 
            final_iddfs_ply_limit_for_best_move = 1 if max_remaining_plies > 0 else 0
            
    final_best_move_rc = bit_index_to_cell(overall_best_move_idx, current_board_size) if overall_best_move_idx is not None else None
    
    final_top_moves_list_rc = []
    if last_iter_full_eval_details_rc:
        last_iter_full_eval_details_rc.sort(key=lambda x: x['score'], reverse=is_maximizing_search)
        final_top_moves_list_rc = last_iter_full_eval_details_rc[:5]
    elif final_best_move_rc: # Should only be hit if search was extremely short
        final_top_moves_list_rc = [{'move': final_best_move_rc, 'score': overall_best_score, 'nodes': 1, 'depth': final_iddfs_ply_limit_for_best_move, 'actual_eval_depth': final_iddfs_ply_limit_for_best_move}]
        
    result_q.put({'best_move_data': (final_best_move_rc, overall_best_score), 'top_moves_list': final_top_moves_list_rc, 'total_nodes': accumulated_total_nodes, 'achieved_relative_depth': final_iddfs_ply_limit_for_best_move})

# --- Tkinter GUI Class ---
class TicTacToeGUI:
    TOP_MOVES_DISPLAY_LINES = 6
    def __init__(self, root_window):
        self.root = root_window; self.root.title("Tic-Tac-Toe AI"); self.root.configure(bg=COLOR_ROOT_BG)
        try: self.default_font_family = tkfont.nametofont("TkDefaultFont").actual()["family"]
        except: self.default_font_family = "Segoe UI" if "win" in self.root.tk.call("tk", "windowingsystem") else "Helvetica"
        self.fonts = { "header": tkfont.Font(family=self.default_font_family, size=18, weight="bold"), "status": tkfont.Font(family=self.default_font_family, size=12, weight="bold"), "label": tkfont.Font(family=self.default_font_family, size=10), "button": tkfont.Font(family=self.default_font_family, size=10, weight="bold"), "entry": tkfont.Font(family=self.default_font_family, size=10), "info_header": tkfont.Font(family=self.default_font_family, size=11, weight="bold"), "info": tkfont.Font(family=self.default_font_family, size=9), "info_value": tkfont.Font(family=self.default_font_family, size=9, weight="bold"), "move_info": tkfont.Font(family="Consolas", size=9), "top_move_info": tkfont.Font(family="Consolas", size=10)}
        self.setup_styles(); self.board_size = 4; self.k_to_win = 4; self.current_player = PLAYER_HUMAN; self.game_board = []; self.buttons = []; self.game_over = True; self.num_moves_made = 0; self.last_time_taken_update = 0
        self.current_turn_max_searchable_depth = 0; self.num_moves_made_at_calc_start = 0
        self.calculation_manager_thread = None; self.progress_queue = thread_queue.Queue(); self.result_queue = thread_queue.Queue()
        self.root.attributes("-fullscreen", True); self.is_fullscreen = True
        self.nodes_explored_var = tk.StringVar(value="0"); self.actual_depth_var = tk.StringVar(value="N/A"); self.ai_eval_var = tk.StringVar(value="N/A"); self.status_var = tk.StringVar(value="Loading game..."); self.time_taken_var = tk.StringVar(value="N/A"); self.top_moves_var = tk.StringVar(value=self._get_empty_top_moves_text()); self.nps_var = tk.StringVar(value="N/A"); self.progress_percent_var = tk.StringVar(value="Calculating..."); self.detailed_progress_var = tk.StringVar(value="")
        self.max_game_length_var = tk.StringVar(value="N/A")
        self.setup_ui_layout(); self.root.bind("<F11>", self.toggle_fullscreen); self.root.bind("<Escape>", self.on_escape_key); self.root.protocol("WM_DELETE_WINDOW", self.on_silent_close)
        self.root.after_idle(lambda: self.start_new_game(human_starts=True))
    def _get_empty_top_moves_text(self):
        lines = ["Top Move Considerations:", f"  {'Move':<5} {'Score':<6} {'Nodes':<10}", "-" * 30, "  (After calculation)"]
        while len(lines) < self.TOP_MOVES_DISPLAY_LINES: lines.append("")
        return "\n".join(lines[:self.TOP_MOVES_DISPLAY_LINES])
    def _set_status_as_hint(self, text): self.status_var.set(text); self.status_label.config(style="HintSuggest.TLabel", anchor=tk.CENTER)
    def _set_status_normal(self, text): self.status_var.set(text); self.status_label.config(style="Status.TLabel", anchor=tk.W)
    def setup_styles(self):
        self.style = ttk.Style(); available_themes = self.style.theme_names(); current_os = self.root.tk.call("tk", "windowingsystem")
        if 'clam' in available_themes: self.style.theme_use('clam')
        elif current_os == "win32" and 'vista' in available_themes: self.style.theme_use('vista')
        elif current_os == "aqua" and 'aqua' in available_themes: self.style.theme_use('aqua')
        else: self.style.theme_use(available_themes[0] if available_themes else 'default')
        self.style.configure(".", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"])
        self.style.configure("TFrame", background=COLOR_ROOT_BG); self.style.configure("Content.TFrame", background=COLOR_FRAME_BG)
        self.style.configure("TLabel", background=COLOR_ROOT_BG, foreground=COLOR_TEXT_PRIMARY, font=self.fonts["label"], padding=2)
        self.style.configure("Header.TLabel", font=self.fonts["header"], foreground=COLOR_ACCENT_SECONDARY, background=COLOR_FRAME_BG)
        self.style.configure("Status.TLabel", font=self.fonts["status"], foreground=COLOR_TEXT_PRIMARY, background=COLOR_FRAME_BG)
        self.style.configure("Info.TLabel", font=self.fonts["info"], foreground=COLOR_TEXT_SECONDARY, background=COLOR_FRAME_BG)
        self.style.configure("InfoValue.TLabel", font=self.fonts["info_value"], foreground=COLOR_TEXT_PRIMARY, background=COLOR_FRAME_BG)
        self.style.configure("DetailedProgress.TLabel", font=self.fonts["info"], foreground=COLOR_TEXT_SECONDARY, background=COLOR_FRAME_BG, anchor=tk.W)
        self.style.configure("HintSuggest.TLabel", font=self.fonts["status"], foreground=COLOR_HINT_TEXT, background=COLOR_HINT_SUGGEST_BG, padding=5, borderwidth=1, relief="solid", bordercolor=COLOR_HINT)
        self.style.configure("TopMoveInfo.TLabel", font=self.fonts["top_move_info"], background=COLOR_FRAME_BG, foreground=COLOR_TEXT_PRIMARY, padding=5, borderwidth=1, relief="groove", bordercolor="#CCCCCC")
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
        self.style.configure("Hint.Horizontal.TProgressbar", thickness=12, background=COLOR_HINT_PROGRESS_BAR, troughcolor="#DDDDDD", bordercolor=COLOR_HINT)
    def toggle_fullscreen(self, event=None): self.is_fullscreen = not self.is_fullscreen; self.root.attributes("-fullscreen", self.is_fullscreen); self.root.after(50, self.create_board_ui_buttons) if self.buttons else None
    def setup_ui_layout(self):
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL); self.paned_window.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        left_pane_container = ttk.Frame(self.paned_window, padding=(5,0)); self.paned_window.add(left_pane_container, weight=1)
        left_pane_content = ttk.Frame(left_pane_container, style="Content.TFrame", relief="sunken", borderwidth=1, padding=10); left_pane_content.pack(fill=tk.BOTH, expand=True); left_pane_content.grid_columnconfigure(0, weight=1); left_pane_content.grid_rowconfigure(2, weight=1)
        ttk.Label(left_pane_content, text="Tic-Tac-Toe AI", style="Header.TLabel", anchor=tk.CENTER).grid(row=0, column=0, pady=(5,15), sticky="ew")
        control_frame = ttk.Labelframe(left_pane_content, text="Game Settings", padding=(10,5), style="Content.TLabelframe"); control_frame.grid(row=1, column=0, sticky="new", pady=(0,10), padx=0); control_frame.grid_columnconfigure(1, weight=1)
        entry_padx, entry_pady, label_padx = (5,2), (2,2), (5,0)
        ttk.Label(control_frame, text="Board Size:", style="Info.TLabel").grid(row=0, column=0, padx=label_padx, pady=entry_pady, sticky="w"); self.size_entry = ttk.Entry(control_frame, width=4, font=self.fonts["entry"]); self.size_entry.insert(0, str(self.board_size)); self.size_entry.grid(row=0, column=1, padx=entry_padx, pady=entry_pady, sticky="ew")
        ttk.Label(control_frame, text="K-in-a-row:", style="Info.TLabel").grid(row=1, column=0, padx=label_padx, pady=entry_pady, sticky="w"); self.k_entry = ttk.Entry(control_frame, width=4, font=self.fonts["entry"]); self.k_entry.insert(0, str(self.k_to_win)); self.k_entry.grid(row=1, column=1, padx=entry_padx, pady=entry_pady, sticky="ew")
        button_frame = ttk.Frame(control_frame, style="Content.TFrame", padding=(0,5)); button_frame.grid(row=2, column=0, columnspan=2, pady=(8,0), sticky="ew"); button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        self.start_human_button = ttk.Button(button_frame, text="▶ You Start", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=True)); self.start_human_button.grid(row=0, column=0, padx=(0,1), pady=1, sticky="ew")
        self.start_ai_button = ttk.Button(button_frame, text="🤖 AI Starts", style="Accent.TButton", command=lambda: self.start_new_game(human_starts=False)); self.start_ai_button.grid(row=0, column=1, padx=(1,0), pady=1, sticky="ew")
        self.hint_button = ttk.Button(button_frame, text="💡 Suggest", style="Hint.TButton", command=self.get_human_hint, state=tk.DISABLED); self.hint_button.grid(row=1, column=0, columnspan=2, pady=(3,0), sticky="ew")
        status_vis_frame = ttk.Labelframe(left_pane_content, text="AI Insights", style="Content.TLabelframe"); status_vis_frame.grid(row=2, column=0, sticky="nsew", pady=(5,0), padx=0); status_vis_frame.grid_columnconfigure(0, weight=1); status_vis_frame.grid_rowconfigure(4, weight=1)
        self.status_label = ttk.Label(status_vis_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=320, anchor=tk.W, justify=tk.LEFT); self.status_label.grid(row=0, column=0, columnspan=2, pady=(0,8), sticky="new")
        progress_bar_frame = ttk.Frame(status_vis_frame, style="Content.TFrame"); progress_bar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,1)); progress_bar_frame.columnconfigure(0, weight=1); progress_bar_frame.columnconfigure(1, weight=0, minsize=40)
        self.progress_bar = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=100, mode="determinate", style="Horizontal.TProgressbar"); self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,2))
        self.progress_percent_label = ttk.Label(progress_bar_frame, textvariable=self.progress_percent_var, style="Info.TLabel", anchor=tk.W); self.progress_percent_label.grid(row=0, column=1, sticky="w", padx=(2,0))
        self.detailed_progress_label = ttk.Label(status_vis_frame, textvariable=self.detailed_progress_var, style="DetailedProgress.TLabel", anchor=tk.W); self.detailed_progress_label.grid(row=2, column=0, columnspan=2, pady=(0,5), sticky="new", padx=2)
        stats_grid = ttk.Frame(status_vis_frame, style="Content.TFrame"); stats_grid.grid(row=3, column=0, columnspan=2, sticky="new", pady=(0,5)); stats_grid.columnconfigure(0, weight=0, minsize=130); stats_grid.columnconfigure(1, weight=1)
        row_idx_stats = 0; stats_to_display = [("Time Taken (s):", self.time_taken_var), ("Nodes Explored:", self.nodes_explored_var), ("Nodes/Sec (NPS):", self.nps_var), ("Search Depth:", self.actual_depth_var), ("Move Score:", self.ai_eval_var)]
        for label_text, var in stats_to_display: ttk.Label(stats_grid, text=label_text, style="Info.TLabel").grid(row=row_idx_stats, column=0, sticky="e", padx=(2,5), pady=1); ttk.Label(stats_grid, textvariable=var, style="InfoValue.TLabel").grid(row=row_idx_stats, column=1, sticky="w", padx=2, pady=1); row_idx_stats += 1
        ttk.Label(stats_grid, text="(Score: >0 X, <0 O)", style="Info.TLabel", foreground="#6c757d").grid(row=row_idx_stats, column=0, columnspan=2, sticky="w", padx=2, pady=(1,3)); row_idx_stats +=1
        ttk.Label(stats_grid, text="Max Game Length:", style="Info.TLabel").grid(row=row_idx_stats, column=0, sticky="e", padx=(2,5), pady=1); ttk.Label(stats_grid, textvariable=self.max_game_length_var, style="InfoValue.TLabel").grid(row=row_idx_stats, column=1, sticky="w", padx=2, pady=1)
        self.top_moves_label = ttk.Label(status_vis_frame, textvariable=self.top_moves_var, style="TopMoveInfo.TLabel", anchor="nw", wraplength=280); self.top_moves_label.grid(row=4, column=0, columnspan=2, pady=(10,0), sticky="nsew")
        self.board_outer_frame = ttk.Frame(self.paned_window, style="TFrame", padding=10); self.paned_window.add(self.board_outer_frame, weight=2); self.board_outer_frame.grid_rowconfigure(0, weight=1); self.board_outer_frame.grid_columnconfigure(0, weight=1)
        self.board_frame = ttk.Frame(self.board_outer_frame, style="TFrame"); self.board_frame.grid(row=0, column=0, sticky=""); self._set_status_normal("Adjust N, K, then Start.")
    def on_escape_key(self, event=None): self.on_silent_close()
    def on_silent_close(self, event=None): self.root.destroy()

    def start_new_game(self, human_starts=True):
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): messagebox.showwarning("Busy", "AI calculation is currently in progress."); return
        try:
            n_val = int(self.size_entry.get()); k_val = int(self.k_entry.get())
            if not (2 <= n_val <= 7): messagebox.showerror("Error", "Board size N must be between 2 and 7."); return
            if not (2 <= k_val <= n_val): messagebox.showerror("Error", f"K-in-a-row must be between 2 and N (currently {n_val})."); return
            self.board_size = n_val; self.k_to_win = k_val
            
            init_zobrist_bitboard(self.board_size)
            
            # --- CORRECTED CACHE CLEARING ---
            WINNING_MASKS_CACHE.clear() 
            POTENTIAL_LINE_MASKS_CACHE.clear()
            # --- END CORRECTION ---
            
            # Pre-populate for new size/k
            get_win_masks(self.board_size, self.k_to_win)
            get_potential_line_masks(self.board_size, self.k_to_win)
            
            # Clear LRU caches for other functions
            compute_zobrist_hash_from_canonical_bbs.cache_clear()
            get_canonical_bitboards.cache_clear()
            check_win_bb.cache_clear()
            is_board_full_bb.cache_clear()
            _get_immediate_winning_moves_bb.cache_clear()
            can_force_win_on_next_player_turn_bb.cache_clear()
            is_unwinnable_for_either_bb.cache_clear()
            cell_to_bit_index.cache_clear()
            bit_index_to_cell.cache_clear()
            
        except ValueError: messagebox.showerror("Error", "Invalid input for N or K."); return
        
        total_cells = self.board_size * self.board_size; self.max_game_length_var.set(f"{total_cells} plies"); self.current_turn_max_searchable_depth = total_cells
        self.game_board = [[EMPTY_CELL for _ in range(self.board_size)] for _ in range(self.board_size)]; self.num_moves_made = 0; self.current_player = PLAYER_HUMAN if human_starts else PLAYER_AI; self.game_over = False
        self._set_status_normal(f"{self.board_size}x{self.board_size}, {self.k_to_win}-in-a-row. {'Your turn (X).' if human_starts else 'AI`s turn (O).'}"); self.ai_eval_var.set("N/A"); self.nodes_explored_var.set("0");
        self.actual_depth_var.set(f"0/{self.current_turn_max_searchable_depth}")
        self.time_taken_var.set("N/A"); self.nps_var.set("N/A"); self.progress_percent_var.set("Calculating..."); self.detailed_progress_var.set(""); self.top_moves_var.set(self._get_empty_top_moves_text())
        if self.progress_bar.master.winfo_ismapped(): self.progress_bar.master.grid_remove()
        if self.detailed_progress_label.winfo_ismapped(): self.detailed_progress_label.grid_remove()
        self.create_board_ui_buttons(); self.update_hint_button_state()
        if not human_starts: self.root.after(100, lambda: self.trigger_ai_or_hint_calculation(PLAYER_AI))
    
# ... (rest of the GUI class methods: update_hint_button_state, get_human_hint, etc. remain the same as in the previous correct version) ...
# --- The rest of the TicTacToeGUI class and the __main__ block remain unchanged ---
# (Make sure to copy them from the previous version of the code that included the _get_immediate_winning_moves_bb optimization)

    def update_hint_button_state(self): is_calc_running = self.calculation_manager_thread and self.calculation_manager_thread.is_alive(); can_hint = not self.game_over and self.current_player == PLAYER_HUMAN and not is_calc_running; self.hint_button.config(state=tk.NORMAL if can_hint else tk.DISABLED)
    def get_human_hint(self):
        if self.game_over or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): messagebox.showinfo("Busy", "AI calculation is currently in progress."); return
        self.trigger_ai_or_hint_calculation(PLAYER_HUMAN)
    def trigger_ai_or_hint_calculation(self, player_to_calculate_for):
        self.num_moves_made_at_calc_start = self.num_moves_made; self.current_turn_max_searchable_depth = (self.board_size * self.board_size) - self.num_moves_made
        if self.current_turn_max_searchable_depth <= 0: self.current_turn_max_searchable_depth = 1
        if player_to_calculate_for == PLAYER_HUMAN: self._set_status_as_hint("Calculating your best move..."); self.progress_bar.config(style="Hint.Horizontal.TProgressbar")
        else: self._set_status_normal("AI (O) is thinking..."); self.progress_bar.config(style="Horizontal.TProgressbar")
        self.actual_depth_var.set(f"0/{self.current_turn_max_searchable_depth}")
        self.top_moves_var.set(self._get_empty_top_moves_text().replace("(After calculation)", "Calculating...")); self.disable_board_buttons(); self.hint_button.config(state=tk.DISABLED); self.progress_bar.master.grid(); self.detailed_progress_label.grid(); self.progress_bar["value"] = 0; self.progress_percent_var.set("Starting..."); self.detailed_progress_var.set("Initializing search...")
        self.root.config(cursor="watch"); self.root.update_idletasks()
        while not self.progress_queue.empty(): self.progress_queue.get_nowait()
        while not self.result_queue.empty(): self.result_queue.get_nowait()
        self.calculation_manager_thread = threading.Thread(target=find_best_move_iterative_deepening, args=(self.game_board, self.k_to_win, self.progress_queue, self.result_queue, player_to_calculate_for), daemon=True)
        self.ai_start_time = time.monotonic(); self.last_time_taken_update = self.ai_start_time; self._current_calculation_for_player = player_to_calculate_for; self.calculation_manager_thread.start(); self.check_ai_progress()
    def create_board_ui_buttons(self, attempt=1):
        for widget in self.board_frame.winfo_children(): widget.destroy(); self.buttons = []
        self.root.update_idletasks(); container_width, container_height = self.board_outer_frame.winfo_width(), self.board_outer_frame.winfo_height()
        if (container_width < 50 or container_height < 50) and attempt < 10: self.root.after(100, lambda: self.create_board_ui_buttons(attempt + 1)); return
        if container_width < 50 or container_height < 50: min_dim_root = min(self.root.winfo_width(), self.root.winfo_height()) * 0.6; container_width, container_height = max(min_dim_root, 200), max(min_dim_root, 200)
        board_dimension = max(self.board_size * 35, min(container_width, container_height) - 10); cell_dim = board_dimension // self.board_size; self.board_frame.config(width=board_dimension, height=board_dimension)
        board_button_font = tkfont.Font(family=self.default_font_family, size=max(10, int(cell_dim * 0.45)), weight="bold")
        for i in range(self.board_size): self.board_frame.grid_rowconfigure(i, weight=1, minsize=cell_dim); self.board_frame.grid_columnconfigure(i, weight=1, minsize=cell_dim)
        for r in range(self.board_size):
            row_buttons = []
            for c_idx in range(self.board_size): button = tk.Button(self.board_frame, text=EMPTY_CELL, font=board_button_font, relief="flat", borderwidth=1, bg=COLOR_FRAME_BG, activebackground="#DDDDDD", fg=COLOR_TEXT_PRIMARY, command=lambda r_idx=r, c_idx_btn=c_idx: self.handle_cell_click(r_idx, c_idx_btn)); button.grid(row=r, column=c_idx, sticky="nsew", padx=1, pady=1); row_buttons.append(button)
            self.buttons.append(row_buttons)
        self.update_board_button_states()
    def handle_cell_click(self, r, c):
        if self.game_over or self.game_board[r][c] != EMPTY_CELL or self.current_player != PLAYER_HUMAN: return
        if self.calculation_manager_thread and self.calculation_manager_thread.is_alive(): messagebox.showinfo("Wait", "AI calculation is currently in progress."); return
        self._set_status_normal(f"You placed X at {to_algebraic(r,c,self.board_size)}. AI (O) thinking..."); self.clear_all_hint_highlights(); self.make_move(r, c, PLAYER_HUMAN)
        if self.check_game_status(): return
        self.current_player = PLAYER_AI; self.update_hint_button_state(); self.trigger_ai_or_hint_calculation(PLAYER_AI)
    def clear_all_hint_highlights(self):
        if not self.buttons or not self.game_board: return
        for r_idx in range(self.board_size):
            if r_idx < len(self.buttons) and self.buttons[r_idx]:
                for c_idx in range(self.board_size):
                    if c_idx < len(self.buttons[r_idx]) and self.buttons[r_idx][c_idx] and self.game_board[r_idx][c_idx] == EMPTY_CELL and self.buttons[r_idx][c_idx].cget('bg') == COLOR_HINT_SUGGEST_BG:
                        self.buttons[r_idx][c_idx].config(bg=COLOR_FRAME_BG, relief="flat")
    def check_ai_progress(self):
        update_interval_ms, max_messages_to_process = 35, 5; current_runtime = 0.0; is_calculating = self.calculation_manager_thread and self.calculation_manager_thread.is_alive()
        if is_calculating:
            current_monotonic_time = time.monotonic(); current_runtime = current_monotonic_time - self.ai_start_time
            if current_monotonic_time - self.last_time_taken_update > 0.083: self.time_taken_var.set(f"{current_runtime:.2f}"); self.last_time_taken_update = current_monotonic_time
        latest_nodes, latest_best_score_so_far, latest_iddfs_current_iter_depth, latest_iddfs_max_target_depth, latest_iddfs_root_moves_done, latest_iddfs_total_root_moves, latest_message_type = None, None, None, None, None, None, None
        try: current_nodes_str = self.nodes_explored_var.get(); latest_nodes = int(current_nodes_str.replace(',', '')) if current_nodes_str and current_nodes_str != "0" else None
        except ValueError: pass
        for _ in range(max_messages_to_process):
            try:
                progress_update = self.progress_queue.get_nowait()
                if progress_update.get('current_nodes') is not None: latest_nodes = progress_update['current_nodes']
                if 'current_best_score' in progress_update: latest_best_score_so_far = progress_update['current_best_score']
                latest_iddfs_current_iter_depth = progress_update.get('current_depth_iter', latest_iddfs_current_iter_depth)
                latest_iddfs_max_target_depth = progress_update.get('max_total_depth_iters', latest_iddfs_max_target_depth)
                latest_iddfs_root_moves_done = progress_update.get('root_moves_done_this_iter', latest_iddfs_root_moves_done)
                latest_iddfs_total_root_moves = progress_update.get('total_root_moves_this_iter', latest_iddfs_total_root_moves)
                latest_message_type = progress_update.get('type', latest_message_type)
            except thread_queue.Empty: break
        if is_calculating:
            try: previous_displayed_nodes = int(self.nodes_explored_var.get().replace(',', '')) if self.nodes_explored_var.get().replace(',', '').isdigit() else -1
            except ValueError: previous_displayed_nodes = -1
            if latest_nodes is not None:
                if latest_nodes != previous_displayed_nodes:
                    self.nodes_explored_var.set(f"{latest_nodes:,}")
                    if current_runtime > 0.01: self.nps_var.set(f"{(latest_nodes / current_runtime):,.0f}")
                    elif latest_nodes > 0: self.nps_var.set("High")
                    else: self.nps_var.set("0")
            elif previous_displayed_nodes == -1 and latest_nodes is None: self.nodes_explored_var.set("0"); self.nps_var.set("N/A")
            if latest_iddfs_current_iter_depth is not None: self.actual_depth_var.set(f"{latest_iddfs_current_iter_depth}/{self.current_turn_max_searchable_depth}")
            if latest_best_score_so_far is not None:
                score_val = latest_best_score_so_far; score_str = ""
                if isinstance(score_val, (int, float)) and score_val not in (math.inf, -math.inf): score_str = f"{score_val:.0f}?"
                elif isinstance(score_val, str): score_str = score_val
                else: score_str = "Win/Loss?"
                self.ai_eval_var.set(score_str)
            if latest_iddfs_current_iter_depth is not None and latest_iddfs_max_target_depth is not None:
                target_plies_for_progress_calc = max(1, latest_iddfs_max_target_depth); current_iter_depth_disp = latest_iddfs_current_iter_depth
                moves_processed_for_display = latest_iddfs_root_moves_done if latest_iddfs_root_moves_done is not None else 0
                if latest_message_type == 'start_iter': moves_processed_for_display = 0
                total_root_moves_disp = max(1, latest_iddfs_total_root_moves if latest_iddfs_total_root_moves is not None else 1)
                self.detailed_progress_var.set(f"Depth {current_iter_depth_disp}: ({moves_processed_for_display}/{total_root_moves_disp})")
                current_progress_bar_val = moves_processed_for_display
                if self.progress_bar["maximum"] != total_root_moves_disp: self.progress_bar["maximum"] = total_root_moves_disp
                self.progress_bar["value"] = current_progress_bar_val
                frac_current_iter_done = current_progress_bar_val / total_root_moves_disp if total_root_moves_disp > 0 else 0
                eff_completed_depth_iterations = (max(0, current_iter_depth_disp - 1)) + frac_current_iter_done
                overall_percentage = min(100.0, max(0.0, (eff_completed_depth_iterations / target_plies_for_progress_calc) * 100))
                self.progress_percent_var.set(f"{overall_percentage:.0f}%")
        if is_calculating: self.root.after(update_interval_ms, self.check_ai_progress)
        elif self.calculation_manager_thread:
            final_runtime = time.monotonic() - self.ai_start_time
            if time.monotonic() - self.last_time_taken_update > 0.001: self.time_taken_var.set(f"{final_runtime:.3f}")
            last_nodes, last_best_score = latest_nodes, latest_best_score_so_far
            while not self.progress_queue.empty():
                try:
                    final_prog_update = self.progress_queue.get_nowait()
                    if 'current_nodes' in final_prog_update: last_nodes = final_prog_update['current_nodes']
                    if 'current_best_score' in final_prog_update: last_best_score = final_prog_update['current_best_score']
                except thread_queue.Empty: break
            if last_nodes is not None: self.nodes_explored_var.set(f"{last_nodes:,}"); self.nps_var.set(f"{(last_nodes / final_runtime):,.0f}" if final_runtime > 0.01 else ("High" if last_nodes > 0 else "0"))
            else: self.nodes_explored_var.set("0"); self.nps_var.set("N/A")
            if last_best_score is not None: self.ai_eval_var.set(f"{last_best_score:.0f}" if isinstance(last_best_score, (int, float)) and last_best_score not in (math.inf, -math.inf) else str(last_best_score))
            self.handle_calculation_result(); self.calculation_manager_thread = None
            self.progress_percent_var.set("100%"); self.detailed_progress_var.set("Search complete.")
    def handle_calculation_result(self):
        self.root.config(cursor="");
        if self.progress_bar.master.winfo_ismapped(): self.progress_bar.master.grid_remove()
        if self.detailed_progress_label.winfo_ismapped(): self.detailed_progress_label.grid_remove()
        try: time_taken_calc = float(self.time_taken_var.get())
        except ValueError: time_taken_calc = time.monotonic() - self.ai_start_time; self.time_taken_var.set(f"{time_taken_calc:.3f}")
        was_hint = (self._current_calculation_for_player == PLAYER_HUMAN)
        try:
            res = self.result_queue.get_nowait(); move, score = res['best_move_data']; top_moves = res['top_moves_list']; nodes = res['total_nodes']; achieved_relative_depth = res.get('achieved_relative_depth', 0)
        except thread_queue.Empty: self._set_status_normal("Error: No result from AI."); self.enable_board_buttons(); self.update_hint_button_state(); return
        self.nodes_explored_var.set(f"{nodes:,}"); self.nps_var.set(f"{(nodes / time_taken_calc):,.0f}" if time_taken_calc > 1e-4 else ("High" if nodes > 0 else "0"))
        self.actual_depth_var.set(f"{achieved_relative_depth}/{self.current_turn_max_searchable_depth}")
        self.ai_eval_var.set(f"{score:.0f}" if score not in (math.inf, -math.inf) else "Win/Loss")
        move_alg = to_algebraic(move[0], move[1], self.board_size) if move else "N/A"
        top_text_lines = [f"Top {'Human (X)' if was_hint else 'AI (O)'} Moves:", f"  {'Move':<5} {'Score':<6} {'Nodes':<10}", "-" * 30]
        if top_moves:
            for item in top_moves[:self.TOP_MOVES_DISPLAY_LINES - len(top_text_lines)]: top_text_lines.append(f"  {to_algebraic(item['move'][0], item['move'][1], self.board_size):<5} {f"{item['score']:.0f}" if isinstance(item['score'], (int,float)) else str(item['score']):<6} ({f"{item.get('nodes',0):,}"})")
        elif len(top_text_lines) < self.TOP_MOVES_DISPLAY_LINES: top_text_lines.append("  (N/A)")
        while len(top_text_lines) < self.TOP_MOVES_DISPLAY_LINES: top_text_lines.append("")
        self.top_moves_var.set("\n".join(top_text_lines[:self.TOP_MOVES_DISPLAY_LINES]))
        if was_hint:
            if move: self._set_status_as_hint(f"Suggested for X: {move_alg} (Score: {score:.0f})"); self.buttons[move[0]][move[1]].config(bg=COLOR_HINT_SUGGEST_BG, relief="raised") if self.game_board[move[0]][move[1]] == EMPTY_CELL and self.buttons[move[0]][move[1]]['state']!=tk.DISABLED else None
            else: self._set_status_as_hint("Hint: No moves or game ended.")
        else:
            self.clear_all_hint_highlights()
            if move:
                self.make_move(move[0], move[1], PLAYER_AI)
                if self.check_game_status(): self.enable_board_buttons(); self.update_hint_button_state(); return
                self.current_player = PLAYER_HUMAN; self._set_status_normal(f"AI (O) moved to {move_alg}. Your turn (X).")
            else:
                current_x_bb, current_o_bb = list_to_bitboards(self.game_board, self.board_size)
                self._set_status_normal("It's a Draw!" if is_board_full_bb(current_x_bb, current_o_bb, self.board_size) else "AI error/no moves. Game Over?"); self.game_over = True
        self.enable_board_buttons(); self.update_hint_button_state()
    def make_move(self, r, c, player): self.game_board[r][c] = player; self.num_moves_made += 1; btn = self.buttons[r][c]; btn.config(text=player, state=tk.DISABLED, relief=tk.SUNKEN, disabledforeground=(COLOR_ACCENT_SECONDARY if player == PLAYER_HUMAN else COLOR_DANGER), background=(COLOR_HUMAN_MOVE_BG if player == PLAYER_HUMAN else COLOR_AI_MOVE_BG))
    def check_game_status(self):
        human_line, ai_line = get_winning_line(self.game_board, PLAYER_HUMAN, self.k_to_win), get_winning_line(self.game_board, PLAYER_AI, self.k_to_win)
        current_x_bb, current_o_bb = list_to_bitboards(self.game_board, self.board_size)
        full = is_board_full_bb(current_x_bb, current_o_bb, self.board_size)
        over_changed = False
        if not self.game_over:
            if human_line: self._set_status_normal("You (X) Win!"); self.game_over=True; over_changed=True; self.highlight_winning_line(human_line)
            elif ai_line: self._set_status_normal("AI (O) Wins!"); self.game_over=True; over_changed=True; self.highlight_winning_line(ai_line)
            elif full: self._set_status_normal("It's a Draw!"); self.game_over=True; over_changed=True
        if over_changed or self.game_over: self.disable_board_buttons(); self.update_hint_button_state()
        return self.game_over
    def update_board_button_states(self):
        if not self.buttons: return
        for r, row_btns in enumerate(self.buttons):
            for c, button in enumerate(row_btns):
                if not button: continue
                current_cell_text = self.game_board[r][c]
                is_hinted = button.cget('bg') == COLOR_HINT_SUGGEST_BG
                if current_cell_text == EMPTY_CELL: button.config(text=current_cell_text, state=tk.NORMAL, fg=COLOR_TEXT_PRIMARY, bg=COLOR_HINT_SUGGEST_BG if is_hinted else COLOR_FRAME_BG, relief="raised" if is_hinted else "flat")
                else: button.config(text=current_cell_text, state=tk.DISABLED, relief="sunken", disabledforeground=(COLOR_ACCENT_SECONDARY if current_cell_text == PLAYER_HUMAN else COLOR_DANGER), background=(COLOR_HUMAN_MOVE_BG if current_cell_text == PLAYER_HUMAN else COLOR_AI_MOVE_BG))
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
             if r < len(self.buttons) and self.buttons[r] and c < len(self.buttons[r]) and self.buttons[r][c]: self.buttons[r][c].config(background=COLOR_WIN_HIGHLIGHT, relief=tk.GROOVE)
             
# --- Main Execution ---
if __name__ == "__main__":
    main_root = tk.Tk()
    app = TicTacToeGUI(main_root)
    main_root.mainloop()