import tkinter as tk
from tkinter import ttk, font
import chess
import time
import random
from typing import Tuple, List, Optional, Dict, Any
import threading
import queue

# --- Global Configuration ---
default_ai_depth = 4
NUM_CHECKS_TO_WIN = 3
INITIAL_SQUARE_SIZE = 66
INITIAL_WINDOW_WIDTH = (8 * INITIAL_SQUARE_SIZE) + 290
INITIAL_WINDOW_HEIGHT = (8 * INITIAL_SQUARE_SIZE) + 29
CONTROL_PANEL_DEFAULT_WIDTH = INITIAL_WINDOW_WIDTH - (8 * INITIAL_SQUARE_SIZE) - 37
TT_SIZE_POWER_OF_2 = 16 # 2^16 = 65536 entries
# ----------------------------

# --- AI Evaluation Constants ---
EVAL_WIN_SCORE = 1000000 # Score for N-check win
EVAL_LOSS_SCORE = -1000000 # Score for N-check loss
EVAL_CHECKMATE_SCORE = 900000 # Score for traditional checkmate

EVAL_PIECE_VALUES_RAW = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0 # King value for material count, not game eval
}

EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS = 900 # Bonus for being close to N-check win
EVAL_CHECK_THREAT_BONUS = 300 # Bonus for having a checking move available
# ----------------------------

# Piece Square Tables (PSTs) - For White's perspective. Mirrored for Black.
ORIGINAL_PSTS_RAW = {
    chess.PAWN: [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [0,0,0,0,0,0,0,0,5,5,10,10,10,10,5,5,-5,0,0,0,0,0,0,-5,0,0,0,0,0,0,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [ 20, 30, 10,  0,  0, 10, 30, 20, 20, 20,  0,  0,  0,  0, 20, 20,-10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30]
}
# Reverse PSTs for python-chess square indexing (A1=0, H8=63)
PST = {piece_type: list(reversed(values)) for piece_type, values in ORIGINAL_PSTS_RAW.items()}

# Transposition Table Flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2
CHESS_SQUARES = list(chess.SQUARES) # Precompute for iteration

class ChessAI:
    # Move Ordering Bonuses for N-check variant
    CHECK_BONUS_MIN_ORDERING = 5000
    CHECK_BONUS_ADDITIONAL_MAX_ORDERING = 15000
    CHECK_BONUS_DECAY_BASE_ORDERING = 5/6

    def __init__(self):
        self.transposition_table: Dict[int, Dict[str, Any]] = {}
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)] # Max depth 64
        self.history_table: Dict[Tuple[chess.PieceType, chess.Square], int] = {}
        self.tt_size_limit = 2**TT_SIZE_POWER_OF_2
        self.nodes_evaluated = 0
        self.q_nodes_evaluated = 0
        self.current_eval_for_ui = 0 # Stores the raw eval score from AI's perspective

        self.WIN_SCORE = EVAL_WIN_SCORE
        self.LOSS_SCORE = EVAL_LOSS_SCORE
        self.CHECKMATE_SCORE = EVAL_CHECKMATE_SCORE

        self.MAX_Q_DEPTH = 6 # Max depth for quiescence search

        self.EVAL_PIECE_VALUES_LST = [0] * (max(EVAL_PIECE_VALUES_RAW.keys()) + 1)
        for p_type, val in EVAL_PIECE_VALUES_RAW.items():
            self.EVAL_PIECE_VALUES_LST[p_type] = val

        self.SEE_PIECE_VALUES = [0, 100, 320, 330, 500, 900, 20000] # P, N, B, R, Q, K

        self._PIECE_VALUES_LST_NMP = [0, 100, 320, 330, 500, 900, 0] # King 0 for NMP material

        self.NMP_R_REDUCTION = 3
        self.NMP_MIN_DEPTH_THRESHOLD = 1 + self.NMP_R_REDUCTION
        self.NMP_MIN_MATERIAL_FOR_SIDE = self._PIECE_VALUES_LST_NMP[chess.ROOK]

        # --- Move Ordering Bonuses ---
        self.TT_MOVE_ORDER_BONUS = 250000
        self.CAPTURE_BASE_ORDER_BONUS = 100000
        self.QUEEN_PROMO_ORDER_BONUS = 120000
        self.MINOR_PROMO_ORDER_BONUS = 20000
        self.KILLER_1_ORDER_BONUS = 110000
        self.KILLER_2_ORDER_BONUS = 70000
        self.HISTORY_MAX_BONUS = 90000

        # --- SEE Related Bonuses/Penalties for Move Ordering ---
        self.SEE_POSITIVE_BONUS_VALUE = 15000
        self.SEE_NEGATIVE_PENALTY_VALUE = -3500
        self.SEE_VALUE_SCALING_FACTOR = 8
        self.SEE_Q_PRUNING_THRESHOLD = -30 # For quiescence search

        # --- Late Move Reduction (LMR) ---
        self.LMR_MIN_DEPTH = 3
        self.LMR_MIN_MOVES_TRIED = 3
        self.LMR_REDUCTION = 0 # Set to 1 or 2 to activate. Currently deactivated.

        self.init_zobrist_tables()

    def init_zobrist_tables(self):
        random.seed(42) # For reproducible hashes
        self.zobrist_piece_square = [[random.getrandbits(64) for _ in range(12)] for _ in range(64)]
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        self.zobrist_ep = [random.getrandbits(64) for _ in range(8)]
        self.zobrist_side = random.getrandbits(64)

    def compute_hash(self, board: chess.Board) -> int:
        h = 0
        for sq in CHESS_SQUARES:
            piece = board.piece_at(sq)
            if piece:
                piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                h ^= self.zobrist_piece_square[sq][piece_idx]
        h ^= self.zobrist_castling[board.castling_rights & 0xF]
        if board.ep_square is not None:
            h ^= self.zobrist_ep[chess.square_file(board.ep_square)]
        if board.turn == chess.BLACK:
            h ^= self.zobrist_side
        return h

    def get_side_material(self, board: chess.Board, color: chess.Color) -> int: # Used for NMP
        material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            material += len(board.pieces(piece_type, color)) * self._PIECE_VALUES_LST_NMP[piece_type]
        return material

    def get_mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        attacker = board.piece_at(move.from_square)
        victim_piece_at_to_square = board.piece_at(move.to_square)

        if board.is_en_passant(move): victim_piece_type = chess.PAWN
        elif victim_piece_at_to_square: victim_piece_type = victim_piece_at_to_square.piece_type
        else: return 0 # Not a capture
        if not attacker: return 0  # Should not happen

        victim_value = self.EVAL_PIECE_VALUES_LST[victim_piece_type]
        attacker_value = self.EVAL_PIECE_VALUES_LST[attacker.piece_type]
        return victim_value * 10 - attacker_value # Simple MVV-LVA

    def _get_lowest_attacker_see(self, board: chess.Board, to_sq: chess.Square, side: chess.Color) -> Optional[chess.Move]:
        lowest_value = float('inf')
        best_move = None
        attacker_squares = board.attackers(side, to_sq)
        if not attacker_squares: return None

        for from_sq in attacker_squares:
            piece = board.piece_at(from_sq)
            if piece:
                current_value = self.SEE_PIECE_VALUES[piece.piece_type]
                if current_value < lowest_value:
                    lowest_value = current_value
                    promotion_piece = None
                    if piece.piece_type == chess.PAWN and \
                       ((side == chess.WHITE and chess.square_rank(to_sq) == 7) or \
                        (side == chess.BLACK and chess.square_rank(to_sq) == 0)):
                        promotion_piece = chess.QUEEN # Assume queen promotion in SEE
                    best_move = chess.Move(from_sq, to_sq, promotion=promotion_piece)
        return best_move

    def see(self, board: chess.Board, move: chess.Move) -> int: # Static Exchange Evaluation
        target_sq = move.to_square
        initial_attacker_piece = board.piece_at(move.from_square)
        if not initial_attacker_piece: return 0

        if board.is_en_passant(move):
            initial_victim_value = self.SEE_PIECE_VALUES[chess.PAWN]
        else:
            initial_victim_piece = board.piece_at(target_sq)
            if not initial_victim_piece: return 0 # Not a capture
            initial_victim_value = self.SEE_PIECE_VALUES[initial_victim_piece.piece_type]

        see_value_stack = [0] * 32; see_value_stack[0] = initial_victim_value; num_see_moves = 1
        piece_on_target_val = self.SEE_PIECE_VALUES[move.promotion if move.promotion else initial_attacker_piece.piece_type]

        temp_board = board.copy()
        try: temp_board.push(move)
        except AssertionError: return 0

        while num_see_moves < 32:
            recapture_side = temp_board.turn
            recapture_move = self._get_lowest_attacker_see(temp_board, target_sq, recapture_side)
            if not recapture_move: break

            recapturing_piece = temp_board.piece_at(recapture_move.from_square)
            if not recapturing_piece: break

            see_value_stack[num_see_moves] = piece_on_target_val # Value of piece being captured now
            piece_on_target_val = self.SEE_PIECE_VALUES[recapture_move.promotion if recapture_move.promotion else recapturing_piece.piece_type]

            try: temp_board.push(recapture_move)
            except AssertionError: break
            num_see_moves += 1

        score = 0
        for i in range(num_see_moves - 1, -1, -1):
            current_player_net_gain = see_value_stack[i] - score
            if i > 0 and current_player_net_gain < 0: score = 0
            else: score = current_player_net_gain
        return score

    def get_move_score(self, board: chess.Board, move: chess.Move, is_capture: bool, gives_check: bool,
                       tt_move: Optional[chess.Move], ply: int, white_checks_delivered: int, black_checks_delivered: int,
                       max_checks: int, qsearch_mode: bool = False) -> int:
        score = 0
        if not qsearch_mode and tt_move and move == tt_move:
            score += self.TT_MOVE_ORDER_BONUS

        if is_capture:
            score += self.CAPTURE_BASE_ORDER_BONUS + self.get_mvv_lva_score(board, move)
            if not qsearch_mode: # SEE bonus/penalty only in main search ordering
                see_val = self.see(board, move)
                num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + (1 if gives_check else 0)
                is_critical_check_related_capture = gives_check and (num_checks_by_mover_after_move >= max_checks -1)

                if see_val > 0:
                    score += self.SEE_POSITIVE_BONUS_VALUE + see_val * self.SEE_VALUE_SCALING_FACTOR
                elif see_val < 0 and not is_critical_check_related_capture: # Don't penalize critical sacrifices
                         score += self.SEE_NEGATIVE_PENALTY_VALUE + see_val * self.SEE_VALUE_SCALING_FACTOR

        if gives_check:
            num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + 1
            checks_remaining_to_win = max(0, max_checks - num_checks_by_mover_after_move)
            check_bonus_factor = self.CHECK_BONUS_DECAY_BASE_ORDERING ** checks_remaining_to_win
            dynamic_check_bonus = self.CHECK_BONUS_ADDITIONAL_MAX_ORDERING * check_bonus_factor
            total_check_bonus_for_ordering = self.CHECK_BONUS_MIN_ORDERING + dynamic_check_bonus
            score += int(total_check_bonus_for_ordering)

        if not qsearch_mode:
            if move.promotion == chess.QUEEN: score += self.QUEEN_PROMO_ORDER_BONUS
            elif move.promotion: score += self.MINOR_PROMO_ORDER_BONUS

            if not is_capture: # Killer and History only for quiet moves
                if self.killer_moves[ply][0] == move: score += self.KILLER_1_ORDER_BONUS
                elif self.killer_moves[ply][1] == move: score += self.KILLER_2_ORDER_BONUS
                piece = board.piece_at(move.from_square)
                if piece:
                    history_score = self.history_table.get((piece.piece_type, move.to_square), 0)
                    score += min(history_score, self.HISTORY_MAX_BONUS)
        return score

    def order_moves(self, board: chess.Board, legal_moves_generator: chess.LegalMoveGenerator,
                    tt_move: Optional[chess.Move], ply: int,
                    white_checks: int, black_checks: int, max_checks: int,
                    qsearch_mode: bool = False) -> List[Tuple[chess.Move, bool, bool]]:
        moves_to_process_details = []
        for m in legal_moves_generator:
            is_c = board.is_capture(m)
            is_promo = m.promotion is not None
            g_c = board.gives_check(m) # Calculate once, used by both modes

            if qsearch_mode:
                # For q-search: only captures, promotions, or other checks.
                if not is_c and not is_promo and not g_c:
                    continue # Skip non-forcing quiet moves
            moves_to_process_details.append({'move': m, 'is_capture': is_c, 'gives_check': g_c})

        scored_move_data = []
        for move_attrs in moves_to_process_details:
            m = move_attrs['move']; is_c = move_attrs['is_capture']; g_c = move_attrs['gives_check']
            current_score = self.get_move_score(board, m, is_c, g_c, tt_move, ply,
                                               white_checks, black_checks, max_checks, qsearch_mode)
            scored_move_data.append((current_score, m, is_c, g_c))

        scored_move_data.sort(key=lambda x: x[0], reverse=True)
        return [(data[1], data[2], data[3]) for data in scored_move_data]

    def evaluate_position(self, board: chess.Board, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> int:
        if white_checks_delivered >= max_checks: return self.WIN_SCORE
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE

        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE
            return 0 # Draw

        # Heuristic: if opponent is 1 check away, can they deliver it with a free move?
        side_to_move = board.turn
        opponent_checks_count = black_checks_delivered if side_to_move == chess.WHITE else white_checks_delivered
        if opponent_checks_count == max_checks - 1:
            temp_board = board.copy()
            try:
                temp_board.push(chess.Move.null()) # Give turn to opponent
                if temp_board.is_valid() and any(temp_board.gives_check(m) for m in temp_board.legal_moves):
                    return self.LOSS_SCORE # Opponent has a forced Nth check
            except AssertionError: pass # Null move illegal (e.g. current player in check)

        material_positional_score = 0
        for sq in CHESS_SQUARES:
            piece = board.piece_at(sq)
            if piece:
                value = self.EVAL_PIECE_VALUES_LST[piece.piece_type]
                pst_val = PST[piece.piece_type][sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
                if piece.color == chess.WHITE: material_positional_score += value + pst_val
                else: material_positional_score -= (value + pst_val)

        check_bonus_score = 0
        effective_max_checks_for_bonus_calc = max(1, max_checks - 1)
        if EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS > 0 and effective_max_checks_for_bonus_calc > 0:
            if white_checks_delivered > 0:
                ratio_w = min(1.0, white_checks_delivered / effective_max_checks_for_bonus_calc)
                check_bonus_score += int(EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS * (ratio_w ** 2))
            if black_checks_delivered > 0:
                ratio_b = min(1.0, black_checks_delivered / effective_max_checks_for_bonus_calc)
                check_bonus_score -= int(EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS * (ratio_b ** 2))

        current_player_threat_bonus = 0
        if not board.is_game_over(claim_draw=True) and \
           (white_checks_delivered < max_checks and black_checks_delivered < max_checks):
            if any(board.gives_check(m) for m in board.legal_moves):
                current_player_threat_bonus = EVAL_CHECK_THREAT_BONUS

        if board.turn == chess.WHITE: material_positional_score += current_player_threat_bonus
        else: material_positional_score -= current_player_threat_bonus
        return material_positional_score + check_bonus_score

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int, maximizing_player: bool,
                          white_checks: int, black_checks: int, max_checks: int, q_depth: int) -> int:
        self.q_nodes_evaluated += 1
        if white_checks >= max_checks: return self.WIN_SCORE
        if black_checks >= max_checks: return self.LOSS_SCORE
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE
            return 0
        if q_depth <= 0:
            return self.evaluate_position(board, white_checks, black_checks, max_checks)

        stand_pat_score = self.evaluate_position(board, white_checks, black_checks, max_checks)
        if maximizing_player:
            alpha = max(alpha, stand_pat_score) # Corrected: No artifact here
        else:
            beta = min(beta, stand_pat_score)
        if alpha >= beta:
            return beta if maximizing_player else alpha # Or just 'alpha' if beta cutoff in min player

        ordered_forcing_moves_data = self.order_moves(board, board.legal_moves, None, 0,
                                                 white_checks, black_checks, max_checks, qsearch_mode=True)
        for move, is_capture_flag, _ in ordered_forcing_moves_data:
            if is_capture_flag and self.see(board, move) < self.SEE_Q_PRUNING_THRESHOLD:
                continue

            current_w_checks, current_b_checks = white_checks, black_checks
            piece_color_that_moved = board.turn
            board.push(move)
            if board.is_check():
                if piece_color_that_moved == chess.WHITE: current_w_checks += 1
                else: current_b_checks += 1 # Assumes AI is Black if not White

            score = self.quiescence_search(board, alpha, beta, not maximizing_player,
                                       current_w_checks, current_b_checks, max_checks, q_depth - 1)
            board.pop()

            if maximizing_player:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta: break # Cutoff
        return alpha if maximizing_player else beta

    def store_in_tt(self, key: int, depth: int, value: int, flag: int, best_move: Optional[chess.Move]):
        # Simple FIFO replacement if TT is full
        if len(self.transposition_table) >= self.tt_size_limit and self.tt_size_limit > 0 :
            try: self.transposition_table.pop(next(iter(self.transposition_table)))
            except StopIteration: pass
        if self.tt_size_limit > 0 :
             self.transposition_table[key] = {'depth': depth, 'value': value, 'flag': flag, 'best_move': best_move.uci() if best_move else None}

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing_player: bool, ply: int,
                white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> Tuple[int, Optional[chess.Move]]:
        alpha_orig = alpha
        if white_checks_delivered >= max_checks: return self.WIN_SCORE, None
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE, None

        tt_key = self.compute_hash(board); tt_entry = self.transposition_table.get(tt_key); tt_move = None
        if tt_entry:
            if tt_entry['best_move']:
                try: tt_move = board.parse_uci(tt_entry['best_move'])
                except (ValueError, AssertionError): tt_move = None
                if tt_move and not board.is_legal(tt_move): tt_move = None # Ensure legality
            if tt_entry['depth'] >= depth:
                if tt_entry['flag'] == TT_EXACT: return tt_entry['value'], tt_move
                elif tt_entry['flag'] == TT_LOWERBOUND: alpha = max(alpha, tt_entry['value'])
                elif tt_entry['flag'] == TT_UPPERBOUND: beta = min(beta, tt_entry['value'])
                if alpha >= beta: return tt_entry['value'], tt_move # Cutoff from TT

        if depth <= 0:
            self.nodes_evaluated += 1
            return self.quiescence_search(board, alpha, beta, maximizing_player,
                                          white_checks_delivered, black_checks_delivered, max_checks, self.MAX_Q_DEPTH), None

        outcome = board.outcome(claim_draw=True)
        if outcome: # Traditional game end
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE, None
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE, None
            return 0, None # Draw

        in_check_at_node_start = board.is_check()
        # Null Move Pruning (NMP)
        if not in_check_at_node_start and depth >= self.NMP_MIN_DEPTH_THRESHOLD and ply > 0 and \
           self.get_side_material(board, board.turn) >= self.NMP_MIN_MATERIAL_FOR_SIDE:
            try:
                board.push(chess.Move.null())
                null_move_score, _ = self.minimax(board, depth - 1 - self.NMP_R_REDUCTION, -beta, -beta + 1,
                                                not maximizing_player, ply + 1,
                                                white_checks_delivered, black_checks_delivered, max_checks)
                board.pop()
                null_move_score = -null_move_score
                if null_move_score >= beta: return beta, None # Or null_move_score
            except AssertionError: pass # Null move illegal

        best_move_for_node: Optional[chess.Move] = None
        ordered_moves_data = self.order_moves(board, board.legal_moves, tt_move, ply,
                                         white_checks_delivered, black_checks_delivered, max_checks)

        if not ordered_moves_data: # No legal moves
            if in_check_at_node_start: # Checkmate
                return (-self.CHECKMATE_SCORE + ply if maximizing_player else self.CHECKMATE_SCORE - ply), None
            else: return 0, None # Stalemate

        best_val_for_node = -float('inf') if maximizing_player else float('inf')
        moves_searched_count = 0
        for move, is_capture_flag, _ in ordered_moves_data:
            moves_searched_count += 1
            current_w_checks, current_b_checks = white_checks_delivered, black_checks_delivered
            piece_color_that_moved = board.turn
            piece_that_moved_type_before_move: Optional[chess.PieceType] = None
            if not is_capture_flag and move.promotion is None: # For history heuristic
                piece_obj = board.piece_at(move.from_square)
                if piece_obj: piece_that_moved_type_before_move = piece_obj.piece_type

            board.push(move)
            is_check_after_this_move = board.is_check()
            if is_check_after_this_move:
                if piece_color_that_moved == chess.WHITE: current_w_checks += 1
                else: current_b_checks += 1

            eval_val: int; effective_search_depth = depth - 1
            # Late Move Reduction (LMR)
            do_lmr = (self.LMR_REDUCTION > 0 and depth >= self.LMR_MIN_DEPTH and
                      moves_searched_count > self.LMR_MIN_MOVES_TRIED and
                      not in_check_at_node_start and not is_capture_flag and not is_check_after_this_move)

            if do_lmr:
                reduced_depth = max(0, effective_search_depth - self.LMR_REDUCTION)
                eval_val, _ = self.minimax(board, reduced_depth, alpha, beta, not maximizing_player,
                                           ply + 1, current_w_checks, current_b_checks, max_checks)
                if (maximizing_player and eval_val > alpha) or (not maximizing_player and eval_val < beta): # Re-search
                    eval_val, _ = self.minimax(board, effective_search_depth, alpha, beta, not maximizing_player,
                                               ply + 1, current_w_checks, current_b_checks, max_checks)
            else:
                 eval_val, _ = self.minimax(board, effective_search_depth, alpha, beta, not maximizing_player,
                                           ply + 1, current_w_checks, current_b_checks, max_checks)
            board.pop()

            if maximizing_player:
                if eval_val > best_val_for_node: best_val_for_node = eval_val; best_move_for_node = move
                alpha = max(alpha, eval_val)
            else: # Minimizing player
                if eval_val < best_val_for_node: best_val_for_node = eval_val; best_move_for_node = move
                beta = min(beta, eval_val)

            if alpha >= beta: # Cutoff
                if piece_that_moved_type_before_move: # Update history/killer for quiet moves causing cutoff
                    if move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    self.history_table[(piece_that_moved_type_before_move, move.to_square)] = \
                        self.history_table.get((piece_that_moved_type_before_move, move.to_square), 0) + depth**2
                break

        tt_flag = TT_EXACT
        if best_val_for_node <= alpha_orig: tt_flag = TT_UPPERBOUND
        elif best_val_for_node >= beta: tt_flag = TT_LOWERBOUND
        self.store_in_tt(tt_key, depth, best_val_for_node, tt_flag, best_move_for_node)
        return best_val_for_node, best_move_for_node

    def find_best_move(self, board: chess.Board, depth: int, white_checks: int, black_checks: int, max_checks: int) -> Optional[chess.Move]:
        self.nodes_evaluated = 0; self.q_nodes_evaluated = 0
        is_maximizing_player = (board.turn == chess.WHITE)
        best_move_overall: Optional[chess.Move] = None
        final_value_for_best_move = 0 # This will be from AI's perspective

        self.transposition_table.clear()
        self.killer_moves = [[None, None] for _ in range(depth + 10)] # Max depth + buffer
        self.history_table.clear()

        for d_iterative in range(1, depth + 1):
            value, move_at_this_depth = self.minimax(board, d_iterative, -float('inf'), float('inf'),
                                                     is_maximizing_player, 0, # ply = 0 for root
                                                     white_checks, black_checks, max_checks)
            if move_at_this_depth is not None:
                best_move_overall = move_at_this_depth
                final_value_for_best_move = value # Store the value from this depth's search
            elif best_move_overall is None and d_iterative == 1: # No move found at depth 1 (e.g. mate)
                final_value_for_best_move = value # Use the evaluation

            # Update internal AI eval. This value is always from White's perspective,
            # as minimax returns scores on White's POV scale.
            self.current_eval_for_ui = final_value_for_best_move

            # Early exit if mate is found
            should_stop = False
            if is_maximizing_player:
                 if value >= self.WIN_SCORE or value >= (self.CHECKMATE_SCORE - d_iterative * 10): should_stop = True
            else: # Minimizing player
                 if value <= self.LOSS_SCORE or value <= (-self.CHECKMATE_SCORE + d_iterative * 10): should_stop = True
            if should_stop: break

        # Fallback if no move found but legal moves exist
        if best_move_overall is None and list(board.legal_moves) and \
           white_checks < max_checks and black_checks < max_checks and \
           not board.is_game_over(claim_draw=True):
            print("AI Warning: find_best_move returned None but legal moves exist. Picking first legal move.")
            best_move_overall = next(iter(board.legal_moves), None)
            # If fallback, update current_eval_for_ui with a static eval (which is White's POV)
            self.current_eval_for_ui = self.evaluate_position(board, white_checks, black_checks, max_checks)

        return best_move_overall

class ChessUI:
    def __init__(self):
        self.square_size = INITIAL_SQUARE_SIZE
        self.root = tk.Tk()
        self.root.title(f"{NUM_CHECKS_TO_WIN}-Check Chess")
        self.root.geometry(f"{INITIAL_WINDOW_WIDTH}x{INITIAL_WINDOW_HEIGHT}")
        self.root.tk_setPalette(background='#E0E0E0')

        self.base_font_family = "Arial"
        self.base_font = font.Font(family=self.base_font_family, size=10)
        self.bold_font = font.Font(family=self.base_font_family, size=11, weight="bold")
        self.piece_font_family = "Segoe UI Symbol" # For unicode chess pieces

        self.board = chess.Board()
        self.ai_depth = default_ai_depth
        self.flipped = False
        self.game_over_flag = False
        self.ai_thinking = False

        self.dragging_piece_item_id: Optional[int] = None
        self.drag_start_x_offset: int = 0
        self.drag_start_y_offset: int = 0
        self.drag_selected_square: Optional[chess.Square] = None

        self.piece_item_ids: Dict[chess.Square, int] = {}
        self.square_item_ids: Dict[chess.Square, int] = {}
        self.original_square_colors: Dict[int, str] = {} # Stores original color of highlighted square
        self.highlighted_rect_id: Optional[int] = None

        self.piece_images: Dict[str, str] = {}
        self.ai = ChessAI()
        self.ai_move_queue = queue.Queue() # For AI moves from thread to main UI

        self.colors = {
            'square_light': '#E0C4A0', 'square_dark': '#A08060',
            'highlight_selected_fill': '#6FA8DC',
            'highlight_capture': '#FF7F7F80', # RGBA-like, stipple will use RGB
            'highlight_normal_move': '#90EE9080', # RGBA-like, stipple will use RGB
            'piece_white': '#FFFFFF', 'piece_black': '#202020',
            'piece_outline': '#444444'
        }
        self.stipple_patterns = {'light': 'gray25', 'medium': 'gray50', 'heavy': 'gray75'}

        self.MAX_CHECKS = NUM_CHECKS_TO_WIN
        self.white_checks_delivered = 0
        self.black_checks_delivered = 0
        self.check_history: List[Tuple[int, int]] = []

        self._setup_ui_styles()
        self._setup_ui()
        self.load_piece_images()
        self.root.update_idletasks()
        self.reset_game()
        self.root.resizable(True, True)
        self._check_ai_queue_periodically()

    def _setup_ui_styles(self):
        s = ttk.Style(); s.theme_use('clam')
        s.configure("TButton", font=self.base_font, padding=(10, 8))
        s.configure("TLabel", font=self.base_font, padding=3)
        s.configure("TScale", troughcolor='#D8D8D8',sliderlength=25)
        s.configure("TLabelframe", font=self.bold_font, padding=(5,3))
        s.configure("TLabelframe.Label", font=self.bold_font)

    def _setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main_frame, borderwidth=1, relief="sunken", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(5,0), pady=5)
        main_frame.grid_columnconfigure(0, weight=1); main_frame.grid_rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_square_interaction_start)
        self.canvas.bind("<B1-Motion>", self.on_piece_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_square_interaction_end)

        control_panel = ttk.Frame(main_frame, padding=(10,5,8,5), width=CONTROL_PANEL_DEFAULT_WIDTH)
        control_panel.grid(row=0, column=1, sticky="ns", padx=(8,5)); control_panel.grid_propagate(False)
        row_idx = 0; btn_width = 22

        ttk.Button(control_panel, text="New Game", command=self.reset_game, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        ttk.Button(control_panel, text="Flip Board", command=self.flip_board, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        ttk.Button(control_panel, text="Undo", command=self.undo_last_player_ai_moves, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1

        self.checks_labelframe = ttk.LabelFrame(control_panel, text=f"Checks (Goal: {self.MAX_CHECKS})")
        self.checks_labelframe.grid(row=row_idx, column=0,columnspan=2, pady=(8,5), sticky="ew"); row_idx+=1
        inner_checks_frame = ttk.Frame(self.checks_labelframe); inner_checks_frame.pack(fill=tk.X, expand=True, padx=4, pady=(2,4))
        inner_checks_frame.columnconfigure(0, weight=1); inner_checks_frame.columnconfigure(1, weight=1)
        self.white_checks_label = ttk.Label(inner_checks_frame, text="W:", font=(self.piece_font_family, 11), anchor="w")
        self.white_checks_label.grid(row=0, column=0, sticky="ew", padx=(3,2))
        self.black_checks_label = ttk.Label(inner_checks_frame, text="B:", font=(self.piece_font_family, 11), anchor="w")
        self.black_checks_label.grid(row=0, column=1, sticky="ew", padx=(2,3))

        depth_frame = ttk.Frame(control_panel); depth_frame.grid(row=row_idx, column=0, columnspan=2, pady=(5,0), sticky="ew"); row_idx+=1
        ttk.Label(depth_frame, text="AI Depth:", font=self.base_font).pack(side=tk.LEFT, padx=(0,5))
        self.depth_value_label = ttk.Label(depth_frame, text=str(self.ai_depth), font=self.base_font, width=2, anchor="e")
        self.depth_value_label.pack(side=tk.RIGHT, padx=(5,0))
        self.depth_slider = ttk.Scale(control_panel, from_=1, to=6, orient=tk.HORIZONTAL, command=self.update_ai_depth)
        self.depth_slider.set(self.ai_depth); self.depth_slider.grid(row=row_idx, column=0, columnspan=2, pady=(0,5), sticky="ew"); row_idx+=1

        self.status_label = ttk.Label(control_panel, text="White's turn", font=self.bold_font, wraplength=CONTROL_PANEL_DEFAULT_WIDTH-20, anchor="center", justify="center", width=26)
        self.status_label.grid(row=row_idx, column=0,columnspan=2, pady=(4,2), sticky="ew", ipady=3); row_idx+=1
        self.eval_label = ttk.Label(control_panel, text="Eval: +0.00", font=self.base_font, wraplength=CONTROL_PANEL_DEFAULT_WIDTH-20, anchor="center", justify="center", width=26)
        self.eval_label.grid(row=row_idx, column=0, columnspan=2, pady=(0,8), sticky="ew", ipady=2); row_idx+=1
        ttk.Button(control_panel, text="Exit", command=self.root.quit, width=btn_width).grid(row=row_idx, column=0,columnspan=2, pady=4, sticky="ew"); row_idx+=1
        control_panel.grid_rowconfigure(row_idx, weight=1)

    def load_piece_images(self):
        self.piece_images = {'R':'♖','N':'♘','B':'♗','Q':'♕','K':'♔','P':'♙','r':'♜','n':'♞','b':'♝','q':'♛','k':'♚','p':'♟'}

    def on_canvas_resize(self, event=None):
        w = event.width if event else self.canvas.winfo_width(); h = event.height if event else self.canvas.winfo_height()
        if w < 16 or h < 16: return # Avoid issues with tiny canvas
        new_sq_size = max(10, min(w // 8, h // 8))
        if new_sq_size != self.square_size or event is None: # Redraw if size changed or forced (event=None)
            self.square_size = new_sq_size
            self.redraw_board_and_pieces()

    def _get_square_coords(self, sq: chess.Square) -> Tuple[int,int,int,int]:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        # Adjust for flipped board and typical screen coordinates (y=0 at top)
        f_display = 7-f if self.flipped else f
        r_display = r if self.flipped else 7-r # If flipped, rank 0 is at bottom. If not, rank 7 is at top.
        return f_display*self.square_size, r_display*self.square_size, (f_display+1)*self.square_size, (r_display+1)*self.square_size

    def _get_canvas_xy_to_square(self, x:int, y:int) -> Optional[chess.Square]:
        if self.square_size == 0: return None
        f_idx,r_idx = x//self.square_size, y//self.square_size # Canvas column and row index
        if not(0<=f_idx<8 and 0<=r_idx<8): return None

        # Convert canvas grid index back to chess file/rank
        f = 7-f_idx if self.flipped else f_idx
        r = r_idx if self.flipped else 7-r_idx
        return chess.square(f,r) if (0 <= f < 8 and 0 <= r < 8) else None

    def _draw_squares(self):
        self.canvas.delete("square_bg"); self.square_item_ids.clear()
        for r_chess in range(8): # chess ranks 0-7
            for f_chess in range(8): # chess files 0-7
                sq = chess.square(f_chess,r_chess); x1,y1,x2,y2 = self._get_square_coords(sq)
                color = self.colors['square_light'] if (f_chess+r_chess)%2==0 else self.colors['square_dark']
                item_id = self.canvas.create_rectangle(x1,y1,x2,y2,fill=color,outline="",width=0,tags=("square_bg",f"sq_bg_{sq}"))
                self.square_item_ids[sq] = item_id

    def _draw_all_pieces(self):
        self.canvas.delete("piece_outline", "piece_fg"); self.piece_item_ids.clear() # Clear old piece items
        font_size = int(self.square_size * 0.73);
        if font_size < 6: return # Too small to draw
        outline_offset = max(1, int(font_size * 0.03))

        for sq in CHESS_SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                x1,y1,x2,y2 = self._get_square_coords(sq); cx,cy = (x1+x2)//2, (y1+y2)//2
                piece_symbol_str = self.piece_images[piece.symbol()]
                pc = self.colors['piece_white'] if piece.color == chess.WHITE else self.colors['piece_black']
                oc = self.colors['piece_black'] if piece.color == chess.WHITE else self.colors['piece_white']
                # Draw outline first
                self.canvas.create_text(cx+outline_offset, cy+outline_offset, text=piece_symbol_str, font=(self.piece_font_family, font_size), fill=oc, tags=("piece_outline", f"piece_outline_{sq}"), anchor="c")
                # Draw main piece
                item_id = self.canvas.create_text(cx,cy,text=piece_symbol_str,font=(self.piece_font_family,font_size),fill=pc,tags=("piece_fg",f"piece_fg_{sq}"),anchor="c")
                self.piece_item_ids[sq] = item_id

    def redraw_board_and_pieces(self, for_ai_move_eval: Optional[int] = None):
        self._draw_squares(); self._draw_all_pieces()
        self._update_check_display()
        self.update_status_label(for_ai_move_eval=for_ai_move_eval)

    def _clear_highlights(self):
        if self.highlighted_rect_id and self.highlighted_rect_id in self.original_square_colors:
            try: self.canvas.itemconfig(self.highlighted_rect_id, fill=self.original_square_colors.pop(self.highlighted_rect_id))
            except (tk.TclError, KeyError) : pass # Item might be gone or key already removed
        self.highlighted_rect_id = None
        self.canvas.delete("highlight_dot") # Delete all move indicator dots

    def _highlight_square_selected(self, sq: chess.Square):
        item_id = self.square_item_ids.get(sq)
        if item_id:
            try:
                original_color = self.canvas.itemcget(item_id,"fill")
                if item_id not in self.original_square_colors: # Store original only once
                    self.original_square_colors[item_id] = original_color
                self.canvas.itemconfig(item_id, fill=self.colors['highlight_selected_fill'])
                self.highlighted_rect_id = item_id
                self.canvas.tag_raise("piece_outline") # Ensure pieces are above highlight
                self.canvas.tag_raise("piece_fg")
            except tk.TclError: self.highlighted_rect_id = None # Item might not exist

    def _highlight_legal_move(self, sq: chess.Square, is_capture: bool):
        x1,y1,x2,y2 = self._get_square_coords(sq); cx,cy=(x1+x2)//2,(y1+y2)//2
        radius = self.square_size*0.15
        fill_color_rgb = (self.colors['highlight_capture'] if is_capture else self.colors['highlight_normal_move'])[:7] # Get RGB part
        self.canvas.create_oval(cx-radius,cy-radius,cx+radius,cy+radius, fill=fill_color_rgb, stipple=self.stipple_patterns['medium'], outline="", tags="highlight_dot")

    def _show_legal_moves_for_selected_piece(self):
        if self.drag_selected_square is not None:
            self._highlight_square_selected(self.drag_selected_square)
            for m in self.board.legal_moves:
                if m.from_square == self.drag_selected_square:
                    self._highlight_legal_move(m.to_square, self.board.is_capture(m))

    def _update_check_display(self):
        filled_check_char = '✚'; empty_check_char = '⊕'
        max_symbols_to_show = self.MAX_CHECKS if self.MAX_CHECKS <= 7 else 7

        w_filled = self.white_checks_delivered; w_empty = max(0, min(max_symbols_to_show - w_filled, self.MAX_CHECKS - w_filled))
        white_text = "W: " + (filled_check_char * w_filled) + (empty_check_char * w_empty)
        if self.MAX_CHECKS - w_filled > w_empty and w_filled + w_empty >= max_symbols_to_show: white_text += ".."

        b_filled = self.black_checks_delivered; b_empty = max(0, min(max_symbols_to_show - b_filled, self.MAX_CHECKS - b_filled))
        black_text = "B: " + (filled_check_char * b_filled) + (empty_check_char * b_empty)
        if self.MAX_CHECKS - b_filled > b_empty and b_filled + b_empty >= max_symbols_to_show: black_text += ".."

        self.white_checks_label.config(text=white_text.strip())
        self.black_checks_label.config(text=black_text.strip())

    def update_status_label(self, for_ai_move_eval: Optional[int] = None):
        status_text = ""; eval_text = ""

        # Determine game status text first
        if self.game_over_flag: # If already flagged, try to use existing win/draw message
            current_status_from_label = self.status_label.cget("text").split("\n")[0]
            if any(term in current_status_from_label for term in ["wins", "Draw", "Checkmate"]):
                 status_text = current_status_from_label
        if not status_text: # If not set by above, determine game status now
            if self.white_checks_delivered >= self.MAX_CHECKS:
                status_text = f"White wins by {self.MAX_CHECKS} checks!"; self.game_over_flag = True
            elif self.black_checks_delivered >= self.MAX_CHECKS:
                status_text = f"Black wins by {self.MAX_CHECKS} checks!"; self.game_over_flag = True
            else:
                outcome = self.board.outcome(claim_draw=True)
                if outcome:
                    self.game_over_flag = True
                    if outcome.winner == chess.WHITE: status_text = "Checkmate! White wins!"
                    elif outcome.winner == chess.BLACK: status_text = "Checkmate! Black wins!"
                    else: status_text = f"Draw! ({outcome.termination.name.replace('_',' ').title()})"
                else: # Game is ongoing
                    status_text = ("White's turn" if self.board.turn == chess.WHITE else "Black's turn")
                    if self.board.is_check(): status_text += " (Check!)"
                    self.game_over_flag = False

        # Determine evaluation text, ensuring it's always from White's perspective for UI consistency.
        score_to_display_white_pov = 0 # Initialize

        if self.ai_thinking and for_ai_move_eval is None: # AI is thinking, no eval from current search yet
            eval_text = "Eval: Thinking..."
        else:
            # All other cases will determine score_to_display_white_pov from White's perspective.
            if self.game_over_flag:
                if self.white_checks_delivered >= self.MAX_CHECKS: score_to_display_white_pov = self.ai.WIN_SCORE
                elif self.black_checks_delivered >= self.MAX_CHECKS: score_to_display_white_pov = self.ai.LOSS_SCORE
                elif self.board.is_checkmate():
                     # If it's Black's turn, White delivered checkmate. Score is positive (good for White).
                     # If it's White's turn, Black delivered checkmate. Score is negative (bad for White).
                     score_to_display_white_pov = self.ai.CHECKMATE_SCORE if self.board.turn == chess.BLACK else -self.ai.CHECKMATE_SCORE
                else: score_to_display_white_pov = 0 # Draw
            elif for_ai_move_eval is not None: # Evaluation provided from AI search (after AI moved)
                # for_ai_move_eval is self.ai.current_eval_for_ui, which is already White's POV.
                score_to_display_white_pov = for_ai_move_eval
            else: # Game ongoing (e.g. player's turn or initial state), get fresh static eval for current board.
                # self.ai.evaluate_position always returns White's POV.
                score_to_display_white_pov = self.ai.evaluate_position(self.board, self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS)

            # Format eval_text based on score_to_display_white_pov (which is now consistently White's POV)
            if score_to_display_white_pov == 0 and self.game_over_flag and \
               not (self.white_checks_delivered >= self.MAX_CHECKS or self.black_checks_delivered >= self.MAX_CHECKS or self.board.is_checkmate()):
                 eval_text = "Eval: +0.00 (Draw)"
            elif abs(score_to_display_white_pov) >= self.ai.WIN_SCORE:
                 eval_text = "Eval: White winning" if score_to_display_white_pov > 0 else "Eval: Black winning"
            elif abs(score_to_display_white_pov) >= (self.ai.CHECKMATE_SCORE - (self.ai_depth * 20)): # Heuristic for imminent mate
                 eval_text = "Eval: Mate imminent" # Could be refined to "White mates" or "Black mates"
            else:
                eval_text = f"Eval: {score_to_display_white_pov / 100.0:+.2f}"

        if hasattr(self, "status_label") and status_text != self.status_label.cget("text"):
            self.status_label.config(text=status_text)
        if hasattr(self, "eval_label") and eval_text != self.eval_label.cget("text"):
            self.eval_label.config(text=eval_text)

    def reset_game(self):
        self.board.reset()
        self.game_over_flag=False; self.MAX_CHECKS=NUM_CHECKS_TO_WIN
        if hasattr(self,'root'): self.root.title(f"{self.MAX_CHECKS}-Check Chess")
        if hasattr(self,'checks_labelframe'): self.checks_labelframe.config(text=f"Checks (Goal: {self.MAX_CHECKS})")
        self.white_checks_delivered=0; self.black_checks_delivered=0; self.check_history.clear()
        self.ai.transposition_table.clear(); self.ai.killer_moves = [[None,None] for _ in range(self.ai_depth + 10)]; self.ai.history_table.clear()
        self.drag_selected_square=None; self.ai_thinking=False
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
        self._clear_highlights();
        if hasattr(self,'depth_slider'): self.depth_slider.set(self.ai_depth)
        self.update_ai_depth(self.ai_depth) # This calls update_status_label
        self.on_canvas_resize() # This also calls redraw/update_status

    def flip_board(self):
        self.flipped=not self.flipped; self._clear_highlights(); self.drag_selected_square = None
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
        self.redraw_board_and_pieces() # Redraws and updates status

    def undo_last_player_ai_moves(self):
        if self.ai_thinking: return
        num_to_undo = 0
        if len(self.board.move_stack) >= 2 and len(self.check_history) >= 2 : num_to_undo = 2
        elif len(self.board.move_stack) == 1 and len(self.check_history) >= 1: num_to_undo = 1

        for _ in range(num_to_undo):
            if self.board.move_stack and self.check_history:
                self.board.pop(); self.white_checks_delivered, self.black_checks_delivered = self.check_history.pop()
            else: break # Should not happen if num_to_undo is calculated correctly
        if num_to_undo > 0:
            self.game_over_flag=False; self._clear_highlights(); self.drag_selected_square=None
            if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
            self.ai.transposition_table.clear(); self.ai.killer_moves = [[None,None] for _ in range(self.ai_depth + 10)]; self.ai.history_table.clear()
            self.redraw_board_and_pieces() # This will call update_status_label for fresh eval

    def update_ai_depth(self, val_str): # val_str from slider is string
        val_int = int(round(float(val_str)))
        if self.ai_depth != val_int:
            self.ai_depth=val_int; self.ai.transposition_table.clear(); self.ai.killer_moves = [[None,None] for _ in range(val_int + 10)]; self.ai.history_table.clear()
        if hasattr(self,'depth_slider') and abs(self.depth_slider.get()-val_int)>0.01: self.depth_slider.set(val_int)
        if hasattr(self,'depth_value_label'): self.depth_value_label.config(text=str(val_int))
        self.update_status_label() # Update eval display if depth changes strategy

    def on_square_interaction_start(self,event):
        if self.game_over_flag or self.board.turn!=chess.WHITE or self.ai_thinking: return
        clicked_sq = self._get_canvas_xy_to_square(event.x,event.y)
        if clicked_sq is None : return

        if self.drag_selected_square is not None: # A piece was already selected
            if self.drag_selected_square != clicked_sq: # Clicked on a new square - attempt to move
                 self._attempt_player_move(self.drag_selected_square,clicked_sq); return
            else: # Clicked on the same selected square - deselect it
                 self._clear_highlights(); self.drag_selected_square = None; self.dragging_piece_item_id = None
                 self.redraw_board_and_pieces(); return # Redraw to snap piece back

        self._clear_highlights() # Clear previous highlights
        piece = self.board.piece_at(clicked_sq)
        if piece and piece.color==self.board.turn: # Player selected their own piece
            self.drag_selected_square = clicked_sq; self.dragging_piece_item_id = self.piece_item_ids.get(clicked_sq)
            if self.dragging_piece_item_id:
                self.canvas.lift(self.dragging_piece_item_id); self.canvas.lift(f"piece_outline_{clicked_sq}") # Lift outline too
                coords = self.canvas.coords(self.dragging_piece_item_id)
                if coords: self.drag_start_x_offset, self.drag_start_y_offset = event.x-coords[0], event.y-coords[1]
            self._show_legal_moves_for_selected_piece()
        else: # Clicked on empty square or opponent's piece
            self.drag_selected_square=None; self.dragging_piece_item_id=None

    def on_piece_drag_motion(self,event):
        if self.dragging_piece_item_id and self.drag_selected_square is not None and not self.ai_thinking:
            nx,ny = event.x-self.drag_start_x_offset, event.y-self.drag_start_y_offset
            self.canvas.coords(self.dragging_piece_item_id,nx,ny)
            # Also move outline if it exists
            outline_tags = self.canvas.find_withtag(f"piece_outline_{self.drag_selected_square}")
            if outline_tags: # Should find one
                font_size = int(self.square_size * 0.73); outline_offset_val = max(1, int(font_size * 0.03))
                self.canvas.coords(outline_tags[0],nx + outline_offset_val, ny + outline_offset_val)


    def on_square_interaction_end(self,event):
        if self.drag_selected_square is None or self.ai_thinking:
            if self.dragging_piece_item_id and not self.ai_thinking : # Piece was dragged but not properly dropped, or AI started
                self._clear_highlights(); self.redraw_board_and_pieces() # Snap back
            self.drag_selected_square = None; self.dragging_piece_item_id = None; return

        to_sq = self._get_canvas_xy_to_square(event.x,event.y)
        from_sq_before_attempt = self.drag_selected_square # Store before resetting
        self.dragging_piece_item_id = None; self.drag_selected_square = None # Reset drag state early

        move_successfully_made = False
        if to_sq is not None and from_sq_before_attempt is not None and to_sq != from_sq_before_attempt:
            move_successfully_made = self._attempt_player_move(from_sq_before_attempt, to_sq)
        if not move_successfully_made:
            self._clear_highlights(); self.redraw_board_and_pieces() # Redraw to snap piece back if move failed

    def _attempt_player_move(self,from_sq:chess.Square,to_sq:chess.Square) -> bool:
        promo=None; p=self.board.piece_at(from_sq)
        if p and p.piece_type==chess.PAWN:
            if (p.color==chess.WHITE and chess.square_rank(to_sq)==7) or \
               (p.color==chess.BLACK and chess.square_rank(to_sq)==0): promo=chess.QUEEN
        move=chess.Move(from_sq,to_sq,promotion=promo)
        self._clear_highlights() # Clear selection/move dots before checking legality

        if move in self.board.legal_moves:
            self.check_history.append((self.white_checks_delivered,self.black_checks_delivered))
            self.board.push(move)
            if self.board.is_check() and self.board.turn == chess.BLACK: self.white_checks_delivered+=1 # White (player) delivered check
            self.redraw_board_and_pieces() # This will show static eval for AI's turn
            if not self.game_over_flag:
                self.root.update_idletasks(); self._start_ai_move_thread()
            return True
        return False

    def _start_ai_move_thread(self):
        if self.ai_thinking or self.game_over_flag: return
        self.ai_thinking = True; self.status_label.config(text="AI is thinking..."); self.eval_label.config(text="Eval: Thinking...")
        if hasattr(self, 'depth_slider'): self.depth_slider.config(state=tk.DISABLED)
        ai_thread = threading.Thread(target=self._ai_move_worker,
            args=(self.board.copy(), self.ai_depth, self.white_checks_delivered,
                  self.black_checks_delivered, self.MAX_CHECKS), daemon=True)
        ai_thread.start()

    def _ai_move_worker(self, board_copy: chess.Board, depth: int, w_checks: int, b_checks: int, max_c: int):
        start_time = time.monotonic()
        ai_move = self.ai.find_best_move(board_copy, depth, w_checks, b_checks, max_c)
        eval_from_ai_perspective = self.ai.current_eval_for_ui # This is the score from AI's minimax, consistently White's POV.
        elapsed = time.monotonic()-start_time
        total_nodes = self.ai.nodes_evaluated + self.ai.q_nodes_evaluated
        nps = total_nodes/elapsed if elapsed>0 else 0
        tt_fill_percent = (len(self.ai.transposition_table) / self.ai.tt_size_limit) * 100 if self.ai.tt_size_limit > 0 else 0
        ai_player_color_char = 'White' if board_copy.turn == chess.WHITE else 'Black'
        move_uci_str = ai_move.uci() if ai_move else "None (no move)"

        # The evaluation from self.ai.current_eval_for_ui (which is eval_from_ai_perspective here)
        # is always from White's perspective (positive = good for White, negative = good for Black),
        # as the AI's evaluation functions and minimax are structured this way.
        # Therefore, eval_from_ai_perspective can be directly used as eval_for_console_white_pov.
        eval_for_console_white_pov = eval_from_ai_perspective
        
        print_info = (f"AI playing ({ai_player_color_char}) | Move: {move_uci_str} | Eval (White's POV): {eval_for_console_white_pov/100.0:+.2f} | "
                      f"Time: {elapsed:.2f}s | Depth: {depth} | Nodes: {total_nodes} (NPS: {nps:.0f}) | TT Fill: {tt_fill_percent:.1f}%")
        # Pass AI's original perspective eval (which is White's POV) to the queue for UI update
        self.ai_move_queue.put((ai_move, print_info, eval_from_ai_perspective))

    def _check_ai_queue_periodically(self):
        try:
            ai_move, print_info, eval_score_ai_persp = self.ai_move_queue.get_nowait()
            self._process_ai_move_from_queue(ai_move, print_info, eval_score_ai_persp)
        except queue.Empty: pass
        finally: self.root.after(100, self._check_ai_queue_periodically)

    def _process_ai_move_from_queue(self, ai_move: Optional[chess.Move], print_info: str, eval_score_ai_persp: int):
        self.ai_thinking = False
        if hasattr(self, 'depth_slider'): self.depth_slider.config(state=tk.NORMAL)
        print(print_info) # Print detailed info from AI worker

        if self.game_over_flag: # Game might have ended for other reasons (e.g. player disconnect if online)
            self.update_status_label(); return

        if ai_move:
            if ai_move in self.board.legal_moves: # Double check legality on current board state
                self.check_history.append((self.white_checks_delivered,self.black_checks_delivered))
                ai_color_before_move = self.board.turn # AI's color
                self.board.push(ai_move)
                # Check if AI's move delivered a check
                if self.board.is_check() and self.board.turn == chess.WHITE and ai_color_before_move == chess.BLACK: # AI is Black
                    self.black_checks_delivered+=1
                # If AI was White, its check delivery would make it Black's turn, player handles their checks.
            else: # AI suggested an illegal move for the current board state
                print(f"AI Warning: Proposed move {ai_move.uci()} is illegal on current board state!")
                if not list(self.board.legal_moves): self.game_over_flag = True # If no legal moves for current player
            # Redraw and show the AI's search evaluation for the new position
            self.redraw_board_and_pieces(for_ai_move_eval=eval_score_ai_persp)
        else: # AI returned no move.
            print("AI returned no move. Game likely over or AI error.")
            self.game_over_flag = True
            self.update_status_label() # Update based on current board state (likely mate/stalemate)

if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()