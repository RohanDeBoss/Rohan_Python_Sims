import tkinter as tk
from tkinter import ttk, font
import chess
import time
import random
from typing import Tuple, List, Optional, Dict, Any
import threading
import queue

# N-Check Chess v1.1

# --- Global Configuration ---
default_ai_depth = 5
NUM_CHECKS_TO_WIN = 3
INITIAL_SQUARE_SIZE = 66
INITIAL_WINDOW_WIDTH = (8 * INITIAL_SQUARE_SIZE) + 290
INITIAL_WINDOW_HEIGHT = (8 * INITIAL_SQUARE_SIZE) + 29
CONTROL_PANEL_DEFAULT_WIDTH = INITIAL_WINDOW_WIDTH - (8 * INITIAL_SQUARE_SIZE) - 37
TT_SIZE_POWER_OF_2 = 17 # 2^17 = 131,072 entries

# --- AI Evaluation Constants ---
EVAL_WIN_SCORE = 1000000 # Score for N-check win
EVAL_LOSS_SCORE = -1000000 # Score for N-check loss
EVAL_CHECKMATE_SCORE = 900000 # Score for traditional checkmate

# --- Updated piece values for N-Check Chess ---
EVAL_PIECE_VALUES_RAW = {
    chess.PAWN: 100,
    chess.KNIGHT: 380, # Increased value due to non-blockable checks
    chess.BISHOP: 330, # Baseline value
    chess.ROOK: 525,   # Increased value for open-line checks
    chess.QUEEN: 1050, # Greatly increased value for multi-directional check threats
    chess.KING: 0      # King value for material count, not game eval
}

EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS = 700 # Bonus for being close to N-check win
EVAL_CHECK_BONUS_DECAY_RATE = 0.5 # Decay rate for check bonus (e.g., for N-2, N-3 checks).
EVAL_CHECK_THREAT_BONUS = 300 # Bonus for having a checking move available

# These tables are in the standard A1->H8 format (rank 1 at bottom, rank 8 at top).
ORIGINAL_PSTS_RAW = {
    chess.PAWN: [
        0,   0,   0,   0,   0,   0,   0,   0,
       50,  50,  50,  50,  50,  50,  50,  50,
       10,  10,  20,  30,  30,  20,  10,  10,
        5,   5,  10,  25,  25,  10,   5,   5,
        0,   0,   0,  20,  25,   0,   0,   0,
        5,  -5,  -5,   0,   0, -10,  -5,   0,
        5,  10,  10, -20, -20,  10,  10,   5,
        0,   0,   0,   0,   0,   0,   0,   0
    ],
    chess.KNIGHT: [
      -50, -40, -30, -30, -30, -30, -40, -50,
      -40, -20,   0,   0,   0,   0, -20, -40,
      -30,   0,  10,  15,  15,  10,   0, -30,
      -30,   5,  15,  20,  20,  15,   5, -30,
      -30,   0,  15,  20,  20,  15,   0, -30,
      -30,   5,  10,  15,  15,  10,   5, -30,
      -40, -20,   0,   5,   5,   0, -20, -40,
      -50, -40, -30, -30, -30, -30, -40, -50
    ],
    chess.BISHOP: [
      -20, -10, -10, -10, -10, -10, -10, -20,
      -10,   0,   0,   0,   0,   0,   0, -10,
      -10,   0,   5,  10,  10,   5,   0, -10,
      -10,   5,   5,  10,  10,   5,   5, -10,
      -10,   0,  10,  10,  10,  10,   0, -10,
      -10,  10,  10,  10,  10,  10,  10, -10,
      -10,   5,   0,   0,   0,   0,   5, -10,
      -20, -10, -10, -10, -10, -10, -10, -20
    ],
    chess.ROOK: [
        0,   0,   0,   0,   0,   0,   0,   0,
        5,   15,  20,  20,  20,  20,  15,  5,
       -5,   0,   0,   0,   0,   0,   0,  -5,
        0,   0,   0,   0,   0,   0,   0,   0,
       -5,   0,   0,   0,   0,   0,   0,  -5,
       -5,   0,   0,   0,   0,   0,   0,  -5,
       -5,   0,   0,   0,   0,   0,   0,  -5,
        0,   0,   0,   5,   5,   0,   0,   0
    ],
    chess.QUEEN: [
      -20, -10, -10,  -5,  -5, -10, -10, -20,
      -10,   0,   0,   0,   0,   0,   0, -10,
      -10,   0,   5,   5,   5,   5,   0, -10,
       -5,   0,   5,  10,  10,   5,   0,  -5,
        0,   0,   5,   5,   5,   5,   0,  -5,
      -10,   5,   5,   5,   5,   5,   0, -10,
      -10,   0,   5,   0,   0,   0,   0, -10,
      -20, -10, -10,  -5,  -5, -10, -10, -20
    ],
    chess.KING: [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     0,   0,   0,  -10,  -10,  0,  10,  -5,
    10,  20,  10,   0,   10,   5,  40,  20
    ]
}

# Reverse PSTs for python-chess square indexing (A1=0, H8=63)
PST = {piece_type: list(reversed(values)) for piece_type, values in ORIGINAL_PSTS_RAW.items()}


# Transposition Table Flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2
CHESS_SQUARES = list(chess.SQUARES) # Precompute for iteration

class GameState:
    """ Manages the core state of the chess game, including board, check counts, and history. """
    def __init__(self, max_checks: int):
        self.MAX_CHECKS = max_checks
        self.board: chess.Board = chess.Board()
        self.white_checks_delivered: int = 0
        self.black_checks_delivered: int = 0
        self.check_history: List[Tuple[int, int]] = []

    def make_move(self, move: chess.Move) -> bool:
        """
        Applies a move to the board, updates check counts, and records history.
        Returns True if the move was legal and made, False otherwise.
        """
        if move in self.board.legal_moves:
            # Record current state for potential undo
            self.check_history.append((self.white_checks_delivered, self.black_checks_delivered))
            
            player_color = self.board.turn
            self.board.push(move)
            
            # Update check count if the move resulted in a check
            if self.board.is_check():
                if player_color == chess.WHITE:
                    self.white_checks_delivered += 1
                else:
                    self.black_checks_delivered += 1
            return True
        return False

    def undo_move(self):
        """ Reverts the last move and restores the previous check counts. """
        if self.board.move_stack:
            self.board.pop()
            if self.check_history:
                self.white_checks_delivered, self.black_checks_delivered = self.check_history.pop()

    def reset(self):
        """ Resets the game to the initial state. """
        self.board.reset()
        self.white_checks_delivered = 0
        self.black_checks_delivered = 0
        self.check_history.clear()

    def get_outcome(self) -> Optional[chess.Outcome]:
        """ Determines the game outcome, considering N-Check wins first. """
        if self.white_checks_delivered >= self.MAX_CHECKS:
            # Create a custom outcome for N-Check win for white
            return chess.Outcome(termination=chess.Termination.VARIANT_WIN, winner=chess.WHITE)
        if self.black_checks_delivered >= self.MAX_CHECKS:
            # Create a custom outcome for N-Check win for black
            return chess.Outcome(termination=chess.Termination.VARIANT_WIN, winner=chess.BLACK)
        
        # Return standard chess outcome (checkmate, stalemate, etc.)
        return self.board.outcome(claim_draw=True)

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

        self.MAX_Q_DEPTH = 8
        self.LMR_REDUCTION = 1 # Depth reduction for Late Move Reduction

        self.EVAL_PIECE_VALUES_LST = [0] * (max(EVAL_PIECE_VALUES_RAW.keys()) + 1)
        for p_type, val in EVAL_PIECE_VALUES_RAW.items():
            self.EVAL_PIECE_VALUES_LST[p_type] = val

        self.SEE_PIECE_VALUES = [0] * 7
        self.SEE_PIECE_VALUES[chess.PAWN] = EVAL_PIECE_VALUES_RAW[chess.PAWN]
        self.SEE_PIECE_VALUES[chess.KNIGHT] = EVAL_PIECE_VALUES_RAW[chess.KNIGHT]
        self.SEE_PIECE_VALUES[chess.BISHOP] = EVAL_PIECE_VALUES_RAW[chess.BISHOP]
        self.SEE_PIECE_VALUES[chess.ROOK] = EVAL_PIECE_VALUES_RAW[chess.ROOK]
        self.SEE_PIECE_VALUES[chess.QUEEN] = EVAL_PIECE_VALUES_RAW[chess.QUEEN]
        self.SEE_PIECE_VALUES[chess.KING] = 20000

        self.TT_MOVE_ORDER_BONUS = 250000
        self.CAPTURE_BASE_ORDER_BONUS = 100000
        self.QUEEN_PROMO_ORDER_BONUS = 120000
        self.MINOR_PROMO_ORDER_BONUS = 20000
        self.KILLER_1_ORDER_BONUS = 110000
        self.KILLER_2_ORDER_BONUS = 70000
        self.HISTORY_MAX_BONUS = 90000

        self.SEE_POSITIVE_BONUS_VALUE = 15000
        self.SEE_NEGATIVE_PENALTY_VALUE = -3500
        self.SEE_VALUE_SCALING_FACTOR = 8
        self.SEE_Q_PRUNING_THRESHOLD = -30
        
        self._SEE_PIECE_TYPES_SORTED = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

        self.init_zobrist_tables()

    def init_zobrist_tables(self):
        random.seed(42) # For reproducible hashes
        self.zobrist_piece_square = [[random.getrandbits(64) for _ in range(12)] for _ in range(64)]
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        self.zobrist_ep = [random.getrandbits(64) for _ in range(8)]
        self.zobrist_side = random.getrandbits(64)
        self.zobrist_white_checks = [random.getrandbits(64) for _ in range(11)] 
        self.zobrist_black_checks = [random.getrandbits(64) for _ in range(11)]
        self._zobrist_wc = self.zobrist_white_checks
        self._zobrist_bc = self.zobrist_black_checks

    def compute_hash(self, board: chess.Board, white_checks: int, black_checks: int) -> int:
        h = 0
        for sq, piece in board.piece_map().items():
            piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            h ^= self.zobrist_piece_square[sq][piece_idx]
        h ^= self.zobrist_castling[board.castling_rights & 0xF]
        if board.ep_square is not None:
            h ^= self.zobrist_ep[chess.square_file(board.ep_square)]
        if board.turn == chess.BLACK:
            h ^= self.zobrist_side
        if white_checks < 11: h ^= self._zobrist_wc[white_checks]
        if black_checks < 11: h ^= self._zobrist_bc[black_checks]
        return h

    def get_mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        attacker_type = board.piece_type_at(move.from_square)
        victim_type = board.piece_type_at(move.to_square)
        if victim_type is None:
            if not board.is_en_passant(move): return 0
            victim_type = chess.PAWN
        if attacker_type is None: return 0
        return self.EVAL_PIECE_VALUES_LST[victim_type] * 10 - self.EVAL_PIECE_VALUES_LST[attacker_type]

    def _get_lowest_attacker_see(self, board: chess.Board, to_sq: chess.Square, side: chess.Color) -> Optional[chess.Move]:
        attackers_mask = board.attackers_mask(side, to_sq)
        if not attackers_mask: return None
        for piece_type in self._SEE_PIECE_TYPES_SORTED:
            pieces = board.pieces_mask(piece_type, side) & attackers_mask
            if pieces:
                from_sq = chess.msb(pieces)
                return chess.Move(from_sq, to_sq)
        return None

    def see(self, board: chess.Board, move: chess.Move) -> int:
        initial_attacker_piece = board.piece_at(move.from_square)
        if not initial_attacker_piece: return 0
        if board.is_en_passant(move): initial_victim_value = self.SEE_PIECE_VALUES[chess.PAWN]
        else:
            initial_victim_piece = board.piece_at(move.to_square)
            if not initial_victim_piece: return 0
            initial_victim_value = self.SEE_PIECE_VALUES[initial_victim_piece.piece_type]
        see_value_stack = [0] * 32
        see_value_stack[0] = initial_victim_value
        num_see_moves = 1
        piece_on_target_val = self.SEE_PIECE_VALUES[move.promotion or initial_attacker_piece.piece_type]
        temp_board = board.copy()
        try: temp_board.push(move)
        except AssertionError: return 0
        while True:
            recapture_move = self._get_lowest_attacker_see(temp_board, move.to_square, temp_board.turn)
            if not recapture_move: break
            recapturing_piece = temp_board.piece_at(recapture_move.from_square)
            if not recapturing_piece: break
            see_value_stack[num_see_moves] = piece_on_target_val
            piece_on_target_val = self.SEE_PIECE_VALUES[recapture_move.promotion or recapturing_piece.piece_type]
            try: temp_board.push(recapture_move)
            except AssertionError: break
            num_see_moves += 1
        score = 0
        for i in range(num_see_moves - 1, -1, -1): score = max(0, see_value_stack[i] - score)
        return score

    def get_move_score(self, board: chess.Board, move: chess.Move, is_capture: bool, gives_check: bool,
                       tt_move: Optional[chess.Move], ply: int, white_checks_delivered: int, black_checks_delivered: int,
                       max_checks: int, qsearch_mode: bool = False) -> int:
        score = 0
        if not qsearch_mode and tt_move and move == tt_move: score += self.TT_MOVE_ORDER_BONUS
        if is_capture:
            score += self.CAPTURE_BASE_ORDER_BONUS + self.get_mvv_lva_score(board, move)
            if not qsearch_mode:
                see_val = self.see(board, move)
                num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + (1 if gives_check else 0)
                is_critical_check_related_capture = gives_check and (num_checks_by_mover_after_move >= max_checks -1)
                if see_val > 0: score += self.SEE_POSITIVE_BONUS_VALUE + see_val * self.SEE_VALUE_SCALING_FACTOR
                elif see_val < 0 and not is_critical_check_related_capture: score += self.SEE_NEGATIVE_PENALTY_VALUE + see_val * self.SEE_VALUE_SCALING_FACTOR
        if gives_check:
            num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + 1
            checks_remaining_to_win = max(0, max_checks - num_checks_by_mover_after_move)
            check_bonus_factor = self.CHECK_BONUS_DECAY_BASE_ORDERING ** checks_remaining_to_win
            dynamic_check_bonus = self.CHECK_BONUS_ADDITIONAL_MAX_ORDERING * check_bonus_factor
            score += int(self.CHECK_BONUS_MIN_ORDERING + dynamic_check_bonus)
        if not qsearch_mode:
            if move.promotion == chess.QUEEN: score += self.QUEEN_PROMO_ORDER_BONUS
            elif move.promotion: score += self.MINOR_PROMO_ORDER_BONUS
            if not is_capture:
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
            gives_check = board.gives_check(m)
            if qsearch_mode and not is_c and not is_promo and not gives_check: continue
            score = self.get_move_score(board, m, is_c, gives_check, tt_move, ply, white_checks, black_checks, max_checks, qsearch_mode)
            moves_to_process_details.append((score, m, is_c, gives_check))
        moves_to_process_details.sort(key=lambda x: x[0], reverse=True)
        return [(data[1], data[2], data[3]) for data in moves_to_process_details]

    def evaluate_position(self, board: chess.Board, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> int:
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner is not None: return self.CHECKMATE_SCORE if outcome.winner == chess.WHITE else -self.CHECKMATE_SCORE
            return 0
        material_positional_score = 0
        for sq, piece in board.piece_map().items():
            value = self.EVAL_PIECE_VALUES_LST[piece.piece_type]
            pst_val = PST[piece.piece_type][sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
            if piece.color == chess.WHITE: material_positional_score += value + pst_val
            else: material_positional_score -= (value + pst_val)
        check_bonus_score = 0
        if EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS > 0:
            if white_checks_delivered > 0 and white_checks_delivered < max_checks:
                exponent = max_checks - white_checks_delivered - 1
                bonus = EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS * (EVAL_CHECK_BONUS_DECAY_RATE ** exponent)
                check_bonus_score += int(bonus)
            if black_checks_delivered > 0 and black_checks_delivered < max_checks:
                exponent = max_checks - black_checks_delivered - 1
                bonus = EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS * (EVAL_CHECK_BONUS_DECAY_RATE ** exponent)
                check_bonus_score -= int(bonus)
        current_player_threat_bonus = 0
        if any(board.gives_check(m) for m in board.legal_moves): current_player_threat_bonus = EVAL_CHECK_THREAT_BONUS
        if board.turn == chess.WHITE: material_positional_score += current_player_threat_bonus
        else: material_positional_score -= current_player_threat_bonus
        return material_positional_score + check_bonus_score

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int, maximizing_player: bool,
                          white_checks: int, black_checks: int, max_checks: int, q_depth: int) -> int:
        self.q_nodes_evaluated += 1
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE
            return 0
        if q_depth <= 0: return self.evaluate_position(board, white_checks, black_checks, max_checks)
        stand_pat_score = self.evaluate_position(board, white_checks, black_checks, max_checks)
        if maximizing_player:
            if stand_pat_score >= beta: return beta
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha: return alpha
            beta = min(beta, stand_pat_score)
        ordered_forcing_moves_data = self.order_moves(board, board.legal_moves, None, 0, white_checks, black_checks, max_checks, qsearch_mode=True)
        for move, is_capture_flag, _ in ordered_forcing_moves_data:
            if is_capture_flag and self.see(board, move) < self.SEE_Q_PRUNING_THRESHOLD: continue
            board.push(move)
            score = self.quiescence_search(board, alpha, beta, not maximizing_player, white_checks, black_checks, max_checks, q_depth - 1)
            board.pop()
            if maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta: break
            else:
                beta = min(beta, score)
                if beta <= alpha: break
        return alpha if maximizing_player else beta

    def store_in_tt(self, key: int, depth: int, value: int, flag: int, best_move: Optional[chess.Move]):
        existing_entry = self.transposition_table.get(key)
        if existing_entry and existing_entry['depth'] > depth: return
        if len(self.transposition_table) >= self.tt_size_limit and self.tt_size_limit > 0 and not existing_entry:
            try: self.transposition_table.pop(next(iter(self.transposition_table)))
            except StopIteration: pass
        if self.tt_size_limit > 0:
            self.transposition_table[key] = {'depth': depth, 'value': value, 'flag': flag, 'best_move': best_move.uci() if best_move else None}

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing_player: bool, ply: int,
                white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> Tuple[int, Optional[chess.Move]]:
        if white_checks_delivered >= max_checks: return self.WIN_SCORE - ply, None
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE + ply, None

        alpha_orig = alpha
        in_check_at_node_start = board.is_check()
        if in_check_at_node_start: depth += 1

        tt_key = self.compute_hash(board, white_checks_delivered, black_checks_delivered)
        tt_entry = self.transposition_table.get(tt_key)
        tt_move = None
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['best_move']:
                try: tt_move = board.parse_uci(tt_entry['best_move'])
                except (ValueError, AssertionError): tt_move = None
                if tt_move and not board.is_legal(tt_move): tt_move = None
            if tt_entry['flag'] == TT_EXACT: return tt_entry['value'], tt_move
            elif tt_entry['flag'] == TT_LOWERBOUND: alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == TT_UPPERBOUND: beta = min(beta, tt_entry['value'])
            if alpha >= beta: return tt_entry['value'], tt_move
        
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE - ply, None
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE + ply, None
            return 0, None

        if depth <= 0:
            self.nodes_evaluated += 1
            return self.quiescence_search(board, alpha, beta, maximizing_player, white_checks_delivered, black_checks_delivered, max_checks, self.MAX_Q_DEPTH), None
        
        best_move_for_node: Optional[chess.Move] = None
        ordered_moves_data = self.order_moves(board, board.legal_moves, tt_move, ply, white_checks_delivered, black_checks_delivered, max_checks)

        if not ordered_moves_data:
            if in_check_at_node_start: return (-self.CHECKMATE_SCORE + ply), None
            else: return 0, None

        best_val_for_node = -float('inf') if maximizing_player else float('inf')
        for i, (move, is_capture_flag, gives_check_flag) in enumerate(ordered_moves_data):
            current_w_checks, current_b_checks = white_checks_delivered, black_checks_delivered
            
            # --- Late Move Reduction (LMR) ---
            reduction = 0
            # Conditions for LMR: not a tactical move, not in check, not at low depth, and not one of the first few moves.
            if depth > 2 and i > 0 and not in_check_at_node_start and not is_capture_flag and not move.promotion and not gives_check_flag:
                reduction = self.LMR_REDUCTION
            
            new_depth = depth - 1 - reduction
            
            board.push(move)
            if board.is_check():
                if board.turn == chess.BLACK: current_w_checks += 1 # turn has switched, so previous turn was WHITE
                else: current_b_checks += 1
            
            # Search with potentially reduced depth
            eval_val, _ = self.minimax(board, new_depth, alpha, beta, not maximizing_player, ply + 1, current_w_checks, current_b_checks, max_checks)
            
            # If the reduced search produced a high score, it might be a good move. Re-search with full depth.
            if reduction > 0 and eval_val > alpha:
                eval_val, _ = self.minimax(board, depth - 1, alpha, beta, not maximizing_player, ply + 1, current_w_checks, current_b_checks, max_checks)

            board.pop()
            
            if maximizing_player:
                if eval_val > best_val_for_node: best_val_for_node, best_move_for_node = eval_val, move
                alpha = max(alpha, eval_val)
            else:
                if eval_val < best_val_for_node: best_val_for_node, best_move_for_node = eval_val, move
                beta = min(beta, eval_val)
            
            if alpha >= beta:
                if not is_capture_flag:
                    piece = board.piece_at(move.from_square)
                    if piece:
                        if move != self.killer_moves[ply][0]: self.killer_moves[ply][1], self.killer_moves[ply][0] = self.killer_moves[ply][0], move
                        self.history_table[(piece.piece_type, move.to_square)] = self.history_table.get((piece.piece_type, move.to_square), 0) + depth**2
                break
        
        flag = TT_EXACT if alpha_orig < best_val_for_node < beta else TT_LOWERBOUND if best_val_for_node >= beta else TT_UPPERBOUND
        self.store_in_tt(tt_key, depth, best_val_for_node, flag, best_move_for_node)
        return best_val_for_node, best_move_for_node

    def find_best_move(self, game_state: GameState, depth: int) -> Optional[chess.Move]:
        self.nodes_evaluated = 0; self.q_nodes_evaluated = 0
        best_move_overall: Optional[chess.Move] = None
        final_value = 0
        self.killer_moves = [[None, None] for _ in range(depth + 15)]
        self.history_table.clear()
        
        # We search on a copy of the board to not interfere with the main game state during search
        board_copy = game_state.board.copy()

        for d_iterative in range(1, depth + 1):
            value, move = self.minimax(board_copy, d_iterative, -float('inf'), float('inf'),
                                       board_copy.turn == chess.WHITE, 0,
                                       game_state.white_checks_delivered,
                                       game_state.black_checks_delivered,
                                       game_state.MAX_CHECKS)
            if move is not None:
                best_move_overall, final_value = move, value
            elif best_move_overall is None and d_iterative == 1:
                final_value = value
            self.current_eval_for_ui = final_value
        
        if best_move_overall is None and list(board_copy.legal_moves) and not game_state.get_outcome():
            print("AI Warning: Fallback move.")
            best_move_overall = next(iter(board_copy.legal_moves), None)
            self.current_eval_for_ui = self.evaluate_position(board_copy, game_state.white_checks_delivered, game_state.black_checks_delivered, game_state.MAX_CHECKS)
        return best_move_overall

class ChessUI:
    def __init__(self):
        self.square_size = INITIAL_SQUARE_SIZE
        self.root = tk.Tk()
        self.root.title(f"{NUM_CHECKS_TO_WIN}-Check Chess")
        self.root.geometry(f"{INITIAL_WINDOW_WIDTH}x{INITIAL_WINDOW_HEIGHT}")
        self.base_font = font.Font(family="Arial", size=10)
        self.bold_font = font.Font(family="Arial", size=11, weight="bold")
        self.piece_font_family = "Segoe UI Symbol"
        self.game_state = GameState(max_checks=NUM_CHECKS_TO_WIN)
        self.ai_depth = default_ai_depth
        self.flipped = False
        self.game_over_flag = False
        self.ai_thinking = False
        self.player_is_white = True
        self.dragging_piece_item_id: Optional[int] = None
        self.drag_start_x_offset = 0; self.drag_start_y_offset = 0
        self.drag_selected_square: Optional[chess.Square] = None
        self.piece_item_ids: Dict[chess.Square, int] = {}
        self.square_item_ids: Dict[chess.Square, int] = {}
        self.original_square_colors: Dict[int, str] = {}
        self.highlighted_rect_id: Optional[int] = None
        self.piece_images: Dict[str, str] = {}
        self.ai = ChessAI()
        self.ai_move_queue = queue.Queue()
        self.colors = {'square_light': '#E0C4A0', 'square_dark': '#A08060', 'highlight_selected_fill': '#6FA8DC', 'highlight_capture': '#FF7F7F80', 'highlight_normal_move': '#90EE9080', 'piece_white': '#FFFFFF', 'piece_black': '#202020', 'piece_outline': '#444444'}
        self.stipple_patterns = {'light': 'gray25', 'medium': 'gray50', 'heavy': 'gray75'}
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
        main_frame = ttk.Frame(self.root, padding="5"); main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(main_frame, borderwidth=1, relief="sunken", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(5,0), pady=5)
        main_frame.grid_columnconfigure(0, weight=1); main_frame.grid_rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", self.on_canvas_resize); self.canvas.bind("<Button-1>", self.on_square_interaction_start)
        self.canvas.bind("<B1-Motion>", self.on_piece_drag_motion); self.canvas.bind("<ButtonRelease-1>", self.on_square_interaction_end)
        control_panel = ttk.Frame(main_frame, padding=(10,5,8,5), width=CONTROL_PANEL_DEFAULT_WIDTH)
        control_panel.grid(row=0, column=1, sticky="ns", padx=(8,5)); control_panel.grid_propagate(False)
        row_idx, btn_width = 0, 22
        
        ttk.Button(control_panel, text="New Game", command=self.reset_game, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        ttk.Button(control_panel, text="Switch Sides & New Game", command=self.switch_sides, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        ttk.Button(control_panel, text="Undo", command=self.undo_last_player_ai_moves, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1

        self.checks_labelframe = ttk.LabelFrame(control_panel, text=f"Checks (Goal: {self.game_state.MAX_CHECKS})")
        self.checks_labelframe.grid(row=row_idx, column=0,columnspan=2, pady=(8,5), sticky="ew"); row_idx+=1
        inner_checks_frame = ttk.Frame(self.checks_labelframe); inner_checks_frame.pack(fill=tk.X, expand=True, padx=4, pady=(2,4))
        inner_checks_frame.columnconfigure(0, weight=1); inner_checks_frame.columnconfigure(1, weight=1)
        self.white_checks_label = ttk.Label(inner_checks_frame, text="W:", font=(self.piece_font_family, 11), anchor="w"); self.white_checks_label.grid(row=0, column=0, sticky="ew", padx=(3,2))
        self.black_checks_label = ttk.Label(inner_checks_frame, text="B:", font=(self.piece_font_family, 11), anchor="w"); self.black_checks_label.grid(row=0, column=1, sticky="ew", padx=(2,3))
        depth_frame = ttk.Frame(control_panel); depth_frame.grid(row=row_idx, column=0, columnspan=2, pady=(5,0), sticky="ew"); row_idx+=1
        ttk.Label(depth_frame, text="AI Depth:", font=self.base_font).pack(side=tk.LEFT, padx=(0,5))
        self.depth_value_label = ttk.Label(depth_frame, text=str(self.ai_depth), font=self.base_font, width=2, anchor="e"); self.depth_value_label.pack(side=tk.RIGHT, padx=(5,0))
        self.depth_slider = ttk.Scale(control_panel, from_=1, to=6, orient=tk.HORIZONTAL, command=self.update_ai_depth); self.depth_slider.set(self.ai_depth); self.depth_slider.grid(row=row_idx, column=0, columnspan=2, pady=(0,5), sticky="ew"); row_idx+=1
        self.status_label = ttk.Label(control_panel, text="White's turn", font=self.bold_font, wraplength=CONTROL_PANEL_DEFAULT_WIDTH-20, anchor="center", justify="center", width=26)
        self.status_label.grid(row=row_idx, column=0,columnspan=2, pady=(4,2), sticky="ew", ipady=3); row_idx+=1
        self.eval_label = ttk.Label(control_panel, text="Eval: +0.00", font=self.base_font, wraplength=CONTROL_PANEL_DEFAULT_WIDTH-20, anchor="center", justify="center", width=26)
        self.eval_label.grid(row=row_idx, column=0, columnspan=2, pady=(0,8), sticky="ew", ipady=2); row_idx+=1
        ttk.Button(control_panel, text="Exit", command=self.root.quit, width=btn_width).grid(row=row_idx, column=0,columnspan=2, pady=4, sticky="ew"); row_idx+=1
        control_panel.grid_rowconfigure(row_idx, weight=1)

    def load_piece_images(self): self.piece_images = {'R':'♖','N':'♘','B':'♗','Q':'♕','K':'♔','P':'♙','r':'♜','n':'♞','b':'♝','q':'♛','k':'♚','p':'♟'}
    def on_canvas_resize(self, event=None):
        w, h = (event.width, event.height) if event else (self.canvas.winfo_width(), self.canvas.winfo_height())
        if w < 16 or h < 16: return
        new_sq_size = max(10, min(w // 8, h // 8))
        if new_sq_size != self.square_size or event is None:
            self.square_size = new_sq_size; self.redraw_board_and_pieces()

    def _get_square_coords(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        f_display = 7-f if self.flipped else f
        r_display = r if self.flipped else 7-r
        return f_display*self.square_size, r_display*self.square_size, (f_display+1)*self.square_size, (r_display+1)*self.square_size

    def _get_canvas_xy_to_square(self, x, y):
        if self.square_size == 0: return None
        f_idx, r_idx = x // self.square_size, y // self.square_size
        if not (0 <= f_idx < 8 and 0 <= r_idx < 8): return None
        f, r = (7 - f_idx if self.flipped else f_idx), (r_idx if self.flipped else 7 - r_idx)
        return chess.square(f, r) if 0 <= f < 8 and 0 <= r < 8 else None

    def _draw_squares(self):
        self.canvas.delete("square_bg"); self.square_item_ids.clear()
        for r_chess in range(8):
            for f_chess in range(8):
                sq = chess.square(f_chess,r_chess); x1,y1,x2,y2 = self._get_square_coords(sq)
                color = self.colors['square_light'] if (f_chess+r_chess)%2!=0 else self.colors['square_dark']
                self.square_item_ids[sq] = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", tags=("square_bg", f"sq_bg_{sq}"))

    def _draw_all_pieces(self):
        self.canvas.delete("piece_outline", "piece_fg"); self.piece_item_ids.clear()
        font_size = int(self.square_size * 0.73)
        if font_size < 6: return
        outline_offset = max(1, int(font_size * 0.03))
        for sq in CHESS_SQUARES:
            piece = self.game_state.board.piece_at(sq)
            if piece:
                x1,y1,x2,y2 = self._get_square_coords(sq); cx,cy = (x1+x2)//2, (y1+y2)//2
                symbol = self.piece_images[piece.symbol()]
                pc = self.colors['piece_white'] if piece.color == chess.WHITE else self.colors['piece_black']
                oc = self.colors['piece_black'] if piece.color == chess.WHITE else self.colors['piece_white']
                self.canvas.create_text(cx+outline_offset, cy+outline_offset, text=symbol, font=(self.piece_font_family, font_size), fill=oc, tags=("piece_outline", f"piece_outline_{sq}"), anchor="c")
                self.piece_item_ids[sq] = self.canvas.create_text(cx,cy,text=symbol,font=(self.piece_font_family,font_size),fill=pc,tags=("piece_fg",f"piece_fg_{sq}"),anchor="c")

    def redraw_board_and_pieces(self, for_ai_move_eval=None): self._draw_squares(); self._draw_all_pieces(); self._update_check_display(); self.update_status_label(for_ai_move_eval)
    def _clear_highlights(self):
        if self.highlighted_rect_id and self.highlighted_rect_id in self.original_square_colors:
            try: self.canvas.itemconfig(self.highlighted_rect_id, fill=self.original_square_colors.pop(self.highlighted_rect_id))
            except (tk.TclError, KeyError): pass
        self.highlighted_rect_id = None; self.canvas.delete("highlight_dot")

    def _highlight_square_selected(self, sq):
        item_id = self.square_item_ids.get(sq)
        if item_id:
            try:
                if item_id not in self.original_square_colors: self.original_square_colors[item_id] = self.canvas.itemcget(item_id, "fill")
                self.canvas.itemconfig(item_id, fill=self.colors['highlight_selected_fill'])
                self.highlighted_rect_id = item_id; self.canvas.tag_raise("piece_fg", "piece_outline")
            except tk.TclError: self.highlighted_rect_id = None

    def _highlight_legal_move(self, sq, is_capture):
        x1, y1, x2, y2 = self._get_square_coords(sq); cx, cy = (x1 + x2) // 2, (y1 + y2) // 2; radius = self.square_size * 0.15
        fill = (self.colors['highlight_capture'] if is_capture else self.colors['highlight_normal_move'])[:7]
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=fill, stipple=self.stipple_patterns['medium'], outline="", tags="highlight_dot")

    def _show_legal_moves_for_selected_piece(self):
        if self.drag_selected_square is not None:
            self._highlight_square_selected(self.drag_selected_square)
            for m in self.game_state.board.legal_moves:
                if m.from_square == self.drag_selected_square: self._highlight_legal_move(m.to_square, self.game_state.board.is_capture(m))

    def _update_check_display(self):
        filled, empty = '✚', '⊕'
        max_symbols = self.game_state.MAX_CHECKS if self.game_state.MAX_CHECKS <= 7 else 7
        w_filled = self.game_state.white_checks_delivered; w_text = "W: " + (filled * w_filled) + (empty * max(0, min(max_symbols - w_filled, self.game_state.MAX_CHECKS - w_filled))); self.white_checks_label.config(text=w_text.strip())
        b_filled = self.game_state.black_checks_delivered; b_text = "B: " + (filled * b_filled) + (empty * max(0, min(max_symbols - b_filled, self.game_state.MAX_CHECKS - b_filled))); self.black_checks_label.config(text=b_text.strip())

    def update_status_label(self, for_ai_move_eval=None):
        status, eval_text = "", ""
        outcome = self.game_state.get_outcome()

        if outcome:
            self.game_over_flag = True
            if outcome.winner is None:
                status = f"Draw! ({outcome.termination.name.replace('_',' ').title()})"
                eval_text = "Eval: Draw"
            else:
                winner = "White" if outcome.winner == chess.WHITE else "Black"
                if outcome.termination == chess.Termination.VARIANT_WIN:
                    status = f"{winner} wins by {self.game_state.MAX_CHECKS} checks!"
                    eval_text = f"Eval: {self.game_state.MAX_CHECKS}-Check Win!"
                else: # Checkmate
                    status = f"Checkmate! {winner} wins!"
                    eval_text = "Eval: Checkmate!"
        else:
            self.game_over_flag = False
            status = ("White's" if self.game_state.board.turn == chess.WHITE else "Black's") + " turn" + (" (Check!)" if self.game_state.board.is_check() else "")

        if not self.game_over_flag:
            if self.ai_thinking:
                eval_text = "Eval: Thinking..."
            elif for_ai_move_eval is not None:
                score = for_ai_move_eval
                win_thresh, mate_thresh = self.ai.WIN_SCORE - 200, self.ai.CHECKMATE_SCORE - 200
                if abs(score) > win_thresh:
                    plies_to_win = self.ai.WIN_SCORE - abs(score)
                    eval_text = f"Eval: {self.game_state.MAX_CHECKS}-Check Win in {plies_to_win} plies"
                elif abs(score) > mate_thresh:
                    plies_to_mate = self.ai.CHECKMATE_SCORE - abs(score)
                    eval_text = f"Eval: Mate in {plies_to_mate} plies"
                else:
                    eval_text = f"Eval: {score / 100.0:+.2f}"
            else:
                 eval_text = "Eval: +0.00"

        if hasattr(self, "status_label"): self.status_label.config(text=status)
        if hasattr(self, "eval_label"): self.eval_label.config(text=eval_text)

    def reset_game(self):
        self.game_state.reset(); self.game_state.MAX_CHECKS = NUM_CHECKS_TO_WIN
        self.game_over_flag = False
        if hasattr(self, 'root'): self.root.title(f"{self.game_state.MAX_CHECKS}-Check Chess")
        if hasattr(self, 'checks_labelframe'): self.checks_labelframe.config(text=f"Checks (Goal: {self.game_state.MAX_CHECKS})")
        self.ai.transposition_table.clear();
        self.ai.killer_moves = [[None,None] for _ in range(self.ai_depth + 15)]; self.ai.history_table.clear()
        self.drag_selected_square = None; self.ai_thinking = False
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id = None
        self._clear_highlights();
        if hasattr(self, 'depth_slider'): self.depth_slider.set(self.ai_depth)
        self.update_ai_depth(self.ai_depth)
        self.on_canvas_resize()
        
        if not self.is_player_turn():
            self.root.after(250, self._start_ai_move_thread)

    def switch_sides(self):
        self.player_is_white = not self.player_is_white
        self.flipped = not self.player_is_white
        self.reset_game()

    def undo_last_player_ai_moves(self):
        if self.ai_thinking: return
        num_to_undo = 2 if len(self.game_state.board.move_stack) >= 2 else 1 if len(self.game_state.board.move_stack) >= 1 else 0
        for _ in range(num_to_undo):
            self.game_state.undo_move()
        if num_to_undo > 0: self.game_over_flag = False; self._clear_highlights(); self.drag_selected_square = None; self.redraw_board_and_pieces()

    def update_ai_depth(self, val_str):
        val_int = int(round(float(val_str)))
        if self.ai_depth != val_int:
            self.ai_depth=val_int; self.ai.killer_moves = [[None,None] for _ in range(val_int + 15)]; self.ai.history_table.clear()
        if hasattr(self, 'depth_slider') and int(round(self.depth_slider.get())) != val_int:
            self.depth_slider.set(val_int)
        if hasattr(self, 'depth_value_label'): self.depth_value_label.config(text=str(val_int))

    def is_player_turn(self):
        return (self.game_state.board.turn == chess.WHITE and self.player_is_white) or \
               (self.game_state.board.turn == chess.BLACK and not self.player_is_white)

    def on_square_interaction_start(self,event):
        if self.game_over_flag or not self.is_player_turn() or self.ai_thinking: return
        clicked_sq = self._get_canvas_xy_to_square(event.x,event.y)
        if clicked_sq is None : return
        if self.drag_selected_square is not None:
            if self.drag_selected_square != clicked_sq:
                 self._attempt_player_move(self.drag_selected_square,clicked_sq)
                 return
            self._clear_highlights(); self.drag_selected_square = None; self.redraw_board_and_pieces()
        else:
            piece = self.game_state.board.piece_at(clicked_sq)
            if piece and piece.color == self.game_state.board.turn:
                self.drag_selected_square = clicked_sq; self.dragging_piece_item_id = self.piece_item_ids.get(clicked_sq)
                if self.dragging_piece_item_id:
                    self.canvas.lift(self.dragging_piece_item_id); self.canvas.lift(f"piece_outline_{clicked_sq}")
                    coords = self.canvas.coords(self.dragging_piece_item_id)
                    if coords: self.drag_start_x_offset, self.drag_start_y_offset = event.x-coords[0], event.y-coords[1]
                self._show_legal_moves_for_selected_piece()

    def on_piece_drag_motion(self,event):
        if self.dragging_piece_item_id and self.drag_selected_square and not self.ai_thinking:
            nx = event.x-self.drag_start_x_offset
            ny = event.y-self.drag_start_y_offset
            self.canvas.coords(self.dragging_piece_item_id, nx, ny)
            outline_tags = self.canvas.find_withtag(f"piece_outline_{self.drag_selected_square}")
            if outline_tags:
                self.canvas.coords(outline_tags[0], nx + max(1, int(self.square_size * 0.03)), ny + max(1, int(self.square_size * 0.03)))

    def on_square_interaction_end(self,event):
        if self.drag_selected_square is None or self.ai_thinking: return
        to_sq = self._get_canvas_xy_to_square(event.x, event.y)
        from_sq = self.drag_selected_square
        self.drag_selected_square = None; self.dragging_piece_item_id = None
        if to_sq is not None and to_sq != from_sq:
            self._attempt_player_move(from_sq, to_sq)
        else:
            self.redraw_board_and_pieces()

    def _attempt_player_move(self, from_sq, to_sq):
        p = self.game_state.board.piece_at(from_sq)
        promo = chess.QUEEN if p and p.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7] else None
        move = chess.Move(from_sq, to_sq, promotion=promo)
        self._clear_highlights()
        if self.game_state.make_move(move):
            self.redraw_board_and_pieces()
            if not self.game_over_flag:
                self.root.after(50, self._start_ai_move_thread)
        else:
            self.redraw_board_and_pieces()

    def _start_ai_move_thread(self):
        if self.ai_thinking or self.game_over_flag or self.is_player_turn(): return
        self.ai_thinking = True
        self.update_status_label()
        ai_thread = threading.Thread(target=self._ai_move_worker, daemon=True)
        ai_thread.start()

    def _ai_move_worker(self):
        start_time = time.monotonic()
        ai_move = self.ai.find_best_move(self.game_state, self.ai_depth)
        eval_from_ai_perspective = self.ai.current_eval_for_ui
        elapsed = time.monotonic() - start_time
        total_nodes = self.ai.nodes_evaluated + self.ai.q_nodes_evaluated
        nps = total_nodes / elapsed if elapsed > 0 else 0
        tt_fill_percent = (len(self.ai.transposition_table) / self.ai.tt_size_limit) * 100 if self.ai.tt_size_limit > 0 else 0
        ai_player_color_char = 'White' if self.game_state.board.turn == chess.WHITE else 'Black'
        move_uci_str = ai_move.uci() if ai_move else "None (no move)"
        # Adjust eval score to always be from White's perspective for console logging
        eval_for_console_white_pov = eval_from_ai_perspective * (1 if ai_player_color_char == 'White' else -1)
        print_info = (f"AI playing ({ai_player_color_char}) | Move: {move_uci_str} | Eval (White's POV): {eval_for_console_white_pov/100.0:+.2f} | "
                      f"Time: {elapsed:.2f}s | Depth: {self.ai_depth} | Nodes: {total_nodes} (NPS: {nps:.0f}) | TT Fill: {tt_fill_percent:.1f}%")
        self.ai_move_queue.put((ai_move, eval_from_ai_perspective, print_info))

    def _check_ai_queue_periodically(self):
        try:
            move, eval_score, print_info = self.ai_move_queue.get_nowait()
            print(print_info)
            self.ai_thinking = False
            if self.game_over_flag:
                self.update_status_label()
                return
            
            if not self.game_state.make_move(move):
                 print(f"AI Warning: {'Illegal' if move else 'No'} move proposed: {move.uci() if move else 'None'}")
            
            self.redraw_board_and_pieces(for_ai_move_eval=eval_score)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_ai_queue_periodically)

if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()