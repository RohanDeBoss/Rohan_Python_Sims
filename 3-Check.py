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
TT_SIZE_POWER_OF_2 = 16 # 2^16 = 32768 entries
# ----------------------------

# --- AI Evaluation Constants ---
EVAL_WIN_SCORE = 1000000
EVAL_LOSS_SCORE = -1000000
EVAL_CHECKMATE_SCORE = 900000

EVAL_PIECE_VALUES_RAW = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
}

EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS = 900
EVAL_CHECK_THREAT_BONUS = 300
# ----------------------------


ORIGINAL_PSTS_RAW = {
    chess.PAWN: [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [0,0,0,0,0,0,0,0,5,5,10,10,10,10,5,5,-5,0,0,0,0,0,0,-5,0,0,0,0,0,0,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [ 20, 30, 10,  0,  0, 10, 30, 20, 20, 20,  0,  0,  0,  0, 20, 20,-10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30]
}
PST = {piece_type: list(reversed(values)) for piece_type, values in ORIGINAL_PSTS_RAW.items()}

TT_EXACT = 0; TT_LOWERBOUND = 1; TT_UPPERBOUND = 2
CHESS_SQUARES = list(chess.SQUARES)

class ChessAI:
    # These are for MOVE ORDERING - they affect which moves are searched first
    CHECK_BONUS_MIN_ORDERING = 5000
    CHECK_BONUS_ADDITIONAL_MAX_ORDERING = 15000
    CHECK_BONUS_DECAY_BASE_ORDERING = 5/6

    def __init__(self):
        self.transposition_table: Dict[int, Dict[str, Any]] = {}
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[chess.PieceType, chess.Square], int] = {}
        self.tt_size_limit = 2**TT_SIZE_POWER_OF_2
        self.nodes_evaluated = 0
        self.q_nodes_evaluated = 0
        self.current_eval_for_ui = 0
        
        self.WIN_SCORE = EVAL_WIN_SCORE
        self.LOSS_SCORE = EVAL_LOSS_SCORE
        self.CHECKMATE_SCORE = EVAL_CHECKMATE_SCORE
        
        self.MAX_Q_DEPTH = 6
        
        # Piece values for static evaluation
        self.EVAL_PIECE_VALUES_LST = [0] * (max(EVAL_PIECE_VALUES_RAW.keys()) + 1)
        for p_type, val in EVAL_PIECE_VALUES_RAW.items():
            self.EVAL_PIECE_VALUES_LST[p_type] = val
        
        # Piece values for SEE (Static Exchange Evaluation) - King high to avoid trading into mate
        self.SEE_PIECE_VALUES = [0, 100, 320, 330, 500, 900, 20000] 
        
        # Piece values for NMP material check (can be same as EVAL or specific)
        self._PIECE_VALUES_LST_NMP = [0, 100, 320, 330, 500, 900, 0] # King 0 for NMP material
        
        self.NMP_R_REDUCTION = 3
        self.NMP_MIN_DEPTH_THRESHOLD = 1 + self.NMP_R_REDUCTION
        self.NMP_MIN_MATERIAL_FOR_SIDE = self._PIECE_VALUES_LST_NMP[chess.ROOK]

        # --- Move Ordering Bonuses (instance members, could be globals too) ---
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

        # --- Late Move Reduction (LMR) Parameters ---
        self.LMR_MIN_DEPTH = 3
        self.LMR_MIN_MOVES_TRIED = 3
        self.LMR_REDUCTION = 0 #Deactivated for now

        self.init_zobrist_tables()

    def init_zobrist_tables(self):
        random.seed(42)
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
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]: # Exclude King
            material += len(board.pieces(piece_type, color)) * self._PIECE_VALUES_LST_NMP[piece_type]
        return material

    def get_mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        attacker = board.piece_at(move.from_square)
        victim_piece_type_at_to_square = board.piece_at(move.to_square)
        
        if board.is_en_passant(move): victim_piece_type = chess.PAWN
        elif victim_piece_type_at_to_square: victim_piece_type = victim_piece_type_at_to_square.piece_type
        else: return 0 # Not a capture
        if not attacker: return 0  # Should not happen for a legal move
        
        # Using EVAL_PIECE_VALUES_LST for MVV-LVA as it reflects general piece worth
        victim_value = self.EVAL_PIECE_VALUES_LST[victim_piece_type]
        attacker_value = self.EVAL_PIECE_VALUES_LST[attacker.piece_type]
        return victim_value * 10 - attacker_value # Simple MVV-LVA

    def _get_lowest_attacker_see(self, board: chess.Board, to_sq: chess.Square, side: chess.Color) -> Optional[chess.Move]:
        lowest_value = float('inf')
        best_move = None
        
        attacker_squares = board.attackers(side, to_sq)
        if not attacker_squares:
            return None

        for from_sq in attacker_squares:
            piece = board.piece_at(from_sq)
            if piece: 
                current_value = self.SEE_PIECE_VALUES[piece.piece_type] # Use specific SEE values
                if current_value < lowest_value:
                    lowest_value = current_value
                    if piece.piece_type == chess.PAWN and \
                       ((side == chess.WHITE and chess.square_rank(to_sq) == 7) or \
                        (side == chess.BLACK and chess.square_rank(to_sq) == 0)):
                        best_move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                    else:
                        best_move = chess.Move(from_sq, to_sq)
        return best_move

    def see(self, board: chess.Board, move: chess.Move) -> int:
        target_sq = move.to_square
        initial_attacker_piece = board.piece_at(move.from_square)
        if not initial_attacker_piece: return 0

        if board.is_en_passant(move):
            initial_victim_value = self.SEE_PIECE_VALUES[chess.PAWN]
        else:
            initial_victim_piece = board.piece_at(target_sq)
            if not initial_victim_piece: return 0 
            initial_victim_value = self.SEE_PIECE_VALUES[initial_victim_piece.piece_type]

        see_value_stack = [0] * 32 
        see_value_stack[0] = initial_victim_value
        num_see_moves = 1

        if move.promotion:
            piece_on_target_val = self.SEE_PIECE_VALUES[move.promotion]
        else:
            piece_on_target_val = self.SEE_PIECE_VALUES[initial_attacker_piece.piece_type]
            
        temp_board = board.copy()
        try:
            temp_board.push(move)
        except AssertionError: 
            return 0 

        while num_see_moves < 32:
            recapture_side = temp_board.turn
            recapture_move = self._get_lowest_attacker_see(temp_board, target_sq, recapture_side)
            if not recapture_move: break
            
            recapturing_piece = temp_board.piece_at(recapture_move.from_square)
            if not recapturing_piece: break
            
            if recapture_move.promotion:
                recapturing_piece_val = self.SEE_PIECE_VALUES[recapture_move.promotion]
            else:
                recapturing_piece_val = self.SEE_PIECE_VALUES[recapturing_piece.piece_type]

            see_value_stack[num_see_moves] = piece_on_target_val
            piece_on_target_val = recapturing_piece_val 
            
            try:
                temp_board.push(recapture_move)
            except AssertionError: break 
            num_see_moves += 1

        score = 0
        for i in range(num_see_moves - 1, -1, -1):
            current_player_net_gain = see_value_stack[i] - score
            if i > 0 and current_player_net_gain < 0: 
                score = 0 
            else:
                score = current_player_net_gain
        return score

    def get_move_score(self, board: chess.Board, move: chess.Move, 
                       is_capture: bool, gives_check: bool, 
                       tt_move: Optional[chess.Move], ply: int, 
                       white_checks_delivered: int, black_checks_delivered: int, 
                       max_checks: int, qsearch_mode: bool = False) -> int:
        score = 0
        if not qsearch_mode and tt_move and move == tt_move:
            score += self.TT_MOVE_ORDER_BONUS 
        
        if is_capture: 
            score += self.CAPTURE_BASE_ORDER_BONUS + self.get_mvv_lva_score(board, move)
            if not qsearch_mode:
                see_val = self.see(board, move) 
                if see_val > 0:
                    score += self.SEE_POSITIVE_BONUS_VALUE + see_val * self.SEE_VALUE_SCALING_FACTOR
                elif see_val < 0:
                    num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + (1 if gives_check else 0)
                    is_critical_check = gives_check and (num_checks_by_mover_after_move >= max_checks -1)
                    if not is_critical_check :
                         score += self.SEE_NEGATIVE_PENALTY_VALUE + see_val * self.SEE_VALUE_SCALING_FACTOR
        
        if gives_check:
            # This is the MOVE ORDERING bonus for checks
            num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + 1
            checks_remaining_to_win = max(0, max_checks - num_checks_by_mover_after_move)
            
            check_bonus_factor = self.CHECK_BONUS_DECAY_BASE_ORDERING ** checks_remaining_to_win
            dynamic_check_bonus = self.CHECK_BONUS_ADDITIONAL_MAX_ORDERING * check_bonus_factor
            total_check_bonus_for_ordering = self.CHECK_BONUS_MIN_ORDERING + dynamic_check_bonus
            score += int(total_check_bonus_for_ordering)

        if not qsearch_mode:
            if move.promotion == chess.QUEEN:
                score += self.QUEEN_PROMO_ORDER_BONUS
            elif move.promotion: 
                score += self.MINOR_PROMO_ORDER_BONUS
            
            if not is_capture: 
                if self.killer_moves[ply][0] == move:
                    score += self.KILLER_1_ORDER_BONUS
                elif self.killer_moves[ply][1] == move:
                    score += self.KILLER_2_ORDER_BONUS
                
                piece = board.piece_at(move.from_square) 
                if piece: 
                    history_score = self.history_table.get((piece.piece_type, move.to_square), 0)
                    score += min(history_score, self.HISTORY_MAX_BONUS) 
        return score

    def order_moves(self, board: chess.Board, legal_moves_generator: chess.LegalMoveGenerator, 
                    tt_move: Optional[chess.Move], ply: int, 
                    white_checks: int, black_checks: int, max_checks: int, 
                    qsearch_mode: bool = False) -> List[Tuple[chess.Move, bool, bool]]:
        moves_to_process = []
        for m in legal_moves_generator:
            is_c = board.is_capture(m)
            g_c = False # Will be set to true if it gives check (only calculated if needed)
            
            if qsearch_mode:
                passes_q_filter = False
                if is_c: # All captures are considered in qsearch
                    passes_q_filter = True
                else: 
                    # For non-captures in qsearch, only consider "critical" checks
                    # (e.g., those that are 1 away from winning or actually win)
                    # This requires checking if the move gives check.
                    # We need to be careful not to call board.gives_check too often if not needed.
                    # A common approach is to only check 'gives_check' if it's a promotion or for specific heuristics.
                    # Here, we check it if the move could be critical.
                    current_checks_for_mover = white_checks if board.turn == chess.WHITE else black_checks
                    if current_checks_for_mover >= max_checks - 2: # If already 2+ checks away, any check is critical
                        # Temporarily make the move to see if it's a check
                        # This is expensive. A cheaper way is to assume non-captures don't get q-searched
                        # unless they are promotions or part of a specific check evasion sequence.
                        # For now, let's use your existing logic for q-search filtering.
                        g_c = board.gives_check(m) # Calculate only if needed for q-filter
                        if g_c:
                            checks_by_mover_if_this_move = current_checks_for_mover + 1
                            if checks_by_mover_if_this_move >= max_checks -1 : # N-1 or N checks
                                passes_q_filter = True
                if not passes_q_filter:
                    continue 
            else: # Not qsearch_mode, calculate gives_check for all moves for ordering
                 g_c = board.gives_check(m)

            moves_to_process.append({'move': m, 'is_capture': is_c, 'gives_check': g_c})

        scored_move_data = []
        for move_attrs in moves_to_process:
            m = move_attrs['move']
            is_c = move_attrs['is_capture']
            g_c = move_attrs['gives_check']
            
            current_score = self.get_move_score(board, m, is_c, g_c, 
                                               tt_move, ply, 
                                               white_checks, black_checks, max_checks, 
                                               qsearch_mode)
            scored_move_data.append((current_score, m, is_c, g_c))
        
        scored_move_data.sort(key=lambda x: x[0], reverse=True)
        return [(data[1], data[2], data[3]) for data in scored_move_data]

    def evaluate_position(self, board: chess.Board, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> int:
        # 1. Handle immediate N-check win/loss
        if white_checks_delivered >= max_checks: return self.WIN_SCORE
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE
        
        # 2. Handle traditional game outcomes (checkmate/stalemate)
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE 
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE
            return 0 # Draw

        # 3. Imminent Threat of Winning Check by Opponent (Heuristic)
        side_to_move = board.turn
        opponent_color = not side_to_move
        opponent_checks_count = black_checks_delivered if side_to_move == chess.WHITE else white_checks_delivered
        
        if opponent_checks_count == max_checks - 1: 
            temp_board = board.copy()
            temp_board.push(chess.Move.null()) 
            
            can_opponent_deliver_winning_check = False
            if temp_board.is_valid(): 
                for opponent_move in temp_board.legal_moves:
                    temp_board.push(opponent_move)
                    if temp_board.is_check(): 
                        can_opponent_deliver_winning_check = True
                        temp_board.pop() 
                        break
                    temp_board.pop() 
            
            if can_opponent_deliver_winning_check:
                return self.LOSS_SCORE # From perspective of current player to move

        # --- Standard Material & Positional Score ---
        material_positional_score = 0
        for sq in CHESS_SQUARES:
            piece = board.piece_at(sq)
            if piece:
                value = self.EVAL_PIECE_VALUES_LST[piece.piece_type]
                pst_val = PST[piece.piece_type][sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
                if piece.color == chess.WHITE:
                    material_positional_score += value + pst_val
                else:
                    material_positional_score -= (value + pst_val)

        # --- Non-linear "Material" Bonus for Accumulated Checks ---
        check_bonus_score = 0
        effective_max_checks_for_bonus_calc = max(1, max_checks - 1) 

        if EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS > 0 and effective_max_checks_for_bonus_calc > 0:
            if white_checks_delivered > 0:
                ratio_w = min(1.0, white_checks_delivered / effective_max_checks_for_bonus_calc) # Cap ratio at 1.0
                accumulated_bonus_w = EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS * (ratio_w ** 2) 
                check_bonus_score += int(accumulated_bonus_w)

            if black_checks_delivered > 0:
                ratio_b = min(1.0, black_checks_delivered / effective_max_checks_for_bonus_calc) # Cap ratio at 1.0
                accumulated_bonus_b = EVAL_MAX_BONUS_FOR_N_MINUS_1_CHECKS * (ratio_b ** 2)
                check_bonus_score -= int(accumulated_bonus_b)
        
        # --- Check Threat Bonus (for current player to move) ---
        current_player_threat_bonus = 0
        if not board.is_game_over(claim_draw=True) and \
           (white_checks_delivered < max_checks and black_checks_delivered < max_checks):
            for move in board.legal_moves: 
                if board.gives_check(move):
                    current_player_threat_bonus = EVAL_CHECK_THREAT_BONUS
                    break
        
        if board.turn == chess.WHITE:
            material_positional_score += current_player_threat_bonus
        else:
            material_positional_score -= current_player_threat_bonus
            
        return material_positional_score + check_bonus_score

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int, 
                          maximizing_player: bool, white_checks: int, black_checks: int, 
                          max_checks: int, q_depth: int) -> int:
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
            if stand_pat_score >= beta: return beta 
            alpha = max(alpha, stand_pat_score)
        else: 
            if stand_pat_score <= alpha: return alpha 
            beta = min(beta, stand_pat_score)
        
        ordered_forcing_moves_data = self.order_moves(board, board.legal_moves, None, 0,
                                                 white_checks, black_checks, max_checks, 
                                                 qsearch_mode=True)
        
        for move, is_capture_flag, gives_check_flag in ordered_forcing_moves_data:
            # Filter for q-search already happened in order_moves if qsearch_mode=True
            # But we still apply SEE pruning for captures here
            if is_capture_flag: # No need to check not gives_check_flag, as q-search might include critical checks
                see_val = self.see(board, move)
                if see_val < self.SEE_Q_PRUNING_THRESHOLD:
                    continue 

            current_w_checks, current_b_checks = white_checks, black_checks
            piece_color_that_moved = board.turn 
            board.push(move)
            # Re-check if it's a check in the new board state, as gives_check_flag might be from previous state
            if board.is_check(): 
                if piece_color_that_moved == chess.WHITE: current_w_checks += 1
                else: current_b_checks += 1
            
            score = self.quiescence_search(board, alpha, beta, not maximizing_player, 
                                       current_w_checks, current_b_checks, max_checks, 
                                       q_depth - 1)
            board.pop()

            if maximizing_player:
                alpha = max(alpha, score)
                if alpha >= beta: return beta 
            else: 
                beta = min(beta, score)
                if alpha >= beta: return alpha 
            
        return alpha if maximizing_player else beta

    def store_in_tt(self, key: int, depth: int, value: int, flag: int, best_move: Optional[chess.Move]):
        if len(self.transposition_table) >= self.tt_size_limit and self.tt_size_limit > 0 :
            try: self.transposition_table.pop(next(iter(self.transposition_table))) 
            except StopIteration: pass
        
        if self.tt_size_limit > 0 :
             self.transposition_table[key] = {'depth': depth, 'value': value, 'flag': flag, 'best_move': best_move.uci() if best_move else None}

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int, 
                maximizing_player: bool, ply: int, 
                white_checks_delivered: int, black_checks_delivered: int, max_checks: int
                ) -> Tuple[int, Optional[chess.Move]]:
        alpha_orig = alpha 
        
        if white_checks_delivered >= max_checks: return self.WIN_SCORE, None
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE, None

        tt_key = self.compute_hash(board)
        tt_entry = self.transposition_table.get(tt_key)
        tt_move: Optional[chess.Move] = None
        if tt_entry:
            if tt_entry['best_move']:
                 try: tt_move = board.parse_uci(tt_entry['best_move'])
                 except (ValueError, AssertionError): pass # Catch if move is illegal on current board
            if tt_entry['depth'] >= depth:
                # Ensure tt_move is legal before returning it
                if tt_move and not board.is_legal(tt_move):
                    tt_move = None # Invalidate if not legal
                
                if tt_entry['flag'] == TT_EXACT: return tt_entry['value'], tt_move
                elif tt_entry['flag'] == TT_LOWERBOUND: alpha = max(alpha, tt_entry['value'])
                elif tt_entry['flag'] == TT_UPPERBOUND: beta = min(beta, tt_entry['value'])
                if alpha >= beta: return tt_entry['value'], tt_move
        
        if depth <= 0:
            self.nodes_evaluated += 1
            return self.quiescence_search(board, alpha, beta, maximizing_player, 
                                          white_checks_delivered, black_checks_delivered, max_checks, 
                                          self.MAX_Q_DEPTH), None
        
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE, None
            elif outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE, None
            else: return 0, None

        in_check_at_node_start = board.is_check()

        if (not in_check_at_node_start and 
            depth >= self.NMP_MIN_DEPTH_THRESHOLD and 
            ply > 0 and 
            self.get_side_material(board, board.turn) >= self.NMP_MIN_MATERIAL_FOR_SIDE):
            
            # Make sure null move is pseudo-legal (doesn't leave king in check if it wasn't)
            # board.push_uci("0000")
            board.push(chess.Move.null())
            # if board.is_valid(): # Not strictly necessary as python-chess handles illegal null moves
            null_move_score, _ = self.minimax(board, depth - 1 - self.NMP_R_REDUCTION, 
                                                -beta, -beta + 1, not maximizing_player, ply + 1, 
                                                white_checks_delivered, black_checks_delivered, max_checks)
            board.pop()
            null_move_score = -null_move_score 
            if null_move_score >= beta: return beta, None 
            # else: board.pop() # If null move was invalid, it wouldn't have pushed.

        best_move_for_node: Optional[chess.Move] = None
        ordered_moves_data = self.order_moves(board, board.legal_moves, tt_move, ply, 
                                         white_checks_delivered, black_checks_delivered, max_checks)
        
        if not ordered_moves_data: 
            if in_check_at_node_start: # Checkmate
                return (-self.CHECKMATE_SCORE + ply if maximizing_player else self.CHECKMATE_SCORE - ply), None
            else: return 0, None # Stalemate

        best_val_for_node = -float('inf') if maximizing_player else float('inf')
        moves_searched_count = 0

        for move, is_capture_flag, gives_check_flag_from_order in ordered_moves_data: # gives_check_flag_from_order is from order_moves
            moves_searched_count += 1
            current_w_checks, current_b_checks = white_checks_delivered, black_checks_delivered
            piece_color_that_moved = board.turn

            piece_that_moved_type_before_move: Optional[chess.PieceType] = None
            if not is_capture_flag and move.promotion is None: # For history heuristic
                piece_obj = board.piece_at(move.from_square)
                if piece_obj: 
                    piece_that_moved_type_before_move = piece_obj.piece_type

            board.push(move)

            # is_check_after_this_move should be determined from the new board state
            is_check_after_this_move = board.is_check() 
            if is_check_after_this_move: # This is the actual check status after the move
                if piece_color_that_moved == chess.WHITE: current_w_checks += 1
                else: current_b_checks += 1
            
            eval_val: int
            effective_search_depth = depth - 1

            do_lmr = (depth >= self.LMR_MIN_DEPTH and
                      moves_searched_count > self.LMR_MIN_MOVES_TRIED and
                      not in_check_at_node_start and  # Don't reduce if in check
                      not is_capture_flag and         # Don't reduce captures
                      not is_check_after_this_move)   # Don't reduce moves that give check

            if do_lmr:
                reduced_depth = max(0, effective_search_depth - self.LMR_REDUCTION)
                eval_val, _ = self.minimax(board, reduced_depth, alpha, beta, 
                                           not maximizing_player, ply + 1, 
                                           current_w_checks, current_b_checks, max_checks)
                
                # Re-search with full depth if LMR result was promising
                if (maximizing_player and eval_val > alpha) or \
                   (not maximizing_player and eval_val < beta):
                    eval_val, _ = self.minimax(board, effective_search_depth, alpha, beta, 
                                               not maximizing_player, ply + 1, 
                                               current_w_checks, current_b_checks, max_checks)
            else:
                 eval_val, _ = self.minimax(board, effective_search_depth, alpha, beta, 
                                           not maximizing_player, ply + 1, 
                                           current_w_checks, current_b_checks, max_checks)
            
            board.pop()

            if maximizing_player:
                if eval_val > best_val_for_node: 
                    best_val_for_node = eval_val
                    best_move_for_node = move
                alpha = max(alpha, eval_val)
                if beta <= alpha: # Beta cutoff
                    if piece_that_moved_type_before_move: # Update history/killer for non-captures causing cutoff
                        if move != self.killer_moves[ply][0]: # Avoid duplicates
                            self.killer_moves[ply][1] = self.killer_moves[ply][0]
                            self.killer_moves[ply][0] = move
                        self.history_table[(piece_that_moved_type_before_move, move.to_square)] = \
                            self.history_table.get((piece_that_moved_type_before_move, move.to_square), 0) + depth**2
                    break
            else: # Minimizing player
                if eval_val < best_val_for_node: 
                    best_val_for_node = eval_val
                    best_move_for_node = move
                beta = min(beta, eval_val)
                if beta <= alpha: # Alpha cutoff
                    if piece_that_moved_type_before_move: # Update history/killer for non-captures causing cutoff
                        if move != self.killer_moves[ply][0]: # Avoid duplicates
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
        self.nodes_evaluated = 0; self.q_nodes_evaluated = 0; self.current_eval_for_ui = 0
        is_maximizing_player = (board.turn == chess.WHITE)
        best_move_overall: Optional[chess.Move] = None
        final_value_for_best_move = 0
        
        # Consider clearing TT per search or managing entries more carefully for iterative deepening
        # For simplicity, clearing per find_best_move call (as originally)
        self.transposition_table.clear()
        self.killer_moves = [[None, None] for _ in range(64)] 
        self.history_table.clear()

        for d_iterative in range(1, depth + 1):
            value, move_at_this_depth = self.minimax(board, d_iterative, -float('inf'), float('inf'), is_maximizing_player, 0, white_checks, black_checks, max_checks)

            if move_at_this_depth is not None:
                best_move_overall = move_at_this_depth
                final_value_for_best_move = value
            elif best_move_overall is None and d_iterative == 1: # If no move found at depth 1, use the eval
                final_value_for_best_move = value
                 
            # Update UI eval based on the latest iteration's result
            if not board.is_game_over(claim_draw=True) and white_checks < max_checks and black_checks < max_checks:
                 self.current_eval_for_ui = final_value_for_best_move
            else: # If game is over, use the terminal evaluation
                 self.current_eval_for_ui = self.evaluate_position(board, white_checks, black_checks, max_checks)

            # Early exit if mate is found
            should_stop = False
            if is_maximizing_player:
                 # (CHECKMATE_SCORE - d_iterative) accounts for mate in X plies
                 if value >= self.WIN_SCORE or value >= (self.CHECKMATE_SCORE - d_iterative * 10): # Multiplier for ply depth
                     should_stop = True
            else: 
                 if value <= self.LOSS_SCORE or value <= (-self.CHECKMATE_SCORE + d_iterative * 10): 
                     should_stop = True
                     
            if should_stop:
                 break

        # Fallback if no move found but legal moves exist (should be rare with proper game end checks)
        if best_move_overall is None and list(board.legal_moves) and \
           white_checks < max_checks and black_checks < max_checks and \
           not board.is_game_over(claim_draw=True):
            print("AI Warning: find_best_move returned None but legal moves exist. Picking first legal move.")
            best_move_overall = next(iter(board.legal_moves), None)
            # Re-evaluate current_eval_for_ui if we picked a fallback
            if best_move_overall is not None:
                 self.current_eval_for_ui = self.evaluate_position(board, white_checks, black_checks, max_checks)
            else: # Should not happen if list(board.legal_moves) is true
                 self.current_eval_for_ui = self.evaluate_position(board, white_checks, black_checks, max_checks)
        
        # Final eval update if game ended during search or by the found move
        if board.is_game_over(claim_draw=True) or white_checks >= max_checks or black_checks >= max_checks:
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
        self.piece_font_family = "Segoe UI Symbol" 

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
        self.original_square_colors: Dict[int, str] = {} 
        self.highlighted_rect_id: Optional[int] = None 

        self.piece_images: Dict[str, str] = {} 
        self.ai = ChessAI()
        self.ai_move_queue = queue.Queue() 

        self.colors = {
            'square_light': '#E0C4A0', 'square_dark': '#A08060', 
            'highlight_selected_fill': '#6FA8DC', 
            'highlight_capture': '#FF7F7F80', 
            'highlight_normal_move': '#90EE9080', 
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
        if w < 16 or h < 16: return
        new_sq_size = max(10, min(w // 8, h // 8))
        if new_sq_size != self.square_size: self.square_size = new_sq_size; self.redraw_board_and_pieces()
        elif event is None: self.redraw_board_and_pieces()

    def _get_square_coords(self, sq: chess.Square) -> Tuple[int,int,int,int]:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped: f, r = 7-f, r 
        else: r = 7-r 
        return f*self.square_size, r*self.square_size, (f+1)*self.square_size, (r+1)*self.square_size

    def _get_canvas_xy_to_square(self, x:int, y:int) -> Optional[chess.Square]:
        if self.square_size == 0: return None
        f_idx,r_idx = x//self.square_size, y//self.square_size
        if not(0<=f_idx<8 and 0<=r_idx<8): return None
        if self.flipped: f, r = 7-f_idx, r_idx
        else: f, r = f_idx, 7-r_idx
        return chess.square(f,r) if (0 <= f < 8 and 0 <= r < 8) else None

    def _draw_squares(self):
        self.canvas.delete("square_bg"); self.square_item_ids.clear()
        for r_chess in range(8):
            for f_chess in range(8):
                sq = chess.square(f_chess,r_chess); x1,y1,x2,y2 = self._get_square_coords(sq)
                color = self.colors['square_light'] if (f_chess+r_chess)%2==0 else self.colors['square_dark']
                item_id = self.canvas.create_rectangle(x1,y1,x2,y2,fill=color,outline="",width=0,tags=("square_bg",f"sq_bg_{sq}"))
                self.square_item_ids[sq] = item_id
    
    def _draw_all_pieces(self):
        self.canvas.delete("piece_outline"); self.canvas.delete("piece_fg"); self.piece_item_ids.clear()
        font_size = int(self.square_size * 0.73); 
        if font_size < 6: return 
        outline_offset = max(1, int(font_size * 0.03)) 
        for sq in CHESS_SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                x1,y1,x2,y2 = self._get_square_coords(sq); cx,cy = (x1+x2)//2, (y1+y2)//2
                piece_symbol_str = self.piece_images[piece.symbol()]
                pc = self.colors['piece_white'] if piece.color == chess.WHITE else self.colors['piece_black']
                oc = self.colors['piece_black'] if piece.color == chess.WHITE else self.colors['piece_white']
                self.canvas.create_text(cx+outline_offset, cy+outline_offset, text=piece_symbol_str, font=(self.piece_font_family, font_size), fill=oc, tags=("piece_outline", f"piece_outline_{sq}"), anchor="c")
                item_id = self.canvas.create_text(cx,cy,text=piece_symbol_str,font=(self.piece_font_family,font_size),fill=pc,tags=("piece_fg",f"piece_fg_{sq}"),anchor="c")
                self.piece_item_ids[sq] = item_id

    def redraw_board_and_pieces(self):
        self._draw_squares(); self._draw_all_pieces()
        self._update_check_display(); self.update_status_label()

    def _clear_highlights(self):
        if self.highlighted_rect_id and self.highlighted_rect_id in self.original_square_colors:
            try: self.canvas.itemconfig(self.highlighted_rect_id, fill=self.original_square_colors[self.highlighted_rect_id])
            except tk.TclError: pass 
            del self.original_square_colors[self.highlighted_rect_id]
        self.highlighted_rect_id = None
        self.canvas.delete("highlight_dot")

    def _highlight_square_selected(self, sq: chess.Square):
        item_id = self.square_item_ids.get(sq)
        if item_id:
            try:
                original_color = self.canvas.itemcget(item_id,"fill")
                if item_id not in self.original_square_colors: 
                    self.original_square_colors[item_id] = original_color
                self.canvas.itemconfig(item_id, fill=self.colors['highlight_selected_fill'])
                self.highlighted_rect_id = item_id
                self.canvas.tag_raise("piece_outline") 
                self.canvas.tag_raise("piece_fg")
            except tk.TclError:
                self.highlighted_rect_id = None

    def _highlight_legal_move(self, sq: chess.Square, is_capture: bool):
        x1,y1,x2,y2 = self._get_square_coords(sq); cx,cy=(x1+x2)//2,(y1+y2)//2
        radius = self.square_size*0.15
        fill_color_rgba = self.colors['highlight_capture'] if is_capture else self.colors['highlight_normal_move']
        fill_color_rgb = fill_color_rgba[:7] 
        stipple_pattern = self.stipple_patterns['medium']
        self.canvas.create_oval(cx-radius,cy-radius,cx+radius,cy+radius, fill=fill_color_rgb, stipple=stipple_pattern, outline="", tags="highlight_dot")

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
        self.white_checks_label.config(text=white_text.strip()); self.black_checks_label.config(text=black_text.strip())

    def update_status_label(self, for_ai_move_eval: Optional[int] = None):
        status_text = ""; eval_text = ""
        if self.game_over_flag:
            current_status = self.status_label.cget("text").split("\n")[0]
            if "wins" in current_status or "Draw" in current_status or "Checkmate" in current_status: 
                 status_text = current_status
        
        if not status_text: 
            if self.white_checks_delivered >= self.MAX_CHECKS: status_text = f"White wins by {self.MAX_CHECKS} checks!"; self.game_over_flag = True
            elif self.black_checks_delivered >= self.MAX_CHECKS: status_text = f"Black wins by {self.MAX_CHECKS} checks!"; self.game_over_flag = True
            else:
                outcome = self.board.outcome(claim_draw=True)
                if outcome:
                    self.game_over_flag = True
                    if outcome.winner == chess.WHITE: status_text = "Checkmate! White wins!"
                    elif outcome.winner == chess.BLACK: status_text = "Checkmate! Black wins!"
                    else: status_text = f"Draw! ({outcome.termination.name.replace('_',' ').title()})"
                else: 
                    status_text = ("White's turn" if self.board.turn == chess.WHITE else "Black's turn")
                    if self.board.is_check(): status_text += " (Check!)"
                    self.game_over_flag = False

        display_eval_score_internal = 0
        if self.ai_thinking and for_ai_move_eval is None: 
            eval_text = "Eval: Thinking..."
        elif not self.game_over_flag or for_ai_move_eval is not None:
            if for_ai_move_eval is not None: 
                display_eval_score_internal = for_ai_move_eval
            elif not self.ai_thinking : 
                # If it's not AI's turn or AI is not thinking, get a fresh evaluation
                display_eval_score_internal = self.ai.evaluate_position(self.board, self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS)
            
            display_score_ui_units = display_eval_score_internal / 100.0

            if abs(display_eval_score_internal) >= self.ai.WIN_SCORE: 
                eval_text = "Eval: White winning" if display_eval_score_internal > 0 else "Eval: Black winning"
            elif abs(display_eval_score_internal) >= (self.ai.CHECKMATE_SCORE - (self.ai_depth * 20)): # Adjusted mate score threshold
                 eval_text = "Eval: Mate imminent" 
            else: 
                eval_text = f"Eval: {display_score_ui_units:+.2f}"
        else: # Game is over and no specific eval passed
            eval_text = "Eval: ---"
             # If game over, show final eval based on outcome
            if self.white_checks_delivered >= self.MAX_CHECKS: display_eval_score_internal = self.ai.WIN_SCORE
            elif self.black_checks_delivered >= self.MAX_CHECKS: display_eval_score_internal = self.ai.LOSS_SCORE
            elif self.board.is_checkmate():
                 display_eval_score_internal = self.ai.CHECKMATE_SCORE if self.board.turn == chess.BLACK else -self.ai.CHECKMATE_SCORE
            elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
                 display_eval_score_internal = 0
            eval_text = f"Eval: {display_eval_score_internal / 100.0:+.2f}" if abs(display_eval_score_internal) < self.ai.WIN_SCORE else eval_text


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
        
        self.ai.transposition_table.clear()
        self.ai.killer_moves = [[None,None] for _ in range(64)] 
        self.ai.history_table.clear()

        self.drag_selected_square=None; self.ai_thinking=False
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
        self._clear_highlights()
        if hasattr(self,'depth_slider'): self.depth_slider.set(self.ai_depth)
        self.update_ai_depth(self.ai_depth) # This calls update_status_label
        self.on_canvas_resize() # This calls redraw_board_and_pieces which calls update_status_label

    def flip_board(self):
        self.flipped=not self.flipped; self._clear_highlights(); self.drag_selected_square = None 
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
        self.redraw_board_and_pieces()

    def undo_last_player_ai_moves(self):
        if self.ai_thinking: return
        
        moves_to_undo = 0
        # Determine how many full player-AI turn pairs can be undone based on move_stack and check_history
        if len(self.board.move_stack) >= 2 and len(self.check_history) >=2 :
            moves_to_undo = 2
        elif len(self.board.move_stack) == 1 and len(self.check_history) >= 1:
             moves_to_undo = 1
        
        undone_count = 0
        for _ in range(moves_to_undo):
            if self.board.move_stack: # Should always be true if moves_to_undo > 0
                self.board.pop()
                if self.check_history: # Should also be true
                    self.white_checks_delivered, self.black_checks_delivered = self.check_history.pop()
                    undone_count += 1
                else: 
                    # Should not happen if logic for moves_to_undo is correct
                    self.white_checks_delivered=0; self.black_checks_delivered=0; break 
            else: 
                break # Should not happen
        
        if undone_count > 0:
            self.game_over_flag=False # Game might no longer be over
            self._clear_highlights()
            self.drag_selected_square=None
            if self.dragging_piece_item_id: 
                 self.canvas.delete(self.dragging_piece_item_id)
                 self.dragging_piece_item_id=None
            
            # Clear AI state as history is changed
            self.ai.transposition_table.clear()
            self.ai.killer_moves = [[None,None] for _ in range(64)] 
            self.ai.history_table.clear()

            self.redraw_board_and_pieces() # This calls update_status_label
            # Explicitly call update_status_label without for_ai_move_eval to get fresh eval
            self.update_status_label(for_ai_move_eval=None)


    def update_ai_depth(self, val):
        val_int = int(round(float(val)))
        if self.ai_depth != val_int:
            self.ai_depth=val_int
            self.ai.transposition_table.clear() # AI state changes with depth
            self.ai.killer_moves = [[None,None] for _ in range(64)] 
            self.ai.history_table.clear()

        if hasattr(self,'depth_slider') and abs(self.depth_slider.get()-val_int)>0.01: self.depth_slider.set(val_int)
        if hasattr(self,'depth_value_label'): self.depth_value_label.config(text=str(val_int))
        self.update_status_label() # Update eval display if depth changes

    def on_square_interaction_start(self,event):
        if self.game_over_flag or self.board.turn!=chess.WHITE or self.ai_thinking: return
        
        clicked_sq = self._get_canvas_xy_to_square(event.x,event.y)
        if clicked_sq is None : return 

        if self.drag_selected_square is not None: 
            if self.drag_selected_square != clicked_sq: # Attempt to move to new square
                 self._attempt_player_move(self.drag_selected_square,clicked_sq)
                 # State (selected_sq, dragging_id) will be reset in _attempt_player_move or if it fails
                 return
            else: # Clicked on the same selected square - deselect it
                 self._clear_highlights()
                 self.drag_selected_square = None
                 self.dragging_piece_item_id = None # No longer dragging
                 # Redraw to ensure piece is back if it was visually moved slightly by mistake
                 self.redraw_board_and_pieces() 
                 return

        self._clear_highlights() # Clear previous highlights if any
        piece = self.board.piece_at(clicked_sq)

        if piece and piece.color==self.board.turn: # Player selected their own piece
            self.drag_selected_square = clicked_sq
            
            # For dragging visual
            self.dragging_piece_item_id = self.piece_item_ids.get(clicked_sq) # Get canvas ID of piece
            if self.dragging_piece_item_id:
                self.canvas.lift(self.dragging_piece_item_id) # Lift piece to top
                outline_tags = self.canvas.find_withtag(f"piece_outline_{clicked_sq}")
                for item_id_outline in outline_tags: self.canvas.lift(item_id_outline)

                coords = self.canvas.coords(self.dragging_piece_item_id)
                if coords: # Should always have coords
                    self.drag_start_x_offset = event.x-coords[0]
                    self.drag_start_y_offset = event.y-coords[1]
                    # Visually move piece slightly to indicate selection for drag (optional)
                    # nx,ny = event.x-self.drag_start_x_offset, event.y-self.drag_start_y_offset
                    # self.canvas.coords(self.dragging_piece_item_id,nx,ny)
                    # for item_id_outline in outline_tags: self.canvas.coords(item_id_outline,nx,ny)
            
            self._show_legal_moves_for_selected_piece() # Highlight selected sq and legal moves
        else: # Clicked on empty square or opponent's piece
            self.drag_selected_square=None
            self.dragging_piece_item_id=None
            # No action, just clear highlights (already done)

    def on_piece_drag_motion(self,event):
        if self.dragging_piece_item_id and self.drag_selected_square is not None and not self.ai_thinking:
            nx,ny = event.x-self.drag_start_x_offset, event.y-self.drag_start_y_offset
            self.canvas.coords(self.dragging_piece_item_id,nx,ny)
            # Also move outline if it exists
            if self.drag_selected_square is not None: # Should always be true if dragging_piece_item_id is set
                 outline_tags = self.canvas.find_withtag(f"piece_outline_{self.drag_selected_square}")
                 for item_id_outline in outline_tags:
                    self.canvas.coords(item_id_outline,nx + (self.canvas.coords(item_id_outline)[0] - self.canvas.coords(self.dragging_piece_item_id)[0]),
                                       ny + (self.canvas.coords(item_id_outline)[1] - self.canvas.coords(self.dragging_piece_item_id)[1]))


    def on_square_interaction_end(self,event):
        if self.drag_selected_square is None or self.ai_thinking: 
            # If not dragging anything, or AI is thinking, do nothing on release.
            # If piece was visually moved but not dropped on a square, redraw to snap back.
            if self.dragging_piece_item_id: 
                self.redraw_board_and_pieces() # Snap back
            self._clear_highlights() # Always clear highlights
            self.drag_selected_square = None
            self.dragging_piece_item_id = None
            return

        to_sq = self._get_canvas_xy_to_square(event.x,event.y)
        from_sq_before_attempt = self.drag_selected_square # Store before resetting

        # Always reset dragging state and redraw to snap piece to its original or new square.
        # _attempt_player_move will handle actual piece redrawing if move is successful.
        # If move is not made or fails, this redraw ensures the dragged piece snaps back.
        self.dragging_piece_item_id = None # Stop visual dragging
        self.drag_selected_square = None   # Deselect
        self._clear_highlights()           # Clear visual highlights
        self.redraw_board_and_pieces()     # Redraw board to current state (snaps piece back if not moved)
        
        if to_sq is not None and from_sq_before_attempt is not None and to_sq != from_sq_before_attempt:
            self._attempt_player_move(from_sq_before_attempt, to_sq)
        # If dropped on same square or outside board, drag_selected_square is already None
        # and board is redrawn, effectively deselecting.

    def _attempt_player_move(self,from_sq:chess.Square,to_sq:chess.Square):
        promo=None
        p=self.board.piece_at(from_sq)
        if p and p.piece_type==chess.PAWN:
            if (p.color==chess.WHITE and chess.square_rank(to_sq)==7) or \
               (p.color==chess.BLACK and chess.square_rank(to_sq)==0): 
                # For simplicity, auto-queen. Could add a dialog here.
                promo=chess.QUEEN 
        
        move=chess.Move(from_sq,to_sq,promotion=promo)
        
        # Reset selection state regardless of move legality
        self.drag_selected_square=None 
        self.dragging_piece_item_id=None
        self._clear_highlights()

        if move in self.board.legal_moves:
            self.check_history.append((self.white_checks_delivered,self.black_checks_delivered)) # Store pre-move checks
            
            self.board.push(move)
            
            # Check if this move delivered a check
            if self.board.is_check(): # Current board state is after white's move
                # If it's black's turn now and white (who just moved) delivered check
                if self.board.turn == chess.BLACK: 
                     self.white_checks_delivered+=1

            self.redraw_board_and_pieces() # Redraws with new piece positions and updates status
            
            # Check game over conditions after redrawing and status update
            if not self.game_over_flag: # game_over_flag is updated in update_status_label
                # self.update_status_label(for_ai_move_eval=None) # Called by redraw
                self.root.update_idletasks() # Ensure UI updates before AI starts
                self._start_ai_move_thread()
        else: 
            print(f"Illegal move: {move.uci()}")
            self.redraw_board_and_pieces() # Redraw to revert visual state if move was illegal
            # update_status_label is called by redraw_board_and_pieces

    def _start_ai_move_thread(self):
        if self.ai_thinking or self.game_over_flag: return
        
        self.ai_thinking = True
        self.status_label.config(text="AI is thinking...") # Immediate feedback
        self.eval_label.config(text="Eval: Thinking...") 
        if hasattr(self, 'depth_slider'): self.depth_slider.config(state=tk.DISABLED)
        
        # Pass current check counts and MAX_CHECKS to AI
        ai_thread = threading.Thread(target=self._ai_move_worker, 
            args=(self.board.copy(), self.ai_depth, 
                  self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS), 
            daemon=True)
        ai_thread.start()

    def _ai_move_worker(self, board_copy: chess.Board, depth: int, w_checks: int, b_checks: int, max_c: int):
        start_time = time.monotonic()
        
        ai_move = self.ai.find_best_move(board_copy, depth, w_checks, b_checks, max_c)
        
        eval_after_search_internal = self.ai.current_eval_for_ui # Get eval from AI instance
        elapsed = time.monotonic()-start_time
        total_nodes = self.ai.nodes_evaluated + self.ai.q_nodes_evaluated
        nps = total_nodes/elapsed if elapsed>0 else 0
        tt_fill_percent = (len(self.ai.transposition_table) / self.ai.tt_size_limit) * 100 if self.ai.tt_size_limit > 0 else 0
        
        ai_player_color_char = 'White' if board_copy.turn == chess.WHITE else 'Black'
        move_uci_str = ai_move.uci() if ai_move else "None (no move/game over)"
        
        print_info = (f"AI playing ({ai_player_color_char}) | Eval: {eval_after_search_internal/100.0:+.2f} | "
                      f"Time:{elapsed:.2f}s | Depth:{depth} | Nodes:{total_nodes} (NPS:{nps:.0f}) | TT Fill:{tt_fill_percent:.1f}%")
        
        self.ai_move_queue.put((ai_move, print_info, eval_after_search_internal))

    def _check_ai_queue_periodically(self):
        try:
            ai_move, print_info, eval_score_from_ai_internal = self.ai_move_queue.get_nowait()
            self._process_ai_move_from_queue(ai_move, print_info, eval_score_from_ai_internal)
        except queue.Empty: 
            pass 
        finally: 
            self.root.after(100, self._check_ai_queue_periodically) # Poll every 100ms

    def _process_ai_move_from_queue(self, ai_move: Optional[chess.Move], print_info: str, eval_score_from_ai_search_internal: int):
        self.ai_thinking = False # AI has finished
        if hasattr(self, 'depth_slider'): self.depth_slider.config(state=tk.NORMAL)
        
        print(print_info) # Log AI move details
        
        if self.game_over_flag: # Check if game ended while AI was thinking (e.g. by player action if possible)
            self.update_status_label(); # Ensure status is current
            return

        if ai_move:
            if ai_move in self.board.legal_moves: # Double check legality on current board
                self.check_history.append((self.white_checks_delivered,self.black_checks_delivered)) # Store pre-AI-move checks
                
                self.board.push(ai_move)
                
                # Check if AI's move delivered a check
                if self.board.is_check(): # Current board state is after AI's move
                    # If it's white's turn now, AI (Black) delivered check
                    if self.board.turn == chess.WHITE: 
                        self.black_checks_delivered+=1
                    # This case should not happen if AI is White and delivers check (turn would be Black)
                    # elif self.board.turn == chess.BLACK: 
                    #     self.white_checks_delivered +=1 # Should be covered by player's move logic

            else: 
                # This is a problem - AI suggested an illegal move for the current board state.
                # Could happen if board state changed unexpectedly or AI has a bug.
                print(f"AI Warning: Proposed move {ai_move.uci()} is illegal on current board state! Game may be in unexpected state.")
                if not list(self.board.legal_moves): # If no legal moves, game might be over
                     self.game_over_flag = True # Force update_status_label to check outcome

            self.redraw_board_and_pieces() # Updates UI and status_label
            
            # Pass the evaluation from the AI search directly to update_status_label
            # This avoids re-evaluating immediately after a search.
            self.update_status_label(for_ai_move_eval=eval_score_from_ai_search_internal) 
        else: 
            # AI returned no move. This usually means game is over (checkmate/stalemate from AI's perspective).
            print("AI returned no move. Game likely over or AI error.")
            self.game_over_flag = True # Force update_status_label to check outcome
            self.update_status_label() # Update based on current board state

        # If game isn't flagged as over by the move itself, ensure a fresh eval for next player turn.
        if not self.game_over_flag: 
             self.update_status_label(for_ai_move_eval=None) # Get fresh eval for player's turn


if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()