import tkinter as tk
from tkinter import ttk, font
import chess
import time
import random
from typing import Tuple, List, Optional, Dict, Any
import threading
import queue

# --- Global Configuration ---
NUM_CHECKS_TO_WIN = 3
INITIAL_SQUARE_SIZE = 66
INITIAL_WINDOW_WIDTH = (8 * INITIAL_SQUARE_SIZE) + 290
INITIAL_WINDOW_HEIGHT = (8 * INITIAL_SQUARE_SIZE) + 29
CONTROL_PANEL_DEFAULT_WIDTH = INITIAL_WINDOW_WIDTH - (8 * INITIAL_SQUARE_SIZE) - 37
TT_SIZE_POWER_OF_2 = 17 # 2^17 = 131072 entries
# ----------------------------

# Original PSTs at their original scale
ORIGINAL_PSTS_RAW = {
    chess.PAWN: [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [0,0,0,0,0,0,0,0,5,5,10,10,10,10,5,5,-5,0,0,0,0,0,0,-5,0,0,0,0,0,0,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [ 20, 30, 10,  0,  0, 10, 30, 20, 20, 20,  0,  0,  0,  0, 20, 20,-10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30]
}
# PSTs are reversed for a1=0 indexing
PST = {piece_type: list(reversed(values)) for piece_type, values in ORIGINAL_PSTS_RAW.items()}


TT_EXACT = 0; TT_LOWERBOUND = 1; TT_UPPERBOUND = 2
CHESS_SQUARES = list(chess.SQUARES)

class ChessAI:
    CHECK_BONUS_MIN = 10000 
    CHECK_BONUS_ADDITIONAL_MAX = 140000 
    CHECK_BONUS_DECAY_BASE = 6/7

    def __init__(self):
        self.transposition_table: Dict[int, Dict[str, Any]] = {}
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[chess.PieceType, chess.Square], int] = {}
        self.tt_size_limit = 2**TT_SIZE_POWER_OF_2
        self.nodes_evaluated = 0
        self.q_nodes_evaluated = 0
        self.current_eval_for_ui = 0
        self.init_zobrist_tables()

        self.WIN_SCORE = 1000000
        self.LOSS_SCORE = -1000000
        self.CHECKMATE_SCORE = 900000
        
        self.TOTAL_STATIC_CHECK_VALUE = 300 # This value is added to material/PST sum
        self.MAX_Q_DEPTH = 5
        
        # Piece values at their original intended scale for evaluation and NMP
        self._PIECE_VALUES_LST_FOR_NMP_MATERIAL = [0, 100, 320, 330, 500, 900, 20000] # King high for NMP
        self.EVAL_PIECE_VALUES_LST = [0, 100, 320, 330, 500, 900, 0] # King 0 for material eval
        
        self.NMP_R_REDUCTION = 3
        self.NMP_MIN_DEPTH_THRESHOLD = 1 + self.NMP_R_REDUCTION
        self.NMP_MIN_MATERIAL_FOR_SIDE = self._PIECE_VALUES_LST_FOR_NMP_MATERIAL[chess.ROOK]

        self.TT_MOVE_ORDER_BONUS = 200000
        self.CAPTURE_BASE_ORDER_BONUS = 100000
        self.QUEEN_PROMO_ORDER_BONUS = 90000
        self.MINOR_PROMO_ORDER_BONUS = 30000
        self.KILLER_1_ORDER_BONUS = 80000
        self.KILLER_2_ORDER_BONUS = 70000


    def init_zobrist_tables(self):
        random.seed(42)
        self.zobrist_piece_square = [[[random.getrandbits(64) for _ in range(12)] for _ in range(64)]]
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        self.zobrist_ep = [random.getrandbits(64) for _ in range(8)]
        self.zobrist_side = random.getrandbits(64)

    def compute_hash(self, board: chess.Board) -> int:
        h = 0
        for sq in CHESS_SQUARES:
            piece = board.piece_at(sq)
            if piece:
                piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                h ^= self.zobrist_piece_square[0][sq][piece_idx]
        h ^= self.zobrist_castling[board.castling_rights & 0xF]
        if board.ep_square is not None:
            h ^= self.zobrist_ep[chess.square_file(board.ep_square)]
        if board.turn == chess.BLACK:
            h ^= self.zobrist_side
        return h

    def get_side_material(self, board: chess.Board, color: chess.Color) -> int:
        material = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            material += len(board.pieces(piece_type, color)) * self._PIECE_VALUES_LST_FOR_NMP_MATERIAL[piece_type]
        return material

    def get_mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        attacker = board.piece_at(move.from_square)
        victim_piece_type_at_to_square = board.piece_at(move.to_square)
        
        if board.is_en_passant(move): victim_piece_type = chess.PAWN
        elif victim_piece_type_at_to_square: victim_piece_type = victim_piece_type_at_to_square.piece_type
        else: return 0
        if not attacker: return 0 
        return victim_piece_type * 10 - attacker.piece_type

    def get_move_score(self, board: chess.Board, move: chess.Move, tt_move: Optional[chess.Move], ply: int, white_checks_delivered: int, black_checks_delivered: int, max_checks: int, qsearch_mode: bool = False) -> int:
        score = 0
        if not qsearch_mode and tt_move and move == tt_move: score = self.TT_MOVE_ORDER_BONUS
        
        is_capture = board.is_capture(move)
        if is_capture: 
            score += self.CAPTURE_BASE_ORDER_BONUS + self.get_mvv_lva_score(board, move)
        
        if board.gives_check(move):
            num_checks_by_mover_after_move = (white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered) + 1
            checks_remaining = max(0, max_checks - num_checks_by_mover_after_move)
            check_bonus_factor = self.CHECK_BONUS_DECAY_BASE ** checks_remaining
            dynamic_check_bonus = self.CHECK_BONUS_ADDITIONAL_MAX * check_bonus_factor
            total_check_bonus = self.CHECK_BONUS_MIN + dynamic_check_bonus
            score += int(total_check_bonus)

        if not qsearch_mode:
            if move.promotion == chess.QUEEN: score += self.QUEEN_PROMO_ORDER_BONUS
            elif move.promotion: score += self.MINOR_PROMO_ORDER_BONUS
            if not is_capture: 
                if self.killer_moves[ply][0] == move: score += self.KILLER_1_ORDER_BONUS
                elif self.killer_moves[ply][1] == move: score += self.KILLER_2_ORDER_BONUS
                piece = board.piece_at(move.from_square)
                if piece: score += self.history_table.get((piece.piece_type, move.to_square), 0)
        return score

    def order_moves(self, board: chess.Board, legal_moves: chess.LegalMoveGenerator, tt_move: Optional[chess.Move], ply: int, white_checks: int, black_checks: int, max_checks: int, qsearch_mode: bool = False) -> List[chess.Move]:
        if qsearch_mode:
            def is_forcing_q_move(m: chess.Move) -> bool:
                if board.is_capture(m): return True
                if board.gives_check(m):
                    checks_before_move = white_checks if board.turn == chess.WHITE else black_checks
                    num_checks_after = checks_before_move + 1
                    if num_checks_after >= max_checks -1 : return True
                return False
            moves_to_consider_gen = (m for m in legal_moves if is_forcing_q_move(m))
        else:
            moves_to_consider_gen = legal_moves 
        
        moves_to_consider = list(moves_to_consider_gen)
        return sorted(moves_to_consider, key=lambda m: self.get_move_score(board, m, tt_move, ply, white_checks, black_checks, max_checks, qsearch_mode), reverse=True)

    def evaluate_position(self, board: chess.Board, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> int:
        if white_checks_delivered >= max_checks: return self.WIN_SCORE
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE
        
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE 
            if outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE
            return 0 

        material = 0; positional = 0
        for sq in CHESS_SQUARES:
            piece = board.piece_at(sq)
            if piece:
                value = self.EVAL_PIECE_VALUES_LST[piece.piece_type] # Original scale piece value
                pst_val = PST[piece.piece_type][sq if piece.color == chess.WHITE else chess.square_mirror(sq)] # Original scale PST value
                if piece.color == chess.WHITE:
                    material += value; positional += pst_val
                else:
                    material -= value; positional -= pst_val
        
        effective_max_checks = max(1, max_checks) 
        bonus_per_net_check = self.TOTAL_STATIC_CHECK_VALUE / effective_max_checks
        check_score_component = int(bonus_per_net_check * (white_checks_delivered - black_checks_delivered))
        current_eval = material + positional + check_score_component
        return current_eval

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int, maximizing_player: bool, white_checks: int, black_checks: int, max_checks: int, q_depth: int) -> int:
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
        
        forcing_moves = self.order_moves(board, board.legal_moves, None, 0, white_checks, black_checks, max_checks, qsearch_mode=True)
        for move in forcing_moves:
            current_w_checks, current_b_checks = white_checks, black_checks
            piece_color_that_moved = board.turn 
            board.push(move)
            if board.is_check(): 
                if piece_color_that_moved == chess.WHITE: current_w_checks += 1
                else: current_b_checks += 1
            score = self.quiescence_search(board, alpha, beta, not maximizing_player, current_w_checks, current_b_checks, max_checks, q_depth - 1)
            board.pop()
            if maximizing_player:
                alpha = max(alpha, score);
                if alpha >= beta: return beta 
            else: 
                beta = min(beta, score);
                if alpha >= beta: return alpha 
        return alpha if maximizing_player else beta

    def store_in_tt(self, key: int, depth: int, value: int, flag: int, best_move: Optional[chess.Move]):
        if len(self.transposition_table) >= self.tt_size_limit and self.tt_size_limit > 0:
            try: self.transposition_table.pop(next(iter(self.transposition_table))) 
            except StopIteration: pass
        if self.tt_size_limit > 0 :
             self.transposition_table[key] = {'depth': depth, 'value': value, 'flag': flag, 'best_move': best_move}

    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing_player: bool, ply: int, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> Tuple[int, Optional[chess.Move]]:
        alpha_orig = alpha 
        if white_checks_delivered >= max_checks: return self.WIN_SCORE, None
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE, None

        tt_key = self.compute_hash(board)
        tt_entry = self.transposition_table.get(tt_key)
        tt_move: Optional[chess.Move] = None
        if tt_entry and tt_entry['depth'] >= depth:
            tt_move = tt_entry['best_move']
            if tt_entry['flag'] == TT_EXACT: return tt_entry['value'], tt_move
            elif tt_entry['flag'] == TT_LOWERBOUND: alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == TT_UPPERBOUND: beta = min(beta, tt_entry['value'])
            if alpha >= beta: return tt_entry['value'], tt_move
        
        if depth <= 0:
            self.nodes_evaluated +=1 
            return self.quiescence_search(board, alpha, beta, maximizing_player, white_checks_delivered, black_checks_delivered, max_checks, self.MAX_Q_DEPTH), None
        
        outcome = board.outcome(claim_draw=True)
        if outcome:
            if outcome.winner == chess.WHITE: return self.CHECKMATE_SCORE, None
            elif outcome.winner == chess.BLACK: return -self.CHECKMATE_SCORE, None
            else: return 0, None 

        if (depth >= self.NMP_MIN_DEPTH_THRESHOLD and not board.is_check() and ply > 0 and
            self.get_side_material(board, board.turn) >= self.NMP_MIN_MATERIAL_FOR_SIDE):
            board.push(chess.Move.null())
            null_move_score, _ = self.minimax(board, depth - 1 - self.NMP_R_REDUCTION, -beta, -beta + 1, not maximizing_player, ply + 1, white_checks_delivered, black_checks_delivered, max_checks)
            board.pop(); null_move_score = -null_move_score 
            if null_move_score >= beta: return beta, None 

        best_move_for_node: Optional[chess.Move] = None
        ordered_moves = self.order_moves(board, board.legal_moves, tt_move, ply, white_checks_delivered, black_checks_delivered, max_checks)
        
        if not ordered_moves: 
            if board.is_check(): return (-self.CHECKMATE_SCORE if maximizing_player else self.CHECKMATE_SCORE), None
            else: return 0, None

        best_val_for_node = -float('inf') if maximizing_player else float('inf')
        for move_idx, move in enumerate(ordered_moves):
            current_w_checks, current_b_checks = white_checks_delivered, black_checks_delivered
            piece_color_that_moved = board.turn
            board.push(move)
            if board.is_check(): 
                if piece_color_that_moved == chess.WHITE: current_w_checks += 1
                else: current_b_checks += 1
            eval_val, _ = self.minimax(board, depth - 1, alpha, beta, not maximizing_player, ply + 1, current_w_checks, current_b_checks, max_checks)
            board.pop()
            if maximizing_player:
                if eval_val > best_val_for_node: best_val_for_node = eval_val; best_move_for_node = move
                alpha = max(alpha, eval_val)
                if beta <= alpha: 
                    if not board.is_capture(move) and move.promotion is None:
                        piece_that_moved_obj = board.piece_at(move.from_square) 
                        if piece_that_moved_obj: 
                            if move != self.killer_moves[ply][0]: 
                                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                                self.killer_moves[ply][0] = move
                            self.history_table[(piece_that_moved_obj.piece_type, move.to_square)] = \
                                self.history_table.get((piece_that_moved_obj.piece_type, move.to_square), 0) + depth**2
                    break 
            else: 
                if eval_val < best_val_for_node: best_val_for_node = eval_val; best_move_for_node = move
                beta = min(beta, eval_val)
                if beta <= alpha: 
                    if not board.is_capture(move) and move.promotion is None:
                        piece_that_moved_obj = board.piece_at(move.from_square)
                        if piece_that_moved_obj:
                            if move != self.killer_moves[ply][0]:
                                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                                self.killer_moves[ply][0] = move
                            self.history_table[(piece_that_moved_obj.piece_type, move.to_square)] = \
                                self.history_table.get((piece_that_moved_obj.piece_type, move.to_square), 0) + depth**2
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
        
        for d_iterative in range(1, depth + 1):
            value, move_at_this_depth = self.minimax(board, d_iterative, -float('inf'), float('inf'), is_maximizing_player, 0, white_checks, black_checks, max_checks)
            if move_at_this_depth is not None:
                best_move_overall = move_at_this_depth
                final_value_for_best_move = value
            elif best_move_overall is None: 
                final_value_for_best_move = value 
            self.current_eval_for_ui = final_value_for_best_move 
            if abs(value) >= self.WIN_SCORE or abs(value) >= (self.CHECKMATE_SCORE - (d_iterative * 200)):
                break
        
        if best_move_overall is None and list(board.legal_moves) and \
           white_checks < max_checks and black_checks < max_checks and \
           not board.is_game_over(claim_draw=True):
            print("AI Warning: No best move found, picking first legal move.")
            best_move_overall = next(iter(board.legal_moves), None)
            if best_move_overall:
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
        self.ai_depth = 3 
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
                if item_id not in self.original_square_colors: self.original_square_colors[item_id] = self.canvas.itemcget(item_id,"fill")
                self.canvas.itemconfig(item_id, fill=self.colors['highlight_selected_fill']); self.highlighted_rect_id = item_id
                self.canvas.tag_raise("piece_outline"); self.canvas.tag_raise("piece_fg")
            except tk.TclError: self.highlighted_rect_id = None

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
            if "wins" in current_status or "Draw" in current_status or "Checkmate" in current_status: status_text = current_status
        
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
                display_eval_score_internal = self.ai.evaluate_position(self.board, self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS)
            
            display_score_ui_units = display_eval_score_internal / 100.0

            if abs(display_eval_score_internal) >= self.ai.WIN_SCORE: 
                eval_text = "Eval: White winning" if display_eval_score_internal > 0 else "Eval: Black winning"
            elif abs(display_eval_score_internal) >= (self.ai.CHECKMATE_SCORE - (self.ai_depth * 200)):
                eval_text = "Eval: Mate imminent"
            else: 
                eval_text = f"Eval: {display_score_ui_units:+.2f}"
        else: 
            eval_text = "Eval: ---"

        if status_text != self.status_label.cget("text"): self.status_label.config(text=status_text)
        if eval_text != self.eval_label.cget("text"): self.eval_label.config(text=eval_text)

    def reset_game(self):
        self.board.reset(); self.game_over_flag=False; self.MAX_CHECKS=NUM_CHECKS_TO_WIN
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
        self.update_ai_depth(self.ai_depth)
        self.on_canvas_resize()

    def flip_board(self):
        self.flipped=not self.flipped; self._clear_highlights(); self.drag_selected_square = None 
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
        self.redraw_board_and_pieces()

    def undo_last_player_ai_moves(self):
        if self.ai_thinking: return
        moves_to_undo=0
        if self.board.turn==chess.WHITE and len(self.board.move_stack)>=2: moves_to_undo=2
        elif self.board.turn==chess.BLACK and len(self.board.move_stack)>=1: moves_to_undo=1
        elif len(self.board.move_stack)>0: moves_to_undo=1
        undone_count=0
        for _ in range(moves_to_undo):
            if self.board.move_stack:
                self.board.pop()
                if self.check_history: self.white_checks_delivered,self.black_checks_delivered=self.check_history.pop(); undone_count+=1
                else: self.white_checks_delivered=0;self.black_checks_delivered=0; break 
            else: break
        if undone_count>0:
            self.game_over_flag=False
            self._clear_highlights(); self.drag_selected_square=None
            if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id=None
            self.redraw_board_and_pieces(); self.update_status_label(for_ai_move_eval=None)

    def update_ai_depth(self,val):
        val_int = int(round(float(val))); self.ai_depth=val_int
        if hasattr(self,'depth_slider') and abs(self.depth_slider.get()-val_int)>0.01: self.depth_slider.set(val_int)
        if hasattr(self,'depth_value_label'): self.depth_value_label.config(text=str(val_int))

    def on_square_interaction_start(self,event):
        if self.game_over_flag or self.board.turn!=chess.WHITE or self.ai_thinking: return
        clicked_sq = self._get_canvas_xy_to_square(event.x,event.y)
        if not clicked_sq: return
        if self.drag_selected_square and self.drag_selected_square!=clicked_sq: self._attempt_player_move(self.drag_selected_square,clicked_sq); return
        self._clear_highlights()
        piece = self.board.piece_at(clicked_sq)
        if piece and piece.color==self.board.turn: 
            self.drag_selected_square=clicked_sq; self.dragging_piece_item_id=self.piece_item_ids.get(clicked_sq)
            if self.dragging_piece_item_id:
                outline_tags = self.canvas.find_withtag(f"piece_outline_{clicked_sq}")
                for item_id_outline in outline_tags: self.canvas.lift(item_id_outline)
                self.canvas.lift(self.dragging_piece_item_id)
                coords = self.canvas.coords(self.dragging_piece_item_id)
                if coords: 
                    self.drag_start_x_offset=event.x-coords[0]; self.drag_start_y_offset=event.y-coords[1]
                    nx,ny = event.x-self.drag_start_x_offset, event.y-self.drag_start_y_offset
                    self.canvas.coords(self.dragging_piece_item_id,nx,ny)
                    for item_id_outline in outline_tags: self.canvas.coords(item_id_outline,nx,ny)
            self._show_legal_moves_for_selected_piece()
        else: self.drag_selected_square=None; self.dragging_piece_item_id=None

    def on_piece_drag_motion(self,event):
        if self.dragging_piece_item_id and self.drag_selected_square and not self.ai_thinking:
            nx,ny = event.x-self.drag_start_x_offset, event.y-self.drag_start_y_offset
            self.canvas.coords(self.dragging_piece_item_id,nx,ny)
            for item_id_outline in self.canvas.find_withtag(f"piece_outline_{self.drag_selected_square}"):
                 self.canvas.coords(item_id_outline,nx,ny)

    def on_square_interaction_end(self,event):
        if not self.drag_selected_square or self.ai_thinking: 
            if self.dragging_piece_item_id: self.redraw_board_and_pieces()
            self._clear_highlights(); self.drag_selected_square = None; self.dragging_piece_item_id = None
            return
        to_sq = self._get_canvas_xy_to_square(event.x,event.y)
        self.redraw_board_and_pieces(); self._clear_highlights()      
        if to_sq and self.drag_selected_square!=to_sq: self._attempt_player_move(self.drag_selected_square,to_sq)
        else: self.drag_selected_square=None; self.dragging_piece_item_id=None

    def _attempt_player_move(self,from_sq:chess.Square,to_sq:chess.Square):
        promo=None; p=self.board.piece_at(from_sq)
        if p and p.piece_type==chess.PAWN:
            if (p.color==chess.WHITE and chess.square_rank(to_sq)==7) or \
               (p.color==chess.BLACK and chess.square_rank(to_sq)==0): promo=chess.QUEEN
        move=chess.Move(from_sq,to_sq,promotion=promo)
        self.drag_selected_square=None; self.dragging_piece_item_id=None
        if move in self.board.legal_moves:
            self.check_history.append((self.white_checks_delivered,self.black_checks_delivered))
            self.board.push(move)
            if self.board.is_check() and self.board.turn==chess.BLACK: self.white_checks_delivered+=1
            self.redraw_board_and_pieces()
            if not self.game_over_flag: 
                self.update_status_label(for_ai_move_eval=None) 
                self.root.update_idletasks(); self._start_ai_move_thread()
        else: self.redraw_board_and_pieces()

    def _start_ai_move_thread(self):
        if self.ai_thinking: return
        self.ai_thinking = True
        self.status_label.config(text="AI is thinking...")
        self.eval_label.config(text="Eval: Thinking...") 
        if hasattr(self, 'depth_slider'): self.depth_slider.config(state=tk.DISABLED)
        ai_thread = threading.Thread(target=self._ai_move_worker, 
            args=(self.board.copy(), self.ai_depth, self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS), daemon=True)
        ai_thread.start()

    def _ai_move_worker(self, board_copy: chess.Board, depth: int, w_checks: int, b_checks: int, max_c: int):
        start_time = time.monotonic()
        ai_move = self.ai.find_best_move(board_copy, depth, w_checks, b_checks, max_c)
        eval_after_search_internal = self.ai.current_eval_for_ui 
        elapsed = time.monotonic()-start_time
        total_nodes = self.ai.nodes_evaluated + self.ai.q_nodes_evaluated
        nps = total_nodes/elapsed if elapsed>0 else 0
        tt_fill = (len(self.ai.transposition_table)/self.ai.tt_size_limit)*100 if self.ai.tt_size_limit>0 else 0

        ai_player_color_char = 'White' if board_copy.turn == chess.WHITE else 'Black'
        
        print_info = (f"AI playing ({ai_player_color_char}) | Eval: {eval_after_search_internal/100.0:+.2f} | "
                      f"Time:{elapsed:.2f}s | Depth:{depth} | Nodes:{total_nodes} (NPS:{nps:.0f}) | TT Fill:{tt_fill:.1f}%")
        
        self.ai_move_queue.put((ai_move, print_info, eval_after_search_internal))
    def _check_ai_queue_periodically(self):
        try:
            ai_move, print_info, eval_score_from_ai_internal = self.ai_move_queue.get_nowait()
            self._process_ai_move_from_queue(ai_move, print_info, eval_score_from_ai_internal)
        except queue.Empty: pass 
        finally: self.root.after(100, self._check_ai_queue_periodically)

    def _process_ai_move_from_queue(self, ai_move: Optional[chess.Move], print_info: str, eval_score_from_ai_search_internal: int):
        self.ai_thinking = False
        if hasattr(self, 'depth_slider'): self.depth_slider.config(state=tk.NORMAL)
        print(print_info)
        if self.game_over_flag: self.update_status_label(); return

        if ai_move:
            self.check_history.append((self.white_checks_delivered,self.black_checks_delivered))
            if ai_move in self.board.legal_moves:
                self.board.push(ai_move)
                if self.board.is_check() and self.board.turn==chess.WHITE: 
                    self.black_checks_delivered+=1
            else: 
                print(f"AI Warning: Proposed move {ai_move.uci()} is illegal on current board!")
                if not list(self.board.legal_moves): self.game_over_flag = True
            
            self.redraw_board_and_pieces()
            self.update_status_label(for_ai_move_eval=eval_score_from_ai_search_internal) 
        else: 
            self.update_status_label()
            self.game_over_flag = True
        
        if not self.game_over_flag: 
            self.update_status_label(for_ai_move_eval=None) 

if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()