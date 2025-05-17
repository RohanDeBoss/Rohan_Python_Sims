import tkinter as tk
from tkinter import ttk, font
import chess
import time
import random
from typing import Tuple, List, Optional, Dict, Any
import threading
import queue # For thread-safe communication

# --- Global Configuration ---
NUM_CHECKS_TO_WIN = 3
INITIAL_SQUARE_SIZE = 66

INITIAL_WINDOW_WIDTH = (8 * INITIAL_SQUARE_SIZE) + 290 
INITIAL_WINDOW_HEIGHT = (8 * INITIAL_SQUARE_SIZE) + 29 # Slightly more height for eval label
CONTROL_PANEL_DEFAULT_WIDTH = INITIAL_WINDOW_WIDTH - (8 * INITIAL_SQUARE_SIZE) - 37 
TT_SIZE_POWER_OF_2 = 18 # 2^18 = 262144 entries 
# ----------------------------

# Piece-Square Tables (PSTs)
PST = {
    chess.PAWN: [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [ 20, 30, 10,  0,  0, 10, 30, 20, 20, 20,  0,  0,  0,  0, 20, 20,-10,-20,-20,-20,-20,-20,-20,-10,-20,-30,-30,-40,-40,-30,-30,-20,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30]
}
TT_EXACT = 0; TT_LOWERBOUND = 1; TT_UPPERBOUND = 2

class ChessAI: # (Same as previous version)
    def __init__(self):
        self.transposition_table: Dict[int, Dict[str, Any]] = {}
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(64)]
        self.history_table: Dict[Tuple[chess.PieceType, chess.Square], int] = {}
        self.tt_size_limit = 2**TT_SIZE_POWER_OF_2 
        self.nodes_evaluated = 0
        self.q_nodes_evaluated = 0 
        self.prev_best_move: Optional[chess.Move] = None
        self.current_eval_for_ui = 0 # For UI to fetch static eval of current board
        self.init_zobrist_tables()
        self.WIN_SCORE = 1000000; self.LOSS_SCORE = -1000000
        self.CHECKMATE_SCORE = 900000; self.CHECK_PROGRESS_BONUS = 1000 
        self.MAX_Q_DEPTH = 5 
        self.PIECE_VALUES_FOR_EVAL = { chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

    def init_zobrist_tables(self): 
        random.seed(42); self.zobrist_piece_square = [[[random.getrandbits(64) for _ in range(12)] for _ in range(64)]]
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]; self.zobrist_ep = [random.getrandbits(64) for _ in range(8)]
        self.zobrist_side = random.getrandbits(64)
    def compute_hash(self, board: chess.Board) -> int: 
        h = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece: h ^= self.zobrist_piece_square[0][sq][piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)]
        h ^= self.zobrist_castling[board.castling_rights & 0xF]
        if board.ep_square is not None: h ^= self.zobrist_ep[chess.square_file(board.ep_square)]
        if board.turn == chess.BLACK: h ^= self.zobrist_side
        return h
    def get_mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int: 
        if not board.is_capture(move): return 0
        attacker = board.piece_at(move.from_square); 
        victim_piece = board.piece_at(move.to_square)
        victim_type = chess.PAWN if board.is_en_passant(move) else (victim_piece.piece_type if victim_piece else None)
        if not attacker or not victim_type: return 0
        v = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
        return v[victim_type] * 100 - v[attacker.piece_type]
    
    def get_move_score(self, board: chess.Board, move: chess.Move, tt_move: Optional[chess.Move], ply: int, white_checks_delivered: int, black_checks_delivered: int, max_checks: int, qsearch_mode: bool = False) -> int:
        score = 0
        if not qsearch_mode and tt_move and move == tt_move: score = 200000 
        gives_check = board.gives_check(move)
        if gives_check:
            num_checks_by_mover = white_checks_delivered if board.turn == chess.WHITE else black_checks_delivered
            if (num_checks_by_mover + 1) >= max_checks: score += 150000 
            elif (num_checks_by_mover + 1) == 2: score += 70000  
            elif (num_checks_by_mover + 1) == 1: score += 30000  
            else: score += 10000 
        if board.is_capture(move): score += 100000 + self.get_mvv_lva_score(board, move) 
        if not qsearch_mode: 
            if move.promotion == chess.QUEEN: score += 90000
            elif move.promotion: score += 30000 
            if self.killer_moves[ply][0] == move: score += 80000
            elif self.killer_moves[ply][1] == move: score += 70000
            piece = board.piece_at(move.from_square)
            if piece: score += self.history_table.get((piece.piece_type, move.to_square), 0)
        return score

    def order_moves(self, board: chess.Board, legal_moves: chess.LegalMoveGenerator, tt_move: Optional[chess.Move], ply: int, white_checks: int, black_checks: int, max_checks: int, qsearch_mode: bool = False) -> List[chess.Move]:
        if qsearch_mode:
            moves_to_consider = [m for m in legal_moves if board.is_capture(m) or board.gives_check(m)]
        else:
            moves_to_consider = list(legal_moves)
        return sorted(moves_to_consider, key=lambda m: self.get_move_score(board, m, tt_move, ply, white_checks, black_checks, max_checks, qsearch_mode), reverse=True)

    def evaluate_position(self, board: chess.Board, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> int: 
        if white_checks_delivered >= max_checks: return self.WIN_SCORE
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE
        if board.is_checkmate(): return -self.CHECKMATE_SCORE if board.turn == chess.WHITE else self.CHECKMATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition(): return 0
        material = 0; positional = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                value = self.PIECE_VALUES_FOR_EVAL[piece.piece_type]; pst_val = 0
                if piece.piece_type in PST: pst_val = PST[piece.piece_type][sq if piece.color == chess.WHITE else chess.square_mirror(sq)]
                if piece.color == chess.WHITE: material += value; positional += pst_val
                else: material -= value; positional -= pst_val
        return material + positional + self.CHECK_PROGRESS_BONUS * (white_checks_delivered - black_checks_delivered)

    def quiescence_search(self, board: chess.Board, alpha: int, beta: int, maximizing_player: bool, white_checks: int, black_checks: int, max_checks: int, q_depth: int) -> int: 
        self.q_nodes_evaluated += 1
        if white_checks >= max_checks: return self.WIN_SCORE
        if black_checks >= max_checks: return self.LOSS_SCORE
        outcome = board.outcome(claim_draw=True)
        if outcome is not None: return self.CHECKMATE_SCORE if outcome.winner == chess.WHITE else (-self.CHECKMATE_SCORE if outcome.winner == chess.BLACK else 0)
        if q_depth <= 0: return self.evaluate_position(board, white_checks, black_checks, max_checks)
        stand_pat_score = self.evaluate_position(board, white_checks, black_checks, max_checks)
        if maximizing_player:
            if stand_pat_score >= beta: return beta
            alpha = max(alpha, stand_pat_score)
        else:
            if stand_pat_score <= alpha: return alpha
            beta = min(beta, stand_pat_score)
        forcing_moves = self.order_moves(board, board.legal_moves, None, 0, white_checks, black_checks, max_checks, qsearch_mode=True)
        for move in forcing_moves:
            board.push(move)
            score = self.quiescence_search(board, alpha, beta, not maximizing_player, white_checks, black_checks, max_checks, q_depth - 1)
            board.pop()
            if maximizing_player: alpha = max(alpha, score); 
            else: beta = min(beta, score); 
            if alpha >= beta : return beta if maximizing_player else alpha 
        return alpha if maximizing_player else beta
    def store_in_tt(self, key: int, depth: int, value: int, flag: int, best_move: Optional[chess.Move]): 
        if len(self.transposition_table) >= self.tt_size_limit:
            try: self.transposition_table.pop(next(iter(self.transposition_table)))
            except StopIteration: pass
        self.transposition_table[key] = {'depth': depth, 'value': value, 'flag': flag, 'best_move': best_move}
    def minimax(self, board: chess.Board, depth: int, alpha: int, beta: int, maximizing_player: bool, ply: int, white_checks_delivered: int, black_checks_delivered: int, max_checks: int) -> Tuple[int, Optional[chess.Move]]: 
        alpha_orig = alpha
        # current_eval = self.evaluate_position(board, white_checks_delivered, black_checks_delivered, max_checks) # Not needed here
        if white_checks_delivered >= max_checks: return self.WIN_SCORE, None
        if black_checks_delivered >= max_checks: return self.LOSS_SCORE, None
        outcome = board.outcome(claim_draw=True)
        if outcome is not None: return (self.CHECKMATE_SCORE if outcome.winner == chess.WHITE else (-self.CHECKMATE_SCORE if outcome.winner == chess.BLACK else 0)), None
        if depth <= 0:
            self.nodes_evaluated +=1 
            q_score = self.quiescence_search(board, alpha, beta, maximizing_player, white_checks_delivered, black_checks_delivered, max_checks, self.MAX_Q_DEPTH)
            return q_score, None
        tt_key = self.compute_hash(board); tt_entry = self.transposition_table.get(tt_key); tt_move: Optional[chess.Move] = None
        if tt_entry and tt_entry['depth'] >= depth:
            tt_move = tt_entry['best_move']
            if tt_entry['flag'] == TT_EXACT: return tt_entry['value'], tt_move
            elif tt_entry['flag'] == TT_LOWERBOUND: alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == TT_UPPERBOUND: beta = min(beta, tt_entry['value'])
            if alpha >= beta: return tt_entry['value'], tt_move
        best_move_for_node: Optional[chess.Move] = None
        ordered_moves = self.order_moves(board, board.legal_moves, tt_move, ply, white_checks_delivered, black_checks_delivered, max_checks, qsearch_mode=False)
        if not ordered_moves: 
            # This should only happen in a terminal position (mate/stalemate), already caught by outcome.
            # If somehow reached, return static eval of current board.
            return self.evaluate_position(board, white_checks_delivered, black_checks_delivered, max_checks), None

        best_val_for_node = -float('inf') if maximizing_player else float('inf')
        if maximizing_player:
            for move in ordered_moves:
                current_w_checks, current_b_checks = white_checks_delivered, black_checks_delivered; board.push(move)
                if board.is_check(): current_w_checks += 1
                eval_val, _ = self.minimax(board, depth - 1, alpha, beta, False, ply + 1, current_w_checks, current_b_checks, max_checks); board.pop()
                if eval_val > best_val_for_node: best_val_for_node = eval_val; best_move_for_node = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    if not board.is_capture(move) and move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]; self.killer_moves[ply][0] = move
                        piece = board.piece_at(move.from_square)
                        if piece: self.history_table[(piece.piece_type, move.to_square)] = self.history_table.get((piece.piece_type, move.to_square),0) + depth**2
                    break
        else: # Minimizing Player
            for move in ordered_moves:
                current_w_checks, current_b_checks = white_checks_delivered, black_checks_delivered; board.push(move)
                if board.is_check(): current_b_checks += 1
                eval_val, _ = self.minimax(board, depth - 1, alpha, beta, True, ply + 1, current_w_checks, current_b_checks, max_checks); board.pop()
                if eval_val < best_val_for_node: best_val_for_node = eval_val; best_move_for_node = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    if not board.is_capture(move) and move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]; self.killer_moves[ply][0] = move
                        piece = board.piece_at(move.from_square)
                        if piece: self.history_table[(piece.piece_type, move.to_square)] = self.history_table.get((piece.piece_type, move.to_square),0) + depth**2
                    break
        flag = TT_EXACT
        if best_val_for_node <= alpha_orig: flag = TT_UPPERBOUND 
        elif best_val_for_node >= beta: flag = TT_LOWERBOUND
        if best_move_for_node is not None: self.store_in_tt(tt_key, depth, best_val_for_node, flag, best_move_for_node)
        return best_val_for_node, best_move_for_node

    def find_best_move(self, board: chess.Board, depth: int, white_checks: int, black_checks: int, max_checks: int) -> Optional[chess.Move]: 
        self.nodes_evaluated = 0; self.q_nodes_evaluated = 0; best_move_iterative: Optional[chess.Move] = None
        self.current_eval_for_ui = 0 # Reset before search for UI eval
        is_maximizing_player = (board.turn == chess.WHITE)
        final_value_for_best_move = 0

        for d_iterative in range(1, depth + 1):
            value, move = self.minimax(board, d_iterative, -float('inf'), float('inf'), is_maximizing_player, 0, white_checks, black_checks, max_checks)
            if move: 
                best_move_iterative = move; self.prev_best_move = move 
                final_value_for_best_move = value 
            if abs(value) >= self.CHECKMATE_SCORE - 1000: break
        
        # After search, current_eval_for_ui should reflect the value of the chosen path.
        # If AI is White, this value is from White's perspective.
        # If AI is Black, this value is from Black's perspective (minimize White's score).
        # For UI display (always White's perspective), adjust if AI is Black.
        self.current_eval_for_ui = final_value_for_best_move

        if best_move_iterative is None and list(board.legal_moves):
            print("AI Warning: No best move from minimax, picking random."); best_move_iterative = random.choice(list(board.legal_moves))
            # If picking random, get a static eval for the current board state
            self.current_eval_for_ui = self.evaluate_position(board, white_checks, black_checks, max_checks)
            # No need to adjust for whose turn it is here, evaluate_position is always from White's perspective.
        return best_move_iterative

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
        self.highlighted_rect_id: Optional[int] = None # For selected square highlight

        self.piece_images: Dict[str, str] = {}
        self.ai = ChessAI()
        self.ai_move_queue = queue.Queue()

        self.colors = {
            'square_light': '#E0C4A0',     
            'square_dark': '#A08060',      
            'highlight_selected_fill': '#6FA8DC', 
            'highlight_capture': '#FF7F7F80',  
            'highlight_normal_move': '#90EE9080', 
            'piece_white': '#FFFFFF',       
            'piece_black': '#202020',
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
        s = ttk.Style()
        s.theme_use('clam') 
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
        control_panel.grid(row=0, column=1, sticky="ns", padx=(8,5)) 
        control_panel.grid_propagate(False)
        main_frame.grid_columnconfigure(1, weight=0) 

        row_idx = 0; btn_width = 22 

        ttk.Button(control_panel, text="New Game", command=self.reset_game, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        ttk.Button(control_panel, text="Flip Board", command=self.flip_board, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        ttk.Button(control_panel, text="Undo", command=self.undo_last_player_ai_moves, width=btn_width).grid(row=row_idx, column=0, columnspan=2, pady=4, sticky="ew"); row_idx+=1
        
        self.checks_labelframe = ttk.LabelFrame(control_panel, text=f"Checks (Goal: {self.MAX_CHECKS})")
        self.checks_labelframe.grid(row=row_idx, column=0,columnspan=2, pady=(8,5), sticky="ew"); row_idx+=1
        
        inner_checks_frame = ttk.Frame(self.checks_labelframe)
        inner_checks_frame.pack(fill=tk.X, expand=True, padx=4, pady=(2,4))
        inner_checks_frame.columnconfigure(0, weight=1); inner_checks_frame.columnconfigure(1, weight=1)

        self.white_checks_label = ttk.Label(inner_checks_frame, text="W:", font=(self.piece_font_family, 11), anchor="w") 
        self.white_checks_label.grid(row=0, column=0, sticky="ew", padx=(3,2)) 
        self.black_checks_label = ttk.Label(inner_checks_frame, text="B:", font=(self.piece_font_family, 11), anchor="w")
        self.black_checks_label.grid(row=0, column=1, sticky="ew", padx=(2,3)) 
        
        depth_frame = ttk.Frame(control_panel)
        depth_frame.grid(row=row_idx, column=0, columnspan=2, pady=(5,0), sticky="ew"); row_idx+=1
        ttk.Label(depth_frame, text="AI Depth:", font=self.base_font).pack(side=tk.LEFT, padx=(0,5))
        self.depth_value_label = ttk.Label(depth_frame, text=str(self.ai_depth), font=self.base_font, width=2, anchor="e")
        self.depth_value_label.pack(side=tk.RIGHT, padx=(5,0))

        self.depth_slider = ttk.Scale(control_panel, from_=1, to=6, orient=tk.HORIZONTAL, command=self.update_ai_depth) 
        self.depth_slider.set(self.ai_depth)
        self.depth_slider.grid(row=row_idx, column=0, columnspan=2, pady=(0,5), sticky="ew"); row_idx+=1
        
        self.status_label = ttk.Label(control_panel, text="White's turn", font=self.bold_font, wraplength=200, anchor="center", justify="center", width=26) 
        self.status_label.grid(row=row_idx, column=0,columnspan=2, pady=(4,2), sticky="ew", ipady=3); row_idx+=1 # Reduced pady and ipady
        
        self.eval_label = ttk.Label(control_panel, text="Eval: +0.00", font=self.base_font, wraplength=200, anchor="center", justify="center", width=26)
        self.eval_label.grid(row=row_idx, column=0, columnspan=2, pady=(0,8), sticky="ew", ipady=2); row_idx+=1 # Added Eval Label

        ttk.Button(control_panel, text="Exit", command=self.root.quit, width=btn_width).grid(row=row_idx, column=0,columnspan=2, pady=4, sticky="ew"); row_idx+=1
        
        control_panel.grid_rowconfigure(row_idx, weight=1) 

    def load_piece_images(self):
        self.piece_images = {'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'}

    def on_canvas_resize(self, event=None): 
        new_canvas_width = event.width if event else self.canvas.winfo_width()
        new_canvas_height = event.height if event else self.canvas.winfo_height()
        if new_canvas_width < 16 or new_canvas_height < 16: return 
        new_square_size = max(10, min(new_canvas_width // 8, new_canvas_height // 8)) 
        if new_square_size != self.square_size: 
            self.square_size = new_square_size
            self.redraw_board_and_pieces()
        elif event is None : 
             self.redraw_board_and_pieces()

    def _get_square_coords(self, square: chess.Square) -> Tuple[int, int, int, int]: 
        file, rank = chess.square_file(square), chess.square_rank(square)
        if self.flipped: file, rank = 7 - file, rank
        else: rank = 7 - rank
        x1, y1 = file * self.square_size, rank * self.square_size
        return x1, y1, x1 + self.square_size, y1 + self.square_size
    def _get_canvas_xy_to_square(self, canvas_x: int, canvas_y: int) -> Optional[chess.Square]: 
        if self.square_size == 0: return None
        file_idx, rank_idx = canvas_x // self.square_size, canvas_y // self.square_size
        if not (0 <= file_idx < 8 and 0 <= rank_idx < 8): return None
        file = 7 - file_idx if self.flipped else file_idx; rank = rank_idx if self.flipped else 7 - rank_idx
        return chess.square(file, rank) if (0 <= file < 8 and 0 <= rank < 8) else None
    
    def _draw_squares(self): 
        self.canvas.delete("square_bg")
        self.square_item_ids.clear() 
        for r_chess in range(8):
            for f_chess in range(8):
                sq = chess.square(f_chess, r_chess); x1, y1, x2, y2 = self._get_square_coords(sq)
                color = self.colors['square_light'] if (f_chess + r_chess) % 2 == 0 else self.colors['square_dark']
                # Draw square with outline="" or width=0 to make fill precise
                item_id = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", width=0, tags=("square_bg", f"sq_bg_{sq}"))
                self.square_item_ids[sq] = item_id 

    def _draw_all_pieces(self): 
        self.canvas.delete("piece_outline") 
        self.canvas.delete("piece_fg");      
        self.piece_item_ids.clear()
        font_size = int(self.square_size * 0.75) 
        outline_offset = 1 

        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                x1, y1, x2, y2 = self._get_square_coords(sq); cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                piece_color_name = self.colors['piece_white'] if piece.color == chess.WHITE else self.colors['piece_black']
                current_outline_color = self.colors['piece_black'] if piece.color == chess.WHITE else self.colors['piece_white']
                for dx_o, dy_o in [(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)]:
                    self.canvas.create_text(cx + dx_o * outline_offset, cy + dy_o * outline_offset, 
                                            text=self.piece_images[piece.symbol()], 
                                            font=(self.piece_font_family, font_size), 
                                            fill=current_outline_color, tags=("piece_outline", f"piece_outline_{sq}"), anchor="c")
                item_id = self.canvas.create_text(cx, cy, text=self.piece_images[piece.symbol()], 
                                                  font=(self.piece_font_family, font_size), 
                                                  fill=piece_color_name, tags=("piece_fg", f"piece_fg_{sq}"), anchor="c")
                self.piece_item_ids[sq] = item_id 

    def redraw_board_and_pieces(self): 
        self._draw_squares(); self._draw_all_pieces(); self._update_check_display(); self.update_status_label()
    
    def _clear_highlights(self): 
        # Revert the fill color of the previously highlighted square item
        if self.highlighted_rect_id is not None and self.highlighted_rect_id in self.original_square_colors:
            try:
                self.canvas.itemconfig(self.highlighted_rect_id, fill=self.original_square_colors[self.highlighted_rect_id])
            except tk.TclError: pass # Item might have been deleted
            del self.original_square_colors[self.highlighted_rect_id]
            self.highlighted_rect_id = None
        
        self.canvas.delete("highlight_dot") # Delete legal move dots

    def _highlight_square_selected(self, square: chess.Square):
        # self._clear_highlights() # Already called by caller usually
        square_item_id = self.square_item_ids.get(square)
        if square_item_id:
            try:
                # Store the original color of the square_item_id before changing it
                if square_item_id not in self.original_square_colors: # Only store if not already stored (should be cleared)
                    self.original_square_colors[square_item_id] = self.canvas.itemcget(square_item_id, "fill")
                
                self.canvas.itemconfig(square_item_id, fill=self.colors['highlight_selected_fill'])
                self.highlighted_rect_id = square_item_id # Track which square_item_id is highlighted
                
                # Ensure pieces are drawn above this filled square
                self.canvas.tag_raise("piece_outline")
                self.canvas.tag_raise("piece_fg")
            except tk.TclError: 
                self.highlighted_rect_id = None # Item no longer exists

    def _highlight_legal_move(self, square: chess.Square, is_capture: bool): 
        x1, y1, x2, y2 = self._get_square_coords(square); cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dot_radius = self.square_size * 0.15 
        fill_color_rgb = self.colors['highlight_capture'][:7] if is_capture else self.colors['highlight_normal_move'][:7]
        self.canvas.create_oval(cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius, 
                                fill=fill_color_rgb, stipple=self.stipple_patterns['medium'], 
                                outline="", tags="highlight_dot")


    def _show_legal_moves_for_selected_piece(self): 
        self._clear_highlights() 
        if self.drag_selected_square is not None:
            self._highlight_square_selected(self.drag_selected_square) 
            for move in self.board.legal_moves:
                if move.from_square == self.drag_selected_square: 
                    self._highlight_legal_move(move.to_square, self.board.is_capture(move)) 
    
    def _update_check_display(self): 
        filled_marker, empty_marker = '✚', '⊕'
        max_markers_to_show = self.MAX_CHECKS 
        if self.MAX_CHECKS > 7: max_markers_to_show = 7 
        
        w_filled = self.white_checks_delivered
        w_empty = max(0, min(max_markers_to_show - w_filled, self.MAX_CHECKS - w_filled))
        w_text = "W: " + (filled_marker) * w_filled + (empty_marker) * w_empty
        if self.MAX_CHECKS - w_filled > w_empty and w_filled + w_empty >= max_markers_to_show : w_text += ".." 

        b_filled = self.black_checks_delivered
        b_empty = max(0, min(max_markers_to_show - b_filled, self.MAX_CHECKS - b_filled))
        b_text = "B: " + (filled_marker) * b_filled + (empty_marker) * b_empty
        if self.MAX_CHECKS - b_filled > b_empty and b_filled + b_empty >= max_markers_to_show : b_text += ".."

        self.white_checks_label.config(text=w_text.strip())
        self.black_checks_label.config(text=b_text.strip())
    
    def update_status_label(self, for_ai_move_eval: Optional[int] = None):
        status_text = ""
        eval_text = ""

        if self.game_over_flag:
            # Get existing game over message from status_label if already set
            current_status = self.status_label.cget("text").split("\n")[0]
            if "wins" in current_status or "Draw" in current_status:
                status_text = current_status # Keep existing game over message
            # else proceed to determine game over message if not already set appropriately
        
        if not status_text: # If not already set by above
            if self.white_checks_delivered >= self.MAX_CHECKS: 
                status_text = f"White wins by {self.MAX_CHECKS} checks!"
                self.game_over_flag = True
            elif self.black_checks_delivered >= self.MAX_CHECKS: 
                status_text = f"Black wins by {self.MAX_CHECKS} checks!"
                self.game_over_flag = True
            else:
                outcome = self.board.outcome(claim_draw=True)
                if outcome:
                    self.game_over_flag = True
                    if outcome.winner == chess.WHITE: status_text = "Checkmate! White wins!"
                    elif outcome.winner == chess.BLACK: status_text = "Checkmate! Black wins!"
                    else: status_text = f"Draw! ({outcome.termination.name.replace('_', ' ').title()})"
                else: 
                    status_text = ("White's turn" if self.board.turn == chess.WHITE else "Black's turn")
                    if self.board.is_check(): status_text += " (Check!)"
                    self.game_over_flag = False

        # Update Eval Label separately
        current_board_eval = self.ai.evaluate_position(self.board, self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS)
        
        # Use for_ai_move_eval if it's provided (comes from AI's search), otherwise use static eval
        display_eval_score = for_ai_move_eval if for_ai_move_eval is not None else current_board_eval

        if not self.game_over_flag or for_ai_move_eval is not None : # Show eval unless game is over by player's move
            if abs(display_eval_score) >= self.ai.CHECKMATE_SCORE - 1000: 
                    if display_eval_score > 0 : eval_text = "Eval: White winning"
                    else: eval_text = "Eval: Black winning"
            else: 
                pawn_value = display_eval_score / 100.0
                eval_text = f"Eval: {pawn_value:+.2f}"
        else: # Game is over and no specific AI eval to show (e.g., player made mating move)
            eval_text = "Eval: ---"


        if status_text != self.status_label.cget("text"):
            self.status_label.config(text=status_text)
        if eval_text != self.eval_label.cget("text"):
            self.eval_label.config(text=eval_text)


    def reset_game(self): 
        self.board.reset(); self.game_over_flag = False; self.MAX_CHECKS = NUM_CHECKS_TO_WIN
        self.root.title(f"{self.MAX_CHECKS}-Check Chess")
        if hasattr(self, 'checks_labelframe'): self.checks_labelframe.config(text=f"Checks (Goal: {self.MAX_CHECKS})")
        self.white_checks_delivered = 0; self.black_checks_delivered = 0; self.check_history.clear()
        self.ai.prev_best_move = None; self.ai.transposition_table.clear()
        self.ai.killer_moves = [[None, None] for _ in range(64)]; self.ai.history_table.clear()
        self.drag_selected_square = None; self.ai_thinking = False
        if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id = None
        self._clear_highlights(); 
        self.depth_slider.set(self.ai_depth) 
        self.update_ai_depth(self.ai_depth) 
        self.on_canvas_resize() # Calls redraw_board_and_pieces -> update_status_label
        self.update_status_label() # Explicit call to set initial eval

    def flip_board(self): self.flipped = not self.flipped; self.redraw_board_and_pieces() 
    def undo_last_player_ai_moves(self): 
        if self.ai_thinking: return 
        moves_to_undo = 0
        if self.board.turn == chess.WHITE and len(self.board.move_stack) >= 2: moves_to_undo = 2
        elif self.board.turn == chess.BLACK and len(self.board.move_stack) >= 1: moves_to_undo = 1
        elif len(self.board.move_stack) > 0 : moves_to_undo = 1
        undone_count = 0
        for _ in range(moves_to_undo):
            if self.board.move_stack: self.board.pop(); 
            if self.check_history: self.white_checks_delivered, self.black_checks_delivered = self.check_history.pop(); undone_count +=1
            else: break 
        if undone_count > 0:
            self.game_over_flag = False; self.ai.prev_best_move = None; self._clear_highlights(); self.drag_selected_square = None
            if self.dragging_piece_item_id: self.canvas.delete(self.dragging_piece_item_id); self.dragging_piece_item_id = None
            self.redraw_board_and_pieces() 
            
    def update_ai_depth(self, val_str_or_int):
        snapped_value = round(float(val_str_or_int))
        self.ai_depth = int(snapped_value)
        if hasattr(self, 'depth_slider') and abs(self.depth_slider.get() - snapped_value) > 0.01 : 
            self.depth_slider.set(snapped_value) 
        if hasattr(self, 'depth_value_label'): 
            self.depth_value_label.config(text=str(self.ai_depth))

    def on_square_interaction_start(self, event): 
        if self.game_over_flag or self.board.turn != chess.WHITE or self.ai_thinking: return
        clicked_square = self._get_canvas_xy_to_square(event.x, event.y)
        if clicked_square is None: return

        if self.drag_selected_square is not None and self.drag_selected_square != clicked_square:
            self._attempt_player_move(self.drag_selected_square, clicked_square)
            return 

        self._clear_highlights() 

        piece_on_click = self.board.piece_at(clicked_square)
        if piece_on_click and piece_on_click.color == self.board.turn:
            self.drag_selected_square = clicked_square
            self.dragging_piece_item_id = self.piece_item_ids.get(clicked_square) 
            if self.dragging_piece_item_id:
                outline_items = self.canvas.find_withtag(f"piece_outline_{clicked_square}")
                for item_id in outline_items: self.canvas.lift(item_id)
                self.canvas.lift(self.dragging_piece_item_id)    
                main_piece_coords = self.canvas.coords(self.dragging_piece_item_id)
                if main_piece_coords: 
                    cx, cy = main_piece_coords[0], main_piece_coords[1]
                    self.drag_start_x_offset = event.x - cx; self.drag_start_y_offset = event.y - cy
                    new_x_fg = event.x - self.drag_start_x_offset; new_y_fg = event.y - self.drag_start_y_offset
                    self.canvas.coords(self.dragging_piece_item_id, new_x_fg, new_y_fg)
                    for item_id in outline_items: self.canvas.coords(item_id, new_x_fg, new_y_fg) 
            self._show_legal_moves_for_selected_piece()
        else: 
            self.drag_selected_square = None; self.dragging_piece_item_id = None

    def on_piece_drag_motion(self, event): 
        if self.dragging_piece_item_id and self.drag_selected_square is not None and not self.ai_thinking:
            new_x_fg = event.x - self.drag_start_x_offset; new_y_fg = event.y - self.drag_start_y_offset
            self.canvas.coords(self.dragging_piece_item_id, new_x_fg, new_y_fg)
            outline_items = self.canvas.find_withtag(f"piece_outline_{self.drag_selected_square}")
            for item_id in outline_items: self.canvas.coords(item_id, new_x_fg, new_y_fg) 

    def on_square_interaction_end(self, event): 
        if self.drag_selected_square is None or self.dragging_piece_item_id is None or self.ai_thinking: return
        to_square = self._get_canvas_xy_to_square(event.x, event.y)
        self.redraw_board_and_pieces() 
        self._clear_highlights() 
        if to_square is not None and self.drag_selected_square != to_square: 
            self._attempt_player_move(self.drag_selected_square, to_square)
        else: 
            self.drag_selected_square = None; self.dragging_piece_item_id = None

    def _attempt_player_move(self, from_sq: chess.Square, to_sq: chess.Square): 
        promotion = None
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and ((piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0)): promotion = chess.QUEEN
        move = chess.Move(from_sq, to_sq, promotion=promotion)
        
        self.drag_selected_square = None; self.dragging_piece_item_id = None
        self._clear_highlights() 

        if move in self.board.legal_moves:
            self.check_history.append((self.white_checks_delivered, self.black_checks_delivered)); self.board.push(move)
            if self.board.is_check() and self.board.turn == chess.BLACK: self.white_checks_delivered += 1
            self.redraw_board_and_pieces() 
            if not self.game_over_flag: 
                self.update_status_label(for_ai_move_eval=None); 
                self.root.update_idletasks(); 
                self._start_ai_move_thread()
        else:
            self.redraw_board_and_pieces() 
        self._clear_highlights()


    def _start_ai_move_thread(self):
        if self.ai_thinking: return 
        self.ai_thinking = True
        self.status_label.config(text="AI is thinking...") 
        self.depth_slider.config(state=tk.DISABLED)
        
        ai_thread = threading.Thread(target=self._ai_move_worker, daemon=True)
        ai_thread.start()

    def _ai_move_worker(self):
        start_time = time.monotonic()
        ai_move = self.ai.find_best_move(self.board, self.ai_depth, self.white_checks_delivered, self.black_checks_delivered, self.MAX_CHECKS)
        eval_after_search = self.ai.current_eval_for_ui # Get the eval from AI's perspective of the chosen path
        elapsed=time.monotonic()-start_time; 
        total_nodes = self.ai.nodes_evaluated + self.ai.q_nodes_evaluated
        nps = total_nodes/elapsed if elapsed>0 else 0
        
        tt_filled_percentage = (len(self.ai.transposition_table) / self.ai.tt_size_limit) * 100 if self.ai.tt_size_limit > 0 else 0
        
        # Don't include ai_move.uci() in the print_info for the console
        print_info = (f"AI Search Stats: T: {elapsed:.2f}s | "
                      f"Nodes: {total_nodes} (NPS:{nps:.0f}) | TT: {tt_filled_percentage:.2f}%")
        self.ai_move_queue.put((ai_move, print_info, eval_after_search))


    def _check_ai_queue_periodically(self):
        try:
            ai_move, print_info, eval_score = self.ai_move_queue.get_nowait()
            self._process_ai_move_from_queue(ai_move, print_info, eval_score)
        except queue.Empty:
            pass 
        finally:
            self.root.after(100, self._check_ai_queue_periodically) 

    def _process_ai_move_from_queue(self, ai_move: Optional[chess.Move], print_info: str, eval_score_from_ai_search: int):
        self.ai_thinking = False
        self.depth_slider.config(state=tk.NORMAL) 

        print(print_info) 

        if ai_move:
            self.check_history.append((self.white_checks_delivered, self.black_checks_delivered)); self.board.push(ai_move)
            if self.board.is_check() and self.board.turn == chess.WHITE: self.black_checks_delivered += 1
            self.redraw_board_and_pieces() 
            # Use the eval_score_from_ai_search which is the value of the path the AI chose
            self.update_status_label(for_ai_move_eval=eval_score_from_ai_search) 
        else: 
            print("AI Warning: No legal moves from AI thread (or game ended)."); 
            self.update_status_label() # Update status if AI found no move
        
        # If game not over by AI move, status label should reflect player's turn without a specific path eval from AI
        if not self.game_over_flag:
            self.update_status_label(for_ai_move_eval=None) # Clears path eval, shows static


if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()