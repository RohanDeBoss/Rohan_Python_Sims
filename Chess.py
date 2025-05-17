import tkinter as tk
import chess
import time
import random
from typing import Tuple, List, Optional

# Piece-Square Tables (PSTs) for each piece type (White's perspective)
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

class ChessAI:
    def __init__(self):
        """Initialize AI-specific attributes."""
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(64)]  # Two killer moves per ply
        self.history_table = {}  # (piece_type, to_square) -> score
        self.tt_size = 2**20  # Transposition table size
        self.nodes_evaluated = 0  # Counter for evaluated nodes
        self.prev_best_move = None  # Previous best move for ordering
        self.init_zobrist_tables()

    def init_zobrist_tables(self):
        """Initialize Zobrist hashing tables for position keys."""
        random.seed(42)  # Reproducible hashing
        self.zobrist_piece_square = [
            [[random.getrandbits(64) for _ in range(12)] for _ in range(64)]  # 6 piece types * 2 colors, 64 squares
        ]
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        self.zobrist_ep = [random.getrandbits(64) for _ in range(8)]  # En passant files
        self.zobrist_side = random.getrandbits(64)

    def compute_hash(self, board):
        """Compute Zobrist hash for the given board position."""
        h = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)
                h ^= self.zobrist_piece_square[0][square][piece_idx]
        castling = (board.castling_rights & 0xF)
        if castling:
            h ^= self.zobrist_castling[castling]
        if board.ep_square:
            file = chess.square_file(board.ep_square)
            h ^= self.zobrist_ep[file]
        if board.turn == chess.BLACK:
            h ^= self.zobrist_side
        return h

    def get_mvv_lva_score(self, board, move):
        """Calculate MVV-LVA score for move ordering."""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
        if not board.is_capture(move):
            return 0
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        if not (attacker and victim):
            return 0
        return 6 * piece_values[victim.piece_type] - piece_values[attacker.piece_type]

    def is_good_capture(self, board, move):
        """Determine if a capture is favorable using SEE-like logic."""
        if not board.is_capture(move):
            return False
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if not (victim and attacker):
            return False
        gain = piece_values[victim.piece_type] - piece_values[attacker.piece_type]
        if gain >= 0:
            return True
        if piece_values[attacker.piece_type] - piece_values[victim.piece_type] > 200:
            return False
        return True

    def get_move_score(self, board, move, tt_move, ply):
        """Score a move for ordering based on multiple heuristics."""
        score = 0
        if tt_move and move == tt_move:
            return 20000
        if self.prev_best_move and move == self.prev_best_move:
            return 19000
        if board.is_capture(move):
            if self.is_good_capture(board, move):
                score = 18000 + self.get_mvv_lva_score(board, move)
            else:
                score = 15000 + self.get_mvv_lva_score(board, move)
        if self.killer_moves[ply][0] == move:
            score = 17000
        elif self.killer_moves[ply][1] == move:
            score = 16000
        piece = board.piece_at(move.from_square)
        if piece:
            history_key = (piece.piece_type, move.to_square)
            score += self.history_table.get(history_key, 0)
        if move.to_square in {chess.E4, chess.E5, chess.D4, chess.D5}:
            score += 100
        if move.promotion:
            score += 16500 + (500 if move.promotion == chess.QUEEN else 0)
        return score

    def order_moves(self, board, moves, tt_move, ply):
        """Order moves using advanced heuristics."""
        scored_moves = [(move, self.get_move_score(board, move, tt_move, ply)) for move in moves]
        return [move for move, _ in sorted(scored_moves, key=lambda x: -x[1])]

    def evaluate_position(self, board):
        """Evaluate the board position from White's perspective, including PSTs."""
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate():
            return 0
        self.nodes_evaluated += 1
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        material = 0
        positional = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Material value
                value = piece_values[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                material += value
                
                # PST value
                if piece.piece_type in PST:
                    if piece.color == chess.WHITE:
                        # White: use square directly
                        pst_value = PST[piece.piece_type][square]
                    else:
                        # Black: mirror the square and negate the value
                        mirrored_square = chess.square_mirror(square)
                        pst_value = -PST[piece.piece_type][mirrored_square]
                    positional += pst_value
        
        return material + positional

    def store_in_tt(self, key, depth, value, flag, best_move):
        """Store position in transposition table with replacement strategy."""
        if len(self.transposition_table) >= self.tt_size:
            self.transposition_table.pop(next(iter(self.transposition_table)))
        self.transposition_table[key] = {
            'depth': depth, 'value': value, 'flag': flag, 'best_move': best_move
        }

    def minimax(self, board, depth, alpha, beta, maximizing, ply):
        """Minimax with alpha-beta pruning and optimizations."""
        tt_key = self.compute_hash(board)
        tt_entry = self.transposition_table.get(tt_key)
        tt_move = tt_entry['best_move'] if tt_entry else None
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'exact':
                return tt_entry['value'], tt_entry['best_move']
            elif tt_entry['flag'] == 'lower' and tt_entry['value'] > alpha:
                alpha = tt_entry['value']
            elif tt_entry['flag'] == 'upper' and tt_entry['value'] < beta:
                beta = tt_entry['value']
            if alpha >= beta:
                return tt_entry['value'], tt_entry['best_move']
        if depth == 0 or board.is_game_over():
            value = self.evaluate_position(board)
            self.store_in_tt(tt_key, depth, value, 'exact', None)
            return value, None
        best_move = None
        if maximizing:
            value = -float('inf')
            for move in self.order_moves(board, board.legal_moves, tt_move, ply):
                board.push(move)
                new_value, _ = self.minimax(board, depth - 1, alpha, beta, False, ply + 1)
                board.pop()
                if new_value > value:
                    value = new_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    if not board.is_capture(move) and move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    break
            flag = 'exact'
            if value <= alpha:
                flag = 'upper'
            elif value >= beta:
                flag = 'lower'
        else:
            value = float('inf')
            for move in self.order_moves(board, board.legal_moves, tt_move, ply):
                board.push(move)
                new_value, _ = self.minimax(board, depth - 1, alpha, beta, True, ply + 1)
                board.pop()
                if new_value < value:
                    value = new_value
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    if not board.is_capture(move) and move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    break
            flag = 'exact'
            if value <= alpha:
                flag = 'upper'
            elif value >= beta:
                flag = 'lower'
        if best_move and not board.is_capture(best_move):
            piece = board.piece_at(best_move.from_square)
            if piece:
                history_key = (piece.piece_type, best_move.to_square)
                self.history_table[history_key] = self.history_table.get(history_key, 0) + depth * depth
        self.store_in_tt(tt_key, depth, value, flag, best_move)
        return value, best_move

    def find_best_move(self, board, depth):
        self.nodes_evaluated = 0
        start_time = time.time()
        best_move = None
        for d in range(1, depth + 1):
            value, move = self.minimax(board, d, -float('inf'), float('inf'), False, 0)
            if move:
                best_move = move
                self.prev_best_move = move
            if time.time() - start_time > 5:
                break
        return best_move or next(iter(board.legal_moves))

class ChessUI:
    def __init__(self, square_size=60):
        """Initialize the chess UI."""
        self.square_size = square_size
        self.root = tk.Tk()
        self.root.title("Python Chess")
        self.board = chess.Board()
        self.ai_depth = 4  # Default depth for AI
        self.flipped = False
        self.selected_square = None  # Initialized as None to avoid 0 ambiguity
        self.dragging_item = None
        self.highlights = []
        self.piece_images = {}
        self.ai = ChessAI()  # Create AI instance

        # Chess.com-like colors
        self.colors = {
            'background': '#333333', 'square_light': '#F0D9B5', 'square_dark': '#B58863',
            'highlight': '#FFD700', 'move_indicator': '#7FFF00'
        }

        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Canvas
        self.canvas = tk.Canvas(main_frame, width=8 * self.square_size, height=8 * self.square_size, bg=self.colors['background'])
        self.canvas.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Bind events
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag_piece)
        self.canvas.bind("<ButtonRelease-1>", self.drop_piece)

        # Control panel
        control_panel = tk.Frame(main_frame)
        control_panel.grid(row=0, column=1, padx=10, sticky="n")

        # Buttons
        tk.Button(control_panel, text="New Game", command=self.reset_game).pack(pady=5, fill='x')
        tk.Button(control_panel, text="Flip Board", command=self.flip_board).pack(pady=5, fill='x')
        tk.Button(control_panel, text="Undo Move", command=self.undo_move).pack(pady=5, fill='x')
        tk.Button(control_panel, text="Exit", command=self.root.quit).pack(pady=5, fill='x')

        # Depth slider
        tk.Label(control_panel, text="Bot Depth", font=("Arial", 10)).pack(pady=5)
        self.depth_slider = tk.Scale(control_panel, from_=1, to=7, orient=tk.HORIZONTAL, command=self.update_depth)
        self.depth_slider.set(self.ai_depth)
        self.depth_slider.pack(pady=5, fill="x")

        # Status label
        self.status_label = tk.Label(control_panel, text="White's turn", font=('Arial', 12, 'bold'), fg='#333333', width=20, anchor="center")
        self.status_label.pack(pady=20)

        self.load_piece_images()
        self.update_display()

    def load_piece_images(self):
        """Load Unicode chess symbols for pieces."""
        piece_symbols = {
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
        }
        self.piece_images.update(piece_symbols)

    def on_canvas_resize(self, event):
        """Adjust square size and redraw on resize."""
        new_square_size = int(min(event.width, event.height) / 8)
        if new_square_size != self.square_size:
            self.square_size = new_square_size
            self.update_display()

    def square_to_canvas(self, file, rank):
        """Convert board coordinates to canvas coordinates."""
        if self.flipped:
            canvas_file, canvas_rank = 7 - file, rank
        else:
            canvas_file, canvas_rank = file, 7 - rank
        x1, y1 = canvas_file * self.square_size, canvas_rank * self.square_size
        return x1, y1, x1 + self.square_size, y1 + self.square_size

    def canvas_to_square(self, x, y):
        """Convert canvas coordinates to chess square."""
        file_index, rank_index = x // self.square_size, y // self.square_size
        file = 7 - file_index if self.flipped else file_index
        rank = rank_index if self.flipped else 7 - rank_index
        return chess.square(file, rank)

    def draw_board(self):
        """Draw the chessboard squares."""
        self.canvas.delete("square")
        for rank in range(8):
            for file in range(8):
                x1, y1, x2, y2 = self.square_to_canvas(file, rank)
                color = self.colors['square_light'] if (file + rank) % 2 == 0 else self.colors['square_dark']
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color, tags="square")

    def draw_piece_with_outline(self, cx, cy, text, tag, is_white):
        """Draw a piece with an outline."""
        font_size = int(self.square_size * 0.7)
        font = ('Segoe UI Symbol', font_size, 'bold')
        fill_color, outline_color = ("#FFFFFF", "#000000") if is_white else ("#000000", "#FFFFFF")
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.canvas.create_text(cx + dx, cy + dy, text=text, font=font, fill=outline_color, tags=("piece", tag))
        self.canvas.create_text(cx, cy, text=text, font=font, fill=fill_color, tags=("piece", tag))

    def draw_pieces(self):
        """Draw all pieces on the board."""
        self.canvas.delete("piece")
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file, rank = chess.square_file(square), chess.square_rank(square)
                x1, y1, x2, y2 = self.square_to_canvas(file, rank)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                self.draw_piece_with_outline(cx, cy, self.piece_images[piece.symbol()], f"piece_{square}", piece.color == chess.WHITE)

    def show_legal_moves(self, square):
        """Highlight legal moves for the selected piece."""
        self.clear_highlights()
        for move in [m for m in self.board.legal_moves if m.from_square == square]:
            file, rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
            x1, y1, x2, y2 = self.square_to_canvas(file, rank)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.highlights.append(
                self.canvas.create_oval(
                    cx - self.square_size * 0.25, cy - self.square_size * 0.25,
                    cx + self.square_size * 0.25, cy + self.square_size * 0.25,
                    fill=self.colors['move_indicator'], outline="", tags="highlight"
                )
            )

    def clear_highlights(self):
        """Remove move highlights."""
        for h in self.highlights:
            self.canvas.delete(h)
        self.highlights.clear()

    def update_status(self):
        """Update the status label with game state and evaluation."""
        if self.board.is_checkmate():
            status = "Checkmate! " + ("Black wins!" if self.board.turn == chess.WHITE else "White wins!")
        elif self.board.is_stalemate():
            status = "Stalemate!"
        else:
            eval_score = self.ai.evaluate_position(self.board)
            turn_str = "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
            status = f"{turn_str} (Eval: {eval_score})"
        self.status_label.config(text=status)

    def update_display(self):
        """Redraw the board and update status."""
        self.draw_board()
        self.draw_pieces()
        self.update_status()

    def reset_game(self):
        """Reset to the starting position."""
        self.board = chess.Board()
        self.selected_square = None
        self.clear_highlights()
        self.update_display()

    def flip_board(self):
        """Flip the board orientation."""
        self.flipped = not self.flipped
        self.update_display()

    def undo_move(self):
        """Undo the last move(s) to allow re-entry."""
        if self.board.move_stack:
            self.board.pop()
            if self.board.turn == chess.WHITE and self.board.move_stack:
                self.board.pop()
        self.update_display()

    def update_depth(self, val):
        """Update AI depth from slider."""
        self.ai_depth = int(val)

    def start_drag(self, event):
        """Start dragging a White piece if it's White's turn."""
        if self.board.is_game_over() or self.board.turn != chess.WHITE:
            return
        square = self.canvas_to_square(event.x, event.y)
        piece = self.board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            self.selected_square = square  # Set to square number (can be 0)
            self.show_legal_moves(square)
            self.dragging_item = self.canvas.create_text(
                event.x, event.y, text=self.piece_images[piece.symbol()],
                font=('Segoe UI Symbol', int(self.square_size * 0.8), 'bold'), fill="#000000", tags="dragging"
            )
            self.canvas.delete(f"piece_{square}")

    def drag_piece(self, event):
        """Update dragged piece position."""
        if self.dragging_item:
            self.canvas.coords(self.dragging_item, event.x, event.y)

    def drop_piece(self, event):
        """Drop the piece and make a move if legal."""
        # Explicitly check if a square is selected, avoiding 0 being False
        if self.selected_square is None or self.dragging_item is None:
            return

        to_square = self.canvas_to_square(event.x, event.y)
        move = chess.Move(self.selected_square, to_square)

        if move in self.board.legal_moves:
            self.board.push(move)
            self.update_display()
            self.root.after(100, self.ai_move)
        else:
            self.update_display()

        self.selected_square = None
        self.canvas.delete(self.dragging_item)
        self.dragging_item = None
        self.clear_highlights()

    def ai_move(self):
        """Make an AI move for Black."""
        if self.board.is_game_over() or self.board.turn != chess.BLACK:
            return
        self.status_label.config(text="Black is thinking...")
        self.root.update_idletasks()
        start_time = time.time()
        best_move = self.ai.find_best_move(self.board, self.ai_depth)
        self.board.push(best_move)
        self.update_display()
        elapsed = time.time() - start_time
        print(f"AI move: {best_move}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Nodes evaluated: {self.ai.nodes_evaluated}")
        print(f"Nodes per second: {self.ai.nodes_evaluated / elapsed:.0f}")
        print(f"TT size: {len(self.ai.transposition_table)}")

if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()