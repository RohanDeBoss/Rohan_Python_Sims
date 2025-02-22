import tkinter as tk
import chess
import time
import random
from typing import Tuple, List, Optional

class ChessUI:
    def __init__(self, square_size=60):
        self.square_size = square_size
        self.root = tk.Tk()
        self.root.title("Python Chess")

        # Initialize board and state variables
        self.board = chess.Board()
        self.ai_depth = 3
        self.flipped = False
        self.selected_square = None
        self.dragging_item = None
        self.highlights = []
        self.piece_images = {}
        self.transposition_table = {}  # Transposition table cache

        # Chess.com-like colors
        self.colors = {
            'background': '#333333',
            'square_light': '#F0D9B5',
            'square_dark': '#B58863',
            'highlight': '#FFD700',
            'move_indicator': '#7FFF00'
        }

        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create canvas
        self.canvas = tk.Canvas(
            main_frame,
            width=8 * self.square_size,
            height=8 * self.square_size,
            bg=self.colors['background']
        )
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
        new_game_button = tk.Button(control_panel, text="New Game", command=self.reset_game)
        new_game_button.pack(pady=5, fill='x')

        flip_board_button = tk.Button(control_panel, text="Flip Board", command=self.flip_board)
        flip_board_button.pack(pady=5, fill='x')

        undo_button = tk.Button(control_panel, text="Undo Move", command=self.undo_move)
        undo_button.pack(pady=5, fill='x')

        exit_button = tk.Button(control_panel, text="Exit", command=self.root.quit)
        exit_button.pack(pady=5, fill='x')

        # Depth slider (increased maximum depth)
        depth_slider_label = tk.Label(control_panel, text="Bot Depth", font=("Arial", 10))
        depth_slider_label.pack(pady=5)
        self.depth_slider = tk.Scale(control_panel, from_=1, to=7, orient=tk.HORIZONTAL,
                                   command=self.update_depth)
        self.depth_slider.set(self.ai_depth)
        self.depth_slider.pack(pady=5, fill="x")

        # Status label
        self.status_label = tk.Label(
            control_panel,
            text="White's turn",
            font=('Arial', 12, 'bold'),
            fg='#333333',
            width=20,
            anchor="center"
        )
        self.status_label.pack(pady=20)

        # Initialize zobrist hashing tables
        self.init_zobrist_tables()
        
        # Initialize history table for killer moves
        self.killer_moves = [[None, None] for _ in range(64)]  # Store 2 killer moves per ply
        
        # Initialize history heuristic table
        self.history_table = {}  # (piece_type, to_square) -> score
        
        # Transposition table with fixed size (2^20 entries)
        self.tt_size = 2**20
        self.transposition_table = {}
        
        # Counter for positions evaluated
        self.nodes_evaluated = 0
        
        # Previous best move for move ordering
        self.prev_best_move = None

        self.load_piece_images()
        self.update_display()

    def load_piece_images(self):
        """Load Unicode chess symbols to represent pieces (not used for images here)."""
        piece_symbols = {
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟'
        }
        for piece, symbol in piece_symbols.items():
            self.piece_images[piece] = symbol

    def on_canvas_resize(self, event):
        """Handle canvas resize to update square size and redraw the board."""
        # Choose the new square size so that 8 squares fit in the smallest dimension.
        new_square_size = int(min(event.width, event.height) / 8)
        if new_square_size != self.square_size:
            self.square_size = new_square_size
            self.update_display()

    def square_to_canvas(self, file, rank):
        """
        Convert board (file, rank) to canvas coordinates.
        Files and ranks are 0-indexed; if board is flipped, adjust accordingly.
        """
        if self.flipped:
            canvas_file = 7 - file
            canvas_rank = rank
        else:
            canvas_file = file
            canvas_rank = 7 - rank
        x1 = canvas_file * self.square_size
        y1 = canvas_rank * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        return x1, y1, x2, y2

    def canvas_to_square(self, x, y):
        """Convert canvas (x, y) coordinates to a chess square."""
        file_index = x // self.square_size
        rank_index = y // self.square_size
        if self.flipped:
            file = 7 - file_index
            rank = rank_index
        else:
            file = file_index
            rank = 7 - rank_index
        return chess.square(file, rank)

    def draw_board(self):
        """Draw the board squares using chess.com colors."""
        self.canvas.delete("square")
        for rank in range(8):
            for file in range(8):
                x1, y1, x2, y2 = self.square_to_canvas(file, rank)
                # Use light square if file+rank is even; dark otherwise.
                color = self.colors['square_light'] if (file + rank) % 2 == 0 else self.colors['square_dark']
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color, tags="square")

    def draw_piece_with_outline(self, cx, cy, text, tag, is_white):
        """
        Draw a piece at center (cx,cy) with a simulated outline.
        For white pieces, use white fill with black outline.
        For black pieces, use black fill with white outline.
        The font size scales with the square size.
        """
        font_size = int(self.square_size * 0.8)
        font = ('Segoe UI Symbol', font_size, 'bold')
        # Determine colors.
        if is_white:
            fill_color = "#FFFFFF"
            outline_color = "#000000"
        else:
            fill_color = "#000000"
            outline_color = "#FFFFFF"
        # Offsets to simulate an outline.
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in offsets:
            self.canvas.create_text(cx + dx, cy + dy,
                                    text=text, font=font,
                                    fill=outline_color, tags=("piece", tag))
        # Draw the main piece text.
        self.canvas.create_text(cx, cy, text=text, font=font,
                                fill=fill_color, tags=("piece", tag))

    def draw_pieces(self):
        """Draw all pieces on the board using outlined text for clarity."""
        self.canvas.delete("piece")
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x1, y1, x2, y2 = self.square_to_canvas(file, rank)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # Use a unique tag for each piece.
                tag = f"piece_{square}"
                is_white = piece.color == chess.WHITE
                self.draw_piece_with_outline(cx, cy, self.piece_images[piece.symbol()], tag, is_white)

    def show_legal_moves(self, square):
        """Highlight legal moves for the selected piece."""
        self.clear_highlights()
        legal_moves = [move for move in self.board.legal_moves if move.from_square == square]
        for move in legal_moves:
            file = chess.square_file(move.to_square)
            rank = chess.square_rank(move.to_square)
            x1, y1, x2, y2 = self.square_to_canvas(file, rank)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.highlights.append(
                self.canvas.create_oval(
                    cx - self.square_size * 0.25, cy - self.square_size * 0.25,
                    cx + self.square_size * 0.25, cy + self.square_size * 0.25,
                    fill=self.colors['move_indicator'], outline="",
                    tags="highlight"
                )
            )

    def clear_highlights(self):
        """Clear any move highlights."""
        for h in self.highlights:
            self.canvas.delete(h)
        self.highlights = []

    def update_status(self):
        """Update the status label based on the game state, including the evaluation."""
        if self.board.is_checkmate():
            status = "Checkmate! " + ("Black wins!" if self.board.turn == chess.WHITE else "White wins!")
        elif self.board.is_stalemate():
            status = "Stalemate!"
        elif self.board.is_check():
            status = "Check!"
        else:
            eval_score = self.evaluate_position()
            turn_str = "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
            status = f"{turn_str} (Eval: {eval_score})"
        self.status_label.config(text=status)



    def update_display(self):
        """Redraw the board, pieces, and update the status."""
        self.draw_board()
        self.draw_pieces()
        self.update_status()

    def reset_game(self):
        """Reset the game to the starting position."""
        self.board = chess.Board()
        self.selected_square = None
        self.clear_highlights()
        self.update_display()

    def flip_board(self):
        """Flip the board orientation."""
        self.flipped = not self.flipped
        self.update_display()

    def undo_move(self):
        """
        Undo the last move. If possible, undo a full round (both Black and White moves)
        so that you can re-enter a move.
        """
        if self.board.move_stack:
            self.board.pop()
            # If it is now White's turn and there's another move to undo, pop again.
            if self.board.turn == chess.WHITE and self.board.move_stack:
                self.board.pop()
        self.update_display()

    def update_depth(self, val):
        """Update the AI depth based on the slider value."""
        self.ai_depth = int(val)

    def start_drag(self, event):
        """Begin dragging a White piece (only allowed when it's White's turn)."""
        if self.board.is_game_over() or self.board.turn != chess.WHITE:
            return
        square = self.canvas_to_square(event.x, event.y)
        piece = self.board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            self.selected_square = square
            self.show_legal_moves(square)
            # Create a temporary dragging item (simulate outlined piece).
            self.dragging_item = self.canvas.create_text(
                event.x, event.y,
                text=self.piece_images[piece.symbol()],
                font=('Segoe UI Symbol', int(self.square_size * 0.8), 'bold'),
                fill="#000000", tags="dragging"
            )
            # Remove only the dragged piece from the board.
            self.canvas.delete(f"piece_{square}")

    def drag_piece(self, event):
        """Update the location of the dragged piece."""
        if self.dragging_item is not None:
            self.canvas.coords(self.dragging_item, event.x, event.y)

    def drop_piece(self, event):
        """Drop the dragged piece, making a move if legal."""
        if self.selected_square is None or self.dragging_item is None:
            return
        from_square = self.selected_square
        to_square = self.canvas_to_square(event.x, event.y)
        move = chess.Move(from_square, to_square)
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

    def evaluate_mobility(self):
        """Evaluate piece mobility (number of legal moves) for both sides."""
        mobility_score = 0
        
        # Store current turn
        original_turn = self.board.turn
        
        # Evaluate Black's mobility
        self.board.turn = chess.BLACK
        black_mobility = self.board.legal_moves.count()
        
        # Evaluate White's mobility
        self.board.turn = chess.WHITE
        white_mobility = self.board.legal_moves.count()

        # Restore original turn
        self.board.turn = original_turn
        
        # Return mobility difference (positive favors Black)
        return (black_mobility - white_mobility) * 10

    def evaluate_center_control(self):
        """Evaluate control of the center squares."""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        score = 0
        
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece:
                score += 30 if piece.color == chess.BLACK else -30

        attacks = self.board.attackers_mask(chess.BLACK, square)  # Bitboard attack lookup
        score += 10 if attacks else 0

        attacks = self.board.attackers_mask(chess.WHITE, square)
        score -= 10 if attacks else 0
        return score

    def evaluate_king_safety(self):
        """Evaluate king safety based on surrounding squares and pawn shield."""
        score = 0
        
        for color in [chess.BLACK, chess.WHITE]:
            king_square = self.board.king(color)
            if king_square is not None:
                attacked_squares = self.board.attackers_mask(not color, king_square)
            score += (-30 if color == chess.BLACK else 30) * bin(attacked_squares).count('1')

                
            # Check surrounding squares
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    file = chess.square_file(king_square) + dx
                    rank = chess.square_rank(king_square) + dy
                    
                    if 0 <= file <= 7 and 0 <= rank <= 7:
                        adjacent_square = chess.square(file, rank)
                        if self.board.is_attacked_by(not color, adjacent_square):
                            score += -30 if color == chess.BLACK else 30
        
        return score
    
    def init_zobrist_tables(self):
        """Initialize Zobrist hashing tables for more efficient position keys"""
        random.seed(42)  # For reproducible hashing
        self.zobrist_piece_square = [
            [[random.getrandbits(64) for _ in range(12)]  # 6 piece types * 2 colors
             for _ in range(64)]  # squares
        ]
        self.zobrist_castling = [random.getrandbits(64) for _ in range(16)]
        self.zobrist_ep = [random.getrandbits(64) for _ in range(8)]  # En passant files
        self.zobrist_side = random.getrandbits(64)

    def compute_hash(self) -> int:
        """Compute Zobrist hash for the current position"""
        h = 0
        
        # Hash pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_idx = piece.piece_type - 1 + (6 if piece.color else 0)
                h ^= self.zobrist_piece_square[0][square][piece_idx]
        
        # Hash castling rights
        castling = (self.board.castling_rights & 0xF)
        if castling:
            h ^= self.zobrist_castling[castling]
            
        # Hash en passant
        if self.board.ep_square:
            file = chess.square_file(self.board.ep_square)
            h ^= self.zobrist_ep[file]
            
        # Hash side to move
        if self.board.turn == chess.BLACK:
            h ^= self.zobrist_side
            
        return h

    def get_mvv_lva_score(self, move: chess.Move) -> int:
        """Get MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        
        if not self.board.is_capture(move):
            return 0
            
        attacker = self.board.piece_at(move.from_square)
        victim = self.board.piece_at(move.to_square)
        
        if not (attacker and victim):
            return 0
            
        # MVV-LVA score = 6 * victim_value - attacker_value
        return 6 * piece_values[victim.piece_type] - piece_values[attacker.piece_type]

    def is_good_capture(self, move: chess.Move) -> bool:
        """Static Exchange Evaluation (SEE) to determine if a capture is good"""
        if not self.board.is_capture(move):
            return False
            
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Get initial exchange value
        victim = self.board.piece_at(move.to_square)
        attacker = self.board.piece_at(move.from_square)
        
        if not (victim and attacker):
            return False
            
        gain = piece_values[victim.piece_type] - piece_values[attacker.piece_type]
        
        # If we're winning material immediately, it's good
        if gain >= 0:
            return True
            
        # Simple check for obvious bad captures
        if piece_values[attacker.piece_type] - piece_values[victim.piece_type] > 200:
            return False
            
        return True

    def get_move_score(self, move: chess.Move, tt_move: Optional[chess.Move], ply: int) -> int:
        """Score a move for move ordering"""
        score = 0
        
        # 1. TT move gets highest priority
        if tt_move and move == tt_move:
            return 20000
            
        # 2. Previous iteration's best move
        if self.prev_best_move and move == self.prev_best_move:
            return 19000
            
        # 3. Captures with SEE
        if self.board.is_capture(move):
            if self.is_good_capture(move):
                score = 18000 + self.get_mvv_lva_score(move)
            else:
                score = 15000 + self.get_mvv_lva_score(move)
                
        # 4. Killer moves
        if self.killer_moves[ply][0] == move:
            score = 17000
        elif self.killer_moves[ply][1] == move:
            score = 16000
            
        # 5. History heuristic
        piece = self.board.piece_at(move.from_square)
        if piece:
            history_key = (piece.piece_type, move.to_square)
            score += self.history_table.get(history_key, 0)
            
        # 6. Positional bonuses
        if move.to_square in {chess.E4, chess.E5, chess.D4, chess.D5}:  # Center control
            score += 100
            
        # 7. Promotions
        if move.promotion:
            score += 16500 + (500 if move.promotion == chess.QUEEN else 0)
            
        return score

    def order_moves(self, moves: List[chess.Move], tt_move: Optional[chess.Move], ply: int) -> List[chess.Move]:
        """Advanced move ordering with multiple heuristics"""
        scored_moves = [(move, self.get_move_score(move, tt_move, ply)) for move in moves]
        return [move for move, _ in sorted(scored_moves, key=lambda x: -x[1])]

    def evaluate_position(self):
        """Enhanced evaluation function (from White's perspective: +eval means White is ahead, -eval means Black is ahead)"""
        # If checkmate: if it's White's turn, that means White is checkmated, so a very low (negative) score; 
        # if it's Black's turn, Black is checkmated, so a very high (positive) score.
        if self.board.is_checkmate():
            return -99999 if self.board.turn == chess.WHITE else 99999
        if self.board.is_stalemate():
            return 0

        self.nodes_evaluated += 1

        # Material evaluation: add value for White's pieces, subtract for Black's.
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }

        material = 0
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                # White pieces add positively; Black pieces subtract.
                material += piece_values[piece.piece_type] if piece.color == chess.WHITE else -piece_values[piece.piece_type]

        # Positional bonus: for example, control of the center.
        positional = 0
        center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        for square in center_squares:
            piece = self.board.piece_at(square)
            if piece:
                positional += 20 if piece.color == chess.WHITE else -20

        return material + positional

    def store_in_tt(self, key: int, depth: int, value: int, flag: str, best_move: Optional[chess.Move]):
        """Store position in transposition table with replacement strategy"""
        if len(self.transposition_table) >= self.tt_size:
            self.transposition_table.pop(next(iter(self.transposition_table)))  # Remove oldest entry

            
        self.transposition_table[key] = {
            'depth': depth,
            'value': value,
            'flag': flag,
            'best_move': best_move
        }

    def minimax(self, depth: int, alpha: int, beta: int, maximizing: bool, ply: int) -> Tuple[int, Optional[chess.Move]]:
        """Enhanced minimax with all optimizations"""
        # 1. Check transposition table
        tt_key = self.compute_hash()
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

        # 2. Base case
        if depth == 0 or self.board.is_game_over():
            value = self.evaluate_position()
            self.store_in_tt(tt_key, depth, value, 'exact', None)
            return value, None

        # 3. Move generation and ordering
        best_move = None
        if maximizing:
            value = -float('inf')
            for move in self.order_moves(self.board.legal_moves, tt_move, ply):
                self.board.push(move)
                new_value, _ = self.minimax(depth - 1, alpha, beta, False, ply + 1)
                self.board.pop()
                
                if new_value > value:
                    value = new_value
                    best_move = move
                    
                alpha = max(alpha, value)
                if alpha >= beta:
                    # Store killer move
                    if not self.board.is_capture(move) and move != self.killer_moves[ply][0]:
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
            for move in self.order_moves(self.board.legal_moves, tt_move, ply):
                self.board.push(move)
                new_value, _ = self.minimax(depth - 1, alpha, beta, True, ply + 1)
                self.board.pop()
                
                if new_value < value:
                    value = new_value
                    best_move = move
                    
                beta = min(beta, value)
                if alpha >= beta:
                    # Store killer move
                    if not self.board.is_capture(move) and move != self.killer_moves[ply][0]:
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    break
                    
            flag = 'exact'
            if value <= alpha:
                flag = 'upper'
            elif value >= beta:
                flag = 'lower'

        # Update history heuristic for quiet moves
        if best_move and not self.board.is_capture(best_move):
            piece = self.board.piece_at(best_move.from_square)
            if piece:
                history_key = (piece.piece_type, best_move.to_square)
                self.history_table[history_key] = self.history_table.get(history_key, 0) + depth * depth

        # Store position in transposition table
        self.store_in_tt(tt_key, depth, value, flag, best_move)
        return value, best_move

    def find_best_move(self):
        """Find best move with iterative deepening"""
        self.nodes_evaluated = 0
        start_time = time.time()
        best_move = None
        
        # Iterative deepening
        for depth in range(1, self.ai_depth + 1):
            value, move = self.minimax(depth, -float('inf'), float('inf'), True, 0)
            if move:
                best_move = move
                self.prev_best_move = move
                
            # Simple time management
            if time.time() - start_time > 5:  # 5 second limit
                break
                
        return best_move or next(iter(self.board.legal_moves))

    def ai_move(self):
        """Make AI move with enhanced logging"""
        if self.board.is_game_over() or self.board.turn != chess.BLACK:
            return
            
        self.status_label.config(text="Black is thinking...")
        self.root.update_idletasks()
        
        start_time = time.time()
        best_move = self.find_best_move()
        self.board.push(best_move)
        self.update_display()
        
        elapsed = time.time() - start_time
        print(f"AI move: {best_move}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Nodes evaluated: {self.nodes_evaluated}")
        print(f"Nodes per second: {self.nodes_evaluated/elapsed:.0f}")
        print(f"TT size: {len(self.transposition_table)}")

if __name__ == "__main__":
    gui = ChessUI()
    gui.root.mainloop()