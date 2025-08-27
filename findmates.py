import chess
import chess.gaviota
import itertools
from pathlib import Path
import sys
import math
import multiprocessing
import os
from tqdm import tqdm
from typing import List, Tuple, Optional
from functools import lru_cache

# --- CONFIGURATION ---
TABLEBASE_PATH = Path("C:/Program Files (x86)/Arena/TB/gtb.cp4")

# --- MAPPING and HELPERS ---
PIECE_MAP_UPPER = {'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}

# --- OPTIMIZATION: Pre-compute all square distances for instant lookups in the worker ---
SQUARE_DISTANCE_LOOKUP = [[chess.square_distance(s1, s2) for s2 in chess.SQUARES] for s1 in chess.SQUARES]

# Add new constants for optimization
PAWN_INVALID_RANKS = frozenset([0, 7])  # First and last ranks
SYMMETRY_TRANSFORMS = [
    lambda sq: sq,  # Identity
    lambda sq: chess.square_mirror(sq),  # Vertical mirror
    lambda sq: chess.square(7-chess.square_file(sq), chess.square_rank(sq))  # Horizontal mirror
]

def parse_piece_input(user_input: str) -> Optional[Tuple[List[str], List[str]]]:
    """Parses user input like 'w: k,q b: k,n' into piece lists."""
    try:
        user_input = user_input.lower().replace(" ", "")
        parts = user_input.split('b:')
        if len(parts) != 2: return None
        white_part, black_part = parts[0].replace('w:', '').strip(), parts[1].strip()
        white_pieces_str = [p for p in white_part.split(',') if p]
        black_pieces_str = [p for p in black_part.split(',') if p]
        if white_pieces_str.count('k') != 1 or black_pieces_str.count('k') != 1: return None
        return white_pieces_str, black_pieces_str
    except Exception:
        return None

def get_tablebase_filename(white_pieces_str: List[str], black_pieces_str: List[str]) -> str:
    """Generates the canonical Gaviota filename from piece lists."""
    white_types = sorted([PIECE_MAP_UPPER[p] for p in white_pieces_str], key=lambda x: PIECE_VALUES[x], reverse=True)
    black_types = sorted([PIECE_MAP_UPPER[p] for p in black_pieces_str], key=lambda x: PIECE_VALUES[x], reverse=True)
    w_str = "".join(chess.piece_symbol(pt).upper() for pt in white_types)
    b_str = "".join(chess.piece_symbol(pt).lower() for pt in black_types)
    return (w_str + b_str).lower()

def worker_find_mate_in_chunk(
    combinations_chunk: List[Tuple[int, ...]],
    tb_path_str: str,
    all_piece_types: List[Tuple[int, bool]]
) -> Tuple[int, List[str]]:
    """Ultra-optimized worker function."""
    max_dtm_local = -1
    best_fens_local = []
    board = chess.Board(None)
    seen_positions = set()

    # Pre-compute piece type masks for quick lookup
    is_pawn = {i: pt == chess.PAWN for i, (pt, _) in enumerate(all_piece_types)}
    is_king = {i: pt == chess.KING for i, (pt, _) in enumerate(all_piece_types)}
    piece_colors = [color for _, color in all_piece_types]

    with chess.gaviota.open_tablebase(tb_path_str) as tb:
        for squares_combo in combinations_chunk:
            # Quick rank check for pawns
            invalid_pawns = any(
                is_pawn[i] and chess.square_rank(sq) in PAWN_INVALID_RANKS
                for i, sq in enumerate(squares_combo)
            )
            if invalid_pawns:
                continue

            # Get canonical position (considering symmetry)
            canonical_pos = min(
                tuple(sorted(transform(sq) for sq in squares_combo))
                for transform in SYMMETRY_TRANSFORMS
            )
            if canonical_pos in seen_positions:
                continue
            seen_positions.add(canonical_pos)

            # Find king squares directly from the combo
            wk_sq = bk_sq = -1
            for i, sq in enumerate(squares_combo):
                if is_king[i]:
                    if piece_colors[i]:
                        wk_sq = sq
                    else:
                        bk_sq = sq

            if SQUARE_DISTANCE_LOOKUP[wk_sq][bk_sq] <= 1:
                continue

            # Process valid piece permutations
            piece_map = {}
            for perm in set(itertools.permutations(range(len(all_piece_types)))):
                # Create position
                for i, p in enumerate(perm):
                    piece_map[squares_combo[i]] = chess.Piece(all_piece_types[p][0], all_piece_types[p][1])
                
                board.set_piece_map(piece_map)
                board.turn = chess.WHITE

                if not board.is_valid():
                    continue

                dtm_value = tb.probe_dtm(board)
                if dtm_value > max_dtm_local:
                    max_dtm_local = dtm_value
                    best_fens_local = [board.fen()]
                elif dtm_value == max_dtm_local:
                    fen = board.fen()
                    if fen not in best_fens_local:
                        best_fens_local.append(fen)

    return max_dtm_local, best_fens_local

def unpack_and_run_worker(args):
    """Helper function to unpack arguments for the pool mapper."""
    return worker_find_mate_in_chunk(*args)

def find_longest_checkmate(tb_filepath: Path, white_piece_types: List[int], black_piece_types: List[int]):
    """Manages the parallel search."""
    num_pieces = len(white_piece_types) + len(black_piece_types)
    if num_pieces == 5:
        print("\n--- WARNING: 5-piece search can still take a very long time. ---")
        if input("Are you sure you want to continue? (y/n): ").lower() != 'y': return

    all_piece_types = [(pt, chess.WHITE) for pt in white_piece_types] + \
                      [(pt, chess.BLACK) for pt in black_piece_types]

    print("\nGenerating all square combinations...")
    all_combinations = list(itertools.combinations(chess.SQUARES, num_pieces))
    total_combinations = len(all_combinations)
    num_cores = os.cpu_count() or 1
    
    CHUNK_MULTIPLIER = 64 # More chunks for smoother bar with faster processing
    num_chunks = num_cores * CHUNK_MULTIPLIER
    chunk_size = (total_combinations + num_chunks - 1) // num_chunks
    
    print(f"Initializing search across {num_cores} CPU cores, split into {num_chunks} chunks.")
    chunks = [all_combinations[i:i + chunk_size] for i in range(0, total_combinations, chunk_size)]
    worker_args = [(chunk, str(tb_filepath.parent), all_piece_types) for chunk in chunks]
    
    print(f"Analyzing {num_pieces}-piece endgame ({total_combinations:,} combinations)...")
    
    results = []
    current_max_dtm = -1
    with multiprocessing.Pool(processes=num_cores) as pool:
        progress_iterator = pool.imap_unordered(unpack_and_run_worker, worker_args)
        
        with tqdm(total=len(chunks), desc="Processing Chunks") as progress_bar:
            for result in progress_iterator:
                results.append(result)
                max_dtm_result, _ = result
                
                if max_dtm_result > current_max_dtm:
                    current_max_dtm = max_dtm_result
                    mate_in_moves = (current_max_dtm + 1) // 2
                    progress_bar.set_description(f"Best so far: Mate in {mate_in_moves} (DTM={current_max_dtm})")
                progress_bar.update(1)
    
    print("\nCombining final results...")
    final_max_dtm = -1
    final_best_fens = []
    
    for max_dtm_result, best_fens_result in results:
        if max_dtm_result > final_max_dtm:
            final_max_dtm = max_dtm_result
            final_best_fens = best_fens_result
        elif max_dtm_result == final_max_dtm:
            for fen in best_fens_result:
                if fen not in final_best_fens:
                    final_best_fens.append(fen)

    if final_max_dtm > 0:
        mate_in_moves = (final_max_dtm + 1) // 2
        print("\n--- Search Complete ---")
        print(f"Found longest forced checkmate for White: Mate in {mate_in_moves} moves (DTM={final_max_dtm})")
        print(f"Found {len(final_best_fens)} position(s) with this result:")
        for fen in sorted(final_best_fens):
            print(f"  - FEN: {fen}")
    else:
        print("\n--- Search Complete ---")
        print("No forced checkmate for White was found in any legal position for this material.")
        print("This ending is a draw or a win for Black.")

def main():
    """Main function to run the user interface."""
    print("--- Longest Forced Checkmate Finder (Optimized High-Speed Version) ---")
    print(f"Using Gaviota tablebases from: '{TABLEBASE_PATH}'")
    print("Enter pieces for each side, e.g., 'w: k,r b: k'")
    print("Type 'exit' or 'quit' to close.")

    while True:
        user_input = input("\nEnter pieces > ")
        if user_input.lower() in ['exit', 'quit']:
            break

        parsed = parse_piece_input(user_input)
        if not parsed: 
            print("Invalid input format. Please try again.")
            continue
            
        white_pieces_str, black_pieces_str = parsed
        num_pieces = len(white_pieces_str) + len(black_pieces_str)

        tb_dir = TABLEBASE_PATH / f"gtb{num_pieces}"
        filename_stem = get_tablebase_filename(white_pieces_str, black_pieces_str)
        tb_filepath1 = tb_dir / f"{filename_stem}.gtb.cp4"
        tb_filepath2 = tb_dir / f"{filename_stem}.gtb"
        
        tb_filepath = tb_filepath1 if tb_filepath1.exists() else tb_filepath2

        if not tb_filepath.exists():
            print(f"Error: Tablebase file not found for '{filename_stem}' in '{tb_dir}'")
            continue

        white_types = [PIECE_MAP_UPPER[p] for p in white_pieces_str]
        black_types = [PIECE_MAP_UPPER[p] for p in black_pieces_str]
        
        find_longest_checkmate(tb_filepath, white_types, black_types)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if not TABLEBASE_PATH.exists():
        print(f"FATAL ERROR: The specified tablebase path does not exist: '{TABLEBASE_PATH}'")
        sys.exit(1)
    main()