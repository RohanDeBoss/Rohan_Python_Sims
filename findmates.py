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

# --- CONFIGURATION ---
TABLEBASE_PATH = Path("C:/Program Files (x86)/Arena/TB/gtb.cp4")

# --- MAPPING and HELPERS ---
PIECE_MAP_UPPER = {'k': chess.KING, 'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT, 'p': chess.PAWN}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
SQUARE_DISTANCE_LOOKUP = [[chess.square_distance(s1, s2) for s2 in chess.SQUARES] for s1 in chess.SQUARES]
PAWN_INVALID_RANKS = frozenset([0, 7]) # Ranks 1 and 8
SYMMETRY_TRANSFORMS = [
    lambda sq: sq,
    lambda sq: chess.square_mirror(sq),
    lambda sq: chess.square(7 - chess.square_file(sq), chess.square_rank(sq)),
    lambda sq: chess.square(7 - chess.square_file(sq), 7 - chess.square_rank(sq))
]

def parse_piece_input(user_input: str) -> Optional[Tuple[List[str], List[str]]]:
    """Parses user input with detailed feedback."""
    try:
        user_input = user_input.lower().replace(" ", "")
        parts = user_input.split('b:')
        if len(parts) != 2:
            print("Error: Input must contain 'w:' and 'b:'. Example: w: k,r b: k")
            return None
        white_part, black_part = parts[0].replace('w:', '').strip(), parts[1].strip()
        white_pieces_str = [p for p in white_part.split(',') if p]
        black_pieces_str = [p for p in black_part.split(',') if p]

        if white_pieces_str.count('k') != 1 or black_pieces_str.count('k') != 1:
            print("Error: Both White and Black must have exactly one king ('k').")
            return None
        total_pieces = len(white_pieces_str) + len(black_pieces_str)
        if not (3 <= total_pieces <= 5):
             print(f"Warning: This tool is designed for 3-5 pieces. You entered {total_pieces}.")
        return white_pieces_str, black_pieces_str
    except Exception as e:
        print(f"Error parsing input: {e}")
        return None

def get_tablebase_filename(white_pieces_str: List[str], black_pieces_str: List[str]) -> str:
    white_types = sorted([PIECE_MAP_UPPER[p] for p in white_pieces_str], key=lambda x: PIECE_VALUES[x], reverse=True)
    black_types = sorted([PIECE_MAP_UPPER[p] for p in black_pieces_str], key=lambda x: PIECE_VALUES[x], reverse=True)
    w_str = "".join(chess.piece_symbol(pt).upper() for pt in white_types)
    b_str = "".join(chess.piece_symbol(pt).lower() for pt in black_types)
    return (w_str + b_str).lower()

def worker_find_mate_in_chunk(
    combinations_chunk: List[Tuple[int, ...]], tb_path_str: str, all_piece_types: List[Tuple[int, bool]]
) -> Tuple[int, List[str]]:
    """Optimized worker using symmetry pruning for speed."""
    max_dtm_local, best_fens_local = -1, []
    board = chess.Board(None)
    seen_canonical_squares = set()
    is_pawn_at_index = {i for i, (pt, _) in enumerate(all_piece_types) if pt == chess.PAWN}
    is_king_at_index = {i for i, (pt, _) in enumerate(all_piece_types) if pt == chess.KING}

    with chess.gaviota.open_tablebase(tb_path_str) as tb:
        for squares_combo in combinations_chunk:
            if is_pawn_at_index and any(chess.square_rank(squares_combo[i]) in PAWN_INVALID_RANKS for i in is_pawn_at_index):
                continue

            kings = [squares_combo[i] for i in is_king_at_index]
            if len(kings) == 2 and SQUARE_DISTANCE_LOOKUP[kings[0]][kings[1]] <= 1:
                continue

            sorted_squares = tuple(sorted(squares_combo))
            canonical_squares = min(tuple(sorted(t(sq) for sq in sorted_squares)) for t in SYMMETRY_TRANSFORMS)
            if canonical_squares in seen_canonical_squares:
                continue
            seen_canonical_squares.add(canonical_squares)

            for piece_perm in set(itertools.permutations(all_piece_types)):
                board.clear_board()
                for i, (pt, color) in enumerate(piece_perm):
                    board.set_piece_at(squares_combo[i], chess.Piece(pt, color))
                board.turn = chess.WHITE

                if not board.is_valid(): continue
                dtm_value = tb.probe_dtm(board)
                if dtm_value > max_dtm_local:
                    max_dtm_local = dtm_value
                    best_fens_local = [board.fen()]
                elif dtm_value == max_dtm_local and dtm_value > 0:
                    best_fens_local.append(board.fen())
    return max_dtm_local, best_fens_local

def unpack_and_run_worker(args):
    return worker_find_mate_in_chunk(*args)

def find_longest_checkmate(tb_filepath: Path, white_piece_types: List[int], black_piece_types: List[int]):
    num_pieces = len(white_piece_types) + len(black_piece_types)
    if num_pieces == 5:
        print("\n--- WARNING: 5-piece search can still take a very long time. ---")
        if input("Are you sure you want to continue? (y/n): ").lower() != 'y': return

    all_piece_types = [(pt, chess.WHITE) for pt in white_piece_types] + [(pt, chess.BLACK) for pt in black_piece_types]
    
    print("\nGenerating all square combinations...")
    all_combinations = list(itertools.combinations(chess.SQUARES, num_pieces))
    total_combinations, num_cores = len(all_combinations), os.cpu_count() or 1
    
    CHUNK_MULTIPLIER, num_chunks = 64, num_cores * 64
    chunk_size = (total_combinations + num_chunks - 1) // num_chunks
    
    print(f"Initializing search across {num_cores} CPU cores, split into {num_chunks} chunks.")
    chunks = [all_combinations[i:i + chunk_size] for i in range(0, total_combinations, chunk_size)]
    worker_args = [(chunk, str(tb_filepath.parent), all_piece_types) for chunk in chunks]
    
    print(f"Analyzing {num_pieces}-piece endgame ({total_combinations:,} combinations)...")
    
    results, current_max_dtm = [], -1
    with multiprocessing.Pool(processes=num_cores) as pool:
        progress_iterator = pool.imap_unordered(unpack_and_run_worker, worker_args)
        with tqdm(total=len(chunks), desc="Processing Chunks") as p_bar:
            for result in progress_iterator:
                results.append(result)
                max_dtm_result, _ = result
                if max_dtm_result > current_max_dtm:
                    current_max_dtm = max_dtm_result
                    p_bar.set_description(f"Best so far: Mate in {(current_max_dtm + 1) // 2} (DTM={current_max_dtm})")
                p_bar.update(1)
    
    print("\nCombining results and finding longest mate...")
    final_max_dtm, incomplete_best_fens = -1, []
    for max_dtm_result, best_fens_result in results:
        if max_dtm_result > final_max_dtm:
            final_max_dtm = max_dtm_result
            incomplete_best_fens = best_fens_result
        elif max_dtm_result == final_max_dtm and max_dtm_result > 0:
            incomplete_best_fens.extend(best_fens_result)

    if final_max_dtm > 0:
        # --- CORRECTED EXPANSION STAGE ---
        print("Expanding results to include all symmetrical positions...")
        fully_expanded_fens = set()
        for fen in tqdm(list(set(incomplete_best_fens)), desc="Expanding Symmetries"):
            board = chess.Board(fen)
            # Add the original position and its true geometric transformations
            fully_expanded_fens.add(board.fen())
            board_h = board.copy().transform(chess.flip_horizontal)
            fully_expanded_fens.add(board_h.fen())
            board_v = board.copy().transform(chess.flip_vertical)
            fully_expanded_fens.add(board_v.fen())
            board_hv = board_h.copy().transform(chess.flip_horizontal).transform(chess.flip_vertical)
            fully_expanded_fens.add(board_hv.fen())
        
        final_best_fens = sorted(list(fully_expanded_fens))
        # --- END OF CORRECTED STAGE ---

        mate_in_moves = (final_max_dtm + 1) // 2
        print("\n--- Search Complete ---")
        print(f"Found longest forced checkmate for White: Mate in {mate_in_moves} moves (DTM={final_max_dtm})")
        print(f"Found {len(final_best_fens)} position(s) with this result:")
        for fen in final_best_fens:
            print(f"  - FEN: {fen}")
    else:
        print("\n--- Search Complete ---")
        print("No forced checkmate for White was found in any legal position for this material.")

def main():
    print("--- Longest Forced Checkmate Finder (Optimized High-Speed Version) ---")
    while True:
        user_input = input("\nEnter pieces (e.g., 'w: k,r b: k') or 'exit' > ")
        if user_input.lower() in ['exit', 'quit']: break
        parsed = parse_piece_input(user_input)
        if not parsed: continue
        white_pieces_str, black_pieces_str = parsed
        tb_dir = TABLEBASE_PATH / f"gtb{len(white_pieces_str) + len(black_pieces_str)}"
        filename_stem = get_tablebase_filename(white_pieces_str, black_pieces_str)
        tb_filepath = next(tb_dir.glob(f"{filename_stem}.gtb*"), None)
        if not tb_filepath:
            print(f"Error: Tablebase file not found for '{filename_stem}' in '{tb_dir}'")
            continue
        white_types = [PIECE_MAP_UPPER[p] for p in white_pieces_str]
        black_types = [PIECE_MAP_UPPER[p] for p in black_pieces_str]
        find_longest_checkmate(tb_filepath, white_types, black_types)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if not TABLEBASE_PATH.exists():
        print(f"FATAL ERROR: Tablebase path does not exist: '{TABLEBASE_PATH}'")
        sys.exit(1)
    main()