import numpy as np
import sys
from numba import jit


@jit(nopython=True)
def process_chunk_fast(arr, matrix, matrix2, byte_counts,
                       last_byte, last_two_bytes):
    """
    Process a chunk of bytes in a single pass with Numba acceleration.
    Returns updated counts and last bytes.
    """
    n = len(arr)

    # Update byte counts and transitions in a single pass
    for i in range(n):
        byte_counts[arr[i]] += 1

        if i < n - 1:
            # First order transitions
            combined = int(arr[i]) * 256 + int(arr[i + 1])
            matrix[combined // 256, combined % 256] += 1

            # Second order transitions (need 3 bytes)
            if i < n - 2:
                target = int(arr[i + 2])
                matrix2[combined, target] += 1

    # Handle bridging between chunks
    if last_byte is not None and n >= 1:
        combined = int(last_byte) * 256 + int(arr[0])
        matrix[combined // 256, combined % 256] += 1

    if last_two_bytes is not None and n >= 1:
        idx_prev = int(last_two_bytes[0]) * 256 + int(last_two_bytes[1])
        target = int(arr[0])
        matrix2[idx_prev, target] += 1

    # Update last bytes for next chunk
    if n >= 2:
        new_last_two = (arr[-2], arr[-1])
        new_last_byte = arr[-1]
    elif n == 1:
        new_last_two = None
        new_last_byte = arr[0]
    else:
        new_last_two = None
        new_last_byte = None

    return new_last_byte, new_last_two


def print_entropy_stats_optimized(filepath, chunk_size=1024 * 1024 * 64):  # 64MB chunks
    """
    Optimized version with single-pass processing and Numba acceleration.
    """
    try:
        with open(filepath, 'rb') as f:
            # Initialize matrices
            matrix = np.zeros((256, 256), dtype=np.uint64)
            matrix2 = np.zeros((65536, 256), dtype=np.uint64)
            byte_counts = np.zeros(256, dtype=np.uint64)

            total_bytes = 0
            last_byte = None
            last_two_bytes = None

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                arr = np.frombuffer(chunk, dtype=np.uint8)
                current_len = len(arr)

                # Process chunk in a single pass
                last_byte, last_two_bytes = process_chunk_fast(
                    arr, matrix, matrix2, byte_counts,
                    last_byte, last_two_bytes
                )

                total_bytes += current_len
                print(f"Progress: {total_bytes / (1024 ** 3):.2f} GiB processed...", end='\n\r')

    except FileNotFoundError:
        return "File not found."

    print(f"\nFinal Results for {total_bytes / (1024 ** 3):.2f} GiB:")

    # Zero-order entropy
    p0 = byte_counts / total_bytes
    p0 = p0[p0 > 0]
    h0 = -np.sum(p0 * np.log2(p0))
    print(f"H0:                           {h0:.15f} bits/byte")

    # First-order conditional entropy (vectorized)
    row_sums = matrix.sum(axis=1)
    valid_rows = row_sums > 0

    # Vectorized calculation where possible
    p_y = row_sums[valid_rows] / (total_bytes - 1)

    h_rows = np.zeros(np.sum(valid_rows))
    row_indices = np.where(valid_rows)[0]

    for i, row_idx in enumerate(row_indices):
        row = matrix[row_idx, :]
        row_probs = row[row > 0] / row_sums[row_idx]
        if len(row_probs) > 0:
            h_rows[i] = -np.sum(row_probs * np.log2(row_probs))

    h1_cond = np.sum(p_y * h_rows)
    print(f"H1|0:                         {h1_cond:.15f} bits/byte")

    # G-test for first order
    g2 = 2 * (total_bytes - 1) * np.log(2) * (h0 - h1_cond)
    df = 65025
    print(f"G-test:                       {g2:.15f} (df={df})")
    print(f"Z-Score:                      {(g2 - df) / np.sqrt(2 * df):.15f}")
    print(f"Theil's U:                    {(h0 - h1_cond) / h0:.15f}")

    # Second-order conditional entropy
    row_sums2 = matrix2.sum(axis=1)
    active_rows = row_sums2 > 0

    if np.any(active_rows):
        p_condition = row_sums2[active_rows] / (total_bytes - 2)

        h_rows2 = np.zeros(np.sum(active_rows))
        row_indices2 = np.where(active_rows)[0]

        for i, row_idx in enumerate(row_indices2):
            row = matrix2[row_idx, :]
            p_row = row[row > 0] / row_sums2[row_idx]
            if len(p_row) > 0:
                h_rows2[i] = -np.sum(p_row * np.log2(p_row))

        h2_cond = np.sum(p_condition * h_rows2)

        # G-test for second order
        g2_2 = 2 * (total_bytes - 2) * np.log(2) * (h0 - h2_cond)
        df2 = 16711680

        print(f"\nOrder-2 Results (3-byte tuples):")
        print(f"Conditional Entropy (H2|1,0): {h2_cond:.15f} bits/byte")
        print(f"G-test:                       {g2_2:.15f} (df={df2})")
        print(f"Z-Score:                      {(g2_2 - df2) / np.sqrt(2 * df2):.15f}")

    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {__file__} <filename>")
    else:
        print_entropy_stats_optimized(sys.argv[1])