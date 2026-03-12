import math

import numpy as np
import sys
import os
import mmap
from numba import jit


@jit(nopython=True)
def process_mmap_data(data_array, matrix, matrix2, byte_counts):
    """
    Process the entire memory-mapped array in one pass.
    """
    n = len(data_array)

    # Update byte counts
    for i in range(n):
        byte_counts[data_array[i]] += 1
        if i < n - 1:
            # First order transitions
            combined = int(data_array[i]) * 256 + int(data_array[i + 1])
            matrix[combined // 256, combined % 256] += 1

            # Second order transitions
            if i < n - 2:
                target = int(data_array[i + 2])
                matrix2[combined, target] += 1
        if i % (1 << 26) == 0:
            print(f"Processed {i} bytes...")


def print_entropy_stats_mmap(filepath):
    """
    Memory-mapped version - processes entire file as a single numpy array.
    """
    try:
        file_size = os.path.getsize(filepath)
        print(f"File size: {file_size / (1024 ** 3):.2f} GiB")

        # Open file and create memory map
        with open(filepath, 'rb') as f:
            # Memory map the entire file
            data = np.memmap(filepath, dtype=np.uint8, mode='r')

            print(f"Processing {len(data)} bytes as a single array...")

            # Initialize matrices
            matrix = np.zeros((256, 256), dtype=np.uint64)
            matrix2 = np.zeros((65536, 256), dtype=np.uint64)
            byte_counts = np.zeros(256, dtype=np.uint64)

            # Process entire array in one go
            process_mmap_data(data, matrix, matrix2, byte_counts)

            total_bytes = len(data)

    except FileNotFoundError:
        return "File not found."
    except MemoryError:
        return "File too large to map into memory. Try the chunked version."
    print(f"\nFinal Results for {total_bytes / (1024 ** 3):.2f} GiB:")

    # Zero-order entropy
    p0 = byte_counts / total_bytes
    p0 = p0[p0 > 0]
    p0_sorted = np.sort(p0)
    h0 = -math.fsum(p0_sorted * np.log2(p0_sorted))
    print(f"H0:                           {h0:.15f} bits/byte")

    # First-order conditional entropy
    row_sums = matrix.sum(axis=1)
    valid_rows = row_sums > 0

    # Vectorized calculation where possible
    p_y = row_sums[valid_rows] / (total_bytes - 1)

    h_rows = np.zeros(math.fsum(valid_rows))
    row_indices = np.where(valid_rows)[0]

    for i, row_idx in enumerate(row_indices):
        row = matrix[row_idx, :]
        row_probs = row[row > 0] / row_sums[row_idx]
        row_probs_sorted = np.sort(row_probs)
        if len(row_probs) > 0:
            h_rows[i] = -math.fsum(row_probs_sorted * np.log2(row_probs_sorted))
    products = p_y * h_rows
    products_sorted = np.sort(products)  # Smallest to largest
    h1_cond = math.fsum(products_sorted)
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

        h_rows2 = np.zeros(math.fsum(active_rows))
        row_indices2 = np.where(active_rows)[0]

        for i, row_idx in enumerate(row_indices2):
            row = matrix2[row_idx, :]
            p_row = row[row > 0] / row_sums2[row_idx]
            if len(p_row) > 0:
                products = -p_row * np.log2(p_row)
                products_sorted = np.sort(products)
                h_rows2[i] = math.fsum(products_sorted)

    products = p_condition * h_rows2
    products_sorted = np.sort(products)  # Smallest to largest
    h2_cond = math.fsum(products_sorted)  # Sum smallest first for better accuracy

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
        print(f"Usage: python {__file__} <filename> [threshold_gb]")
    else:
        print_entropy_stats_mmap(sys.argv[1])