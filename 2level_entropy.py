import numpy as np
import sys


def analyze_order2_entropy(filepath, chunk_size=1024 * 1024 * 64):
    # Order-2 matrix: [prev_byte1][prev_byte2][current_byte]
    # To save RAM, we flatten the first two dimensions: [65536][256]
    # 65536 * 256 * 8 bytes (uint64) = 128 MB RAM. Very efficient.
    matrix = np.zeros((65536, 256), dtype=np.uint64)

    total_bytes = 0
    overlap = None

    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk: break

                arr = np.frombuffer(chunk, dtype=np.uint8)
                if overlap is not None:
                    arr = np.concatenate((overlap, arr))

                # Create indices for [prev1, prev2, current]
                # Index = (prev1 << 8 | prev2)
                idx_prev = arr[:-2].astype(np.uint32) << 8 | arr[1:-1]
                target = arr[2:]

                # Vectorized update
                combined = (idx_prev.astype(np.uint64) << 8) | target
                matrix.flat += np.bincount(combined, minlength=16777216).astype(np.uint64)

                total_bytes += len(chunk)
                overlap = arr[-2:]  # Keep last 2 bytes for the next chunk transition
                print(f"Processing: {total_bytes / (1024 ** 3):.2f} GiB...", end='\n\r')

    except FileNotFoundError:
        return "File not found."

    # Calculations
    row_sums = matrix.sum(axis=1)
    active_rows = row_sums > 0

    # Conditional Entropy H(X | Y, Z)
    # Probability of the pair (Y, Z)
    p_condition = row_sums[active_rows] / (total_bytes - 2)

    # Entropy of each row
    h_rows = []
    for i in np.where(active_rows)[0]:
        row = matrix[i, :].astype(np.float64)
        p_row = row[row > 0] / row_sums[i]
        h_rows.append(-np.sum(p_row * np.log2(p_row)))

    h2_cond = np.sum(p_condition * np.array(h_rows))

    # G-Test for Order-2
    # df = 256^2 * (256-1) = 16,711,680
    g2 = 2 * (total_bytes - 2) * np.log(2) * (8.0 - h2_cond)  # Assuming H0 is 8.0

    return h2_cond, g2, total_bytes


if __name__ == "__main__":
    h2, g2, size = analyze_order2_entropy(sys.argv[1])
    df = 16711680
    print(f"\nOrder-2 Results (3-byte tuples):")
    print(f"Conditional Entropy (H2|1,0): {h2:.15f} bits/byte")
    print(f"G-test: {g2:.15f} (df={df})")
    print(f"Z-Score: {(g2 - df) / np.sqrt(2 * df):.15f}")
