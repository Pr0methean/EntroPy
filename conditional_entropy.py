import numpy as np
import sys


def print_entropy_stats(filepath, chunk_size=1024 * 1024 * 16):  # 16MB chunks
    # Initialize a 256x256 transition matrix
    # uint64 allows for counts up to 18 quintillion (plenty for 16GiB)
    matrix = np.zeros((256, 256), dtype=np.uint64)
    byte_counts = np.zeros(256, dtype=np.uint64)
    # Order-2 matrix: [prev_byte1][prev_byte2][current_byte]
    # To save RAM, we flatten the first two dimensions: [65536][256]
    # 65536 * 256 * 8 bytes (uint64) = 128 MB RAM. Very efficient.
    matrix2 = np.zeros((65536, 256), dtype=np.uint64)

    total_bytes = 0
    overlap = None

    try:
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Convert buffer to numpy array of bytes (uint8)
                arr = np.frombuffer(chunk, dtype=np.uint8)
                chunk_len = len(arr)
                total_bytes += chunk_len

                # 1. Update Zero-Order counts
                byte_counts += np.bincount(arr, minlength=256).astype(np.uint64)

                if overlap is not None:
                    arr = np.concatenate((overlap[-1], arr))

                # 2. Update First-Order transitions (The Vectorized Magic)
                # We pair arr[:-1] with arr[1:] to get every transition in the chunk
                x = arr[:-1]
                y = arr[1:]

                # Use a single index for the 2D matrix: 256 * row + col
                combined = x.astype(np.uint64) * 256 + y
                matrix.flat += np.bincount(combined, minlength=65536).astype(np.uint64)

                # Create indices for [prev1, prev2, current]
                # Index = (prev1 << 8 | prev2)
                idx_prev = arr[:-2].astype(np.uint32) << 8 | arr[1:-1]
                target = arr[2:]

                # 3. Update 2nd-order counts
                if overlap is not None:
                    arr = np.concatenate((overlap[:-2], arr))
                # Vectorized update
                combined = (idx_prev.astype(np.uint64) << 8) | target
                matrix2.flat += np.bincount(combined, minlength=16777216).astype(np.uint64)

                overlap = arr[-2:]  # Keep last 2 bytes for the next chunk transition
                print(f"Progress: {total_bytes / (1024 ** 3):.2f} GiB processed...", end='\n\r')
    except FileNotFoundError:
        return "File not found."
    print(f"\nFinal Results for {total_bytes / (1024 ** 3):.2f} GiB:")
    # --- Calculations ---
    # H0 (Zero-Order)
    p0 = byte_counts / total_bytes
    p0 = p0[p0 > 0]  # Remove zeros to avoid log(0)
    h0 = -np.sum(p0 * np.log2(p0))
    print(f"H0:         {h0:.15f} bits/byte")
    # H1|0 (Conditional Entropy)
    # row_sums is the count of how many times each byte was a 'predecessor'
    row_sums = matrix.sum(axis=1)

    h1_cond = 0.0
    for i in range(256):
        if row_sums[i] > 0:
            # Probability of preceding byte i
            p_y = row_sums[i] / (total_bytes - 1)

            # Row probabilities P(X|Y=i)
            row_probs = matrix[i, :] / row_sums[i]
            row_probs = row_probs[row_probs > 0]

            h_given_y = -np.sum(row_probs * np.log2(row_probs))
            h1_cond += p_y * h_given_y

    print(f"H1|0:                         {h1_cond:.15f} bits/byte")

    # G-Statistic (Likelihood Ratio)
    # G^2 = 2 * N * ln(2) * (H0 - H1)
    g2 = 2 * (total_bytes - 1) * np.log(2) * (h0 - h1_cond)
    df = 65025

    print(f"G-test:                       {g2:.15f} (df={df})")
    print(f"Z-Score:                      {(g2 - df) / np.sqrt(2 * df):.15f}")
    print(f"Theil's U:                    {(h0 - h1_cond) / h0:.15f}")

    row_sums = matrix2.sum(axis=1)
    active_rows = row_sums > 0

    # Conditional Entropy H(X | Y, Z)
    # Probability of the pair (Y, Z)
    p_condition = row_sums[active_rows] / (total_bytes - 2)

    # Entropy of each row
    h_rows = []
    for i in np.where(active_rows)[0]:
        row = matrix2[i, :].astype(np.float64)
        p_row = row[row > 0] / row_sums[i]
        h_rows.append(-np.sum(p_row * np.log2(p_row)))

    h2_cond = np.sum(p_condition * np.array(h_rows))

    # G-Test for Order-2
    # df = 256^2 * (256-1) = 16,711,680
    g2 = 2 * (total_bytes - 2) * np.log(2) * (h0 - h2_cond)  # Assuming H0 is 8.0
    df = 16711680

    print(f"\nOrder-2 Results (3-byte tuples):")
    print(f"Conditional Entropy (H2|1,0): {h2_cond:.15f} bits/byte")
    print(f"G-test:                       {g2:.15f} (df={df})")
    print(f"Z-Score:                      {(g2 - df) / np.sqrt(2 * df):.15f}")
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
    else:
        print_entropy_stats(sys.argv[1])


