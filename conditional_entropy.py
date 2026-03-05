import numpy as np
import sys


def optimize_stream_entropy(filepath, chunk_size=1024 * 1024 * 16):  # 16MB chunks
    # Initialize a 256x256 transition matrix
    # uint64 allows for counts up to 18 quintillion (plenty for 16GiB)
    matrix = np.zeros((256, 256), dtype=np.uint64)
    byte_counts = np.zeros(256, dtype=np.uint64)

    total_bytes = 0
    last_byte = None

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

                # 2. Update First-Order transitions (The Vectorized Magic)
                # We pair arr[:-1] with arr[1:] to get every transition in the chunk
                x = arr[:-1]
                y = arr[1:]

                # Use a single index for the 2D matrix: 256 * row + col
                combined = x.astype(np.uint64) * 256 + y
                matrix.flat += np.bincount(combined, minlength=65536).astype(np.uint64)

                # Handle the edge case: transition between the last chunk and this one
                if last_byte is not None:
                    matrix[last_byte, arr[0]] += 1

                last_byte = arr[-1]
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

    # G-Statistic (Likelihood Ratio)
    # G^2 = 2 * N * ln(2) * (H0 - H1)
    g2 = 2 * (total_bytes - 1) * np.log(2) * (h0 - h1_cond)

    return h0, h1_cond, g2, total_bytes


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
    else:
        h0, h1, g2, size = optimize_stream_entropy(sys.argv[1])

        print(f"H1|0:       {h1:.15f} bits/byte")
        print(f"G-test:     {g2:.5f} (df=65025)")
        print(f"Theil's U:  {(h0 - h1) / h0:.15f}")
