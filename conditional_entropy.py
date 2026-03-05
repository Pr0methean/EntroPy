import numpy as np
import sys


def print_entropy_stats(filepath, chunk_size=1024 * 1024 * 16):  # 16MB chunks

    try:
        with open(filepath, 'rb') as f:
            # Initialize a 256x256 transition matrix
            # uint64 allows for counts up to 18 quintillion (plenty for 16GiB)
            matrix = np.zeros((256, 256), dtype=np.uint64)
            byte_counts = np.zeros(256, dtype=np.uint64)
            # Order-2 matrix: [prev_byte1][prev_byte2][current_byte]
            # To save RAM, we flatten the first two dimensions: [65536][256]
            # 65536 * 256 * 8 bytes (uint64) = 128 MB RAM. Very efficient.
            matrix2 = np.zeros((65536, 256), dtype=np.uint64)

            total_bytes = 0
            last_byte = None  # For first-order bridging
            last_two_bytes = None  # For second-order bridging (as a tuple or int pair)

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Convert buffer to numpy array of bytes (uint8)
                arr = np.frombuffer(chunk, dtype=np.uint8)
                current_len = len(arr)

                # Update byte counts (only count new bytes)
                byte_counts += np.bincount(arr, minlength=256).astype(np.uint64)

                # Bridging transition from previous chunk
                if last_byte is not None:
                    combined = last_byte.astype(int) * 256 + arr[0]
                    matrix.flat[combined] += 1

                # Transitions within new chunk
                if len(arr) >= 2:
                    x = arr[:-1]
                    y = arr[1:]
                    combined = x.astype(int) * 256 + y
                    matrix.flat += np.bincount(combined, minlength=65536).astype(np.uint64)

                # Need at least 2 bytes for context + 1 target

                # Bridging transitions from previous chunk
                if last_two_bytes is not None:
                    # last_two_bytes is a tuple (byte1, byte2) representing the last
                    # two bytes from the previous chunk in order

                    # We need at least 1 byte in arr to form a triple
                    if len(arr) >= 1:
                        # Triple: (last_two_bytes[0], last_two_bytes[1], arr[0])
                        idx_prev = (last_two_bytes[0] << 8) | last_two_bytes[1]
                        combined = (idx_prev << 8) | arr[0]
                        matrix2.flat[combined] += 1

                # If we have last_byte but not last_two_bytes, we might still form
                # a bridging triple if arr provides enough context
                if last_byte is not None and len(arr) >= 2:
                    # Triple: (last_byte, arr[0], arr[1])
                    idx_prev = (last_byte << 8) | arr[0]
                    combined = (idx_prev << 8) | arr[1]
                    matrix2.flat[combined] += 1

                if len(arr) >= 3:
                    idx_prev = arr[:-2].astype(int) * 256 + arr[1:-1]
                    target = arr[2:]
                    combined = idx_prev * 256 + target
                    matrix2.flat += np.bincount(combined, minlength=16777216).astype(np.uint64)

                total_bytes += current_len
                # Update last bytes for next chunk
                if len(arr) >= 2:
                    last_two_bytes = (arr[-2], arr[-1])
                    last_byte = arr[-1]
                elif len(arr) == 1:
                    last_two_bytes = None  # Not enough for two-byte context
                    last_byte = arr[0]

                print(f"Progress: {total_bytes / (1024 ** 3):.2f} GiB processed...", end='\n\r')

    except FileNotFoundError:
        return "File not found."

    print(f"\nFinal Results for {total_bytes / (1024 ** 3):.2f} GiB:")

    # Zero-order entropy
    p0 = byte_counts / total_bytes
    p0 = p0[p0 > 0]  # Remove zeros to avoid log(0)
    h0 = -np.sum(p0 * np.log2(p0))
    print(f"H0:                           {h0:.15f} bits/byte")

    # First-order conditional entropy
    row_sums = matrix.sum(axis=1)
    h1_cond = 0.0
    for i in range(256):
        if row_sums[i] > 0:
            # Probability of preceding byte i
            p_y = row_sums[i] / (total_bytes - 1)

            # Row probabilities P(X|Y=i)
            row_probs = matrix[i, :] / row_sums[i]
            row_probs = row_probs[row_probs > 0]
            if len(row_probs) > 0:
                h_given_y = -np.sum(row_probs * np.log2(row_probs))
                h1_cond += p_y * h_given_y

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

        h_rows = []
        for i in np.where(active_rows)[0]:
            row = matrix2[i, :].astype(np.float64)
            p_row = row[row > 0] / row_sums2[i]
            if len(p_row) > 0:
                h_rows.append(-np.sum(p_row * np.log2(p_row)))

        h2_cond = np.sum(p_condition * np.array(h_rows))

        # G-test for second order (using actual H0, not hardcoded 8.0)
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
        print_entropy_stats(sys.argv[1])


