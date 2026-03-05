import numpy as np
import sys
import os


def streaming_hurst_inplace(filepath, block_size_gb=1):
    # Pre-calculate sizes
    block_bytes = block_size_gb * 1024 * 1024 * 1024
    lags = np.array([2 ** i for i in range(10, 25)])
    all_log_rs_sums = np.zeros(len(lags))

    # PRE-ALLOCATE MEMORY OBJECTS
    # 1GiB for the raw data (float64 to avoid precision issues in sums)
    # Or keep it as uint8 and convert slices:
    raw_buffer = np.empty(block_bytes, dtype=np.uint8)

    file_size = os.path.getsize(filepath)
    num_blocks = file_size // block_bytes

    print(f"Starting memory-stable Hurst analysis on {num_blocks} blocks...")

    with open(filepath, 'rb') as f:
        for b in range(num_blocks):
            # Read directly into our pre-allocated buffer
            f.readinto(raw_buffer)
            # Cast to float64 for math; we'll reuse 'data' as the working array
            data = raw_buffer.astype(np.float64)

            block_rs = []
            for lag in lags:
                num_windows = len(data) // lag
                reshaped = data[:num_windows * lag].reshape((num_windows, lag))

                # 1. In-place Mean
                # We calculate mean into a small array
                means = np.mean(reshaped, axis=1)[:, np.newaxis]

                # 5. Std Dev (S)
                s = np.std(reshaped, axis=1)

                # 2. In-place Deviation (reuse 'reshaped' memory if possible,
                # but here we'll just be careful with temp objects)
                # Subtraction creates a temp; we can't easily avoid one here without loops
                devs = reshaped - means

                del reshaped
                del means

                # 3. In-place Cumulative Sum (The biggest memory hog)
                # Using out=devs overwrites the deviations with the walk
                np.cumsum(devs, axis=1, out=devs)

                # 4. Range (R)
                r = np.max(devs, axis=1) - np.min(devs, axis=1)

                del devs

                valid = s > 0
                block_rs.append(np.mean(r[valid] / s[valid]))

            all_log_rs_sums += np.log(block_rs)
            print(f"Block {b + 1}/{num_blocks} done. Memory stable.", end='\n\r')

    avg_log_rs = all_log_rs_sums / num_blocks
    hurst_h = np.polyfit(np.log(lags), avg_log_rs, 1)[0]
    return hurst_h


if __name__ == "__main__":
    h_final = streaming_hurst_inplace(sys.argv[1])
    print(f"\nGlobal Hurst Exponent: {h_final:.15f}")

    # Interpretation thresholds for 2026
    if 0.495 < h_final < 0.505:
        print("Verdict: IDEAL RANDOMNESS (Perfectly uncorrelated walk)")
    elif 0.5 <= h_final <= 0.55:
        print("Verdict: NEGLIGIBLE PERSISTENCE (Safe for most uses)")
    elif h_final > 0.55:
        print("Verdict: PERSISTENT BIAS (Long-term trending detected)")
    else:
        print("Verdict: ANTI-PERSISTENT (Aggressive mean-reversion)")
