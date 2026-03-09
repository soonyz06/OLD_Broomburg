import time

def log_info(msg, start=None):
    if start is None:
        print(f"[INFO]{msg}...")
        return time.perf_counter()
    else:
        elapsed = time.perf_counter() - start
        print(f"[INFO]{msg} done in {elapsed:.2f}s\n\n")
        return None
