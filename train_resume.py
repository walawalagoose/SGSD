from __future__ import annotations

import os


def find_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None

    checkpoints: list[tuple[int, str]] = []
    for entry in os.listdir(output_dir):
        if not entry.startswith("checkpoint-"):
            continue
        try:
            step = int(entry.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, entry))

    if not checkpoints:
        return None

    _, latest_entry = max(checkpoints, key=lambda item: item[0])
    return os.path.join(output_dir, latest_entry)
