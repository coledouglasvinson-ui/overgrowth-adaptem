from __future__ import annotations

import math
from typing import BinaryIO

MAX_LINE_BYTES = 128
PROTOCOL_VERSION = 1
ACTION_COUNT = 4
OBSERVATION_COUNT = 8


def encode_line(*parts: object) -> bytes:
    line = " ".join(str(part) for part in parts) + "\n"
    encoded = line.encode("ascii")
    if len(encoded) > MAX_LINE_BYTES:
        raise ValueError(f"protocol line exceeds {MAX_LINE_BYTES} bytes")
    return encoded


def read_line(reader: BinaryIO) -> str:
    raw = reader.readline(MAX_LINE_BYTES + 1)
    if not raw:
        raise ConnectionError("game connection closed")
    if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
        raise ValueError(f"protocol line exceeds {MAX_LINE_BYTES} bytes or is unterminated")
    try:
        return raw[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("protocol messages must be ASCII") from exc


def parse_observation(line: str) -> tuple[int, int, float, bool]:
    fields = line.split()
    if len(fields) != 5 or fields[0] != "OBS":
        raise ValueError(f"expected OBS message, received {line!r}")

    try:
        character_id = int(fields[1])
        observation = int(fields[2])
        reward = float(fields[3])
        terminal_flag = int(fields[4])
    except ValueError as exc:
        raise ValueError(f"invalid OBS message: {line!r}") from exc

    if character_id < 0:
        raise ValueError("observation character ID must be nonnegative")
    if not 0 <= observation < OBSERVATION_COUNT:
        raise ValueError(f"observation must be in [0, {OBSERVATION_COUNT})")
    if not math.isfinite(reward):
        raise ValueError("observation reward must be finite")
    if terminal_flag not in (0, 1):
        raise ValueError("terminal flag must be 0 or 1")

    return character_id, observation, reward, bool(terminal_flag)
