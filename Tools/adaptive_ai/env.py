from __future__ import annotations

import ipaddress
import socket
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from protocol import ACTION_COUNT, PROTOCOL_VERSION, encode_line, parse_observation, read_line


class OvergrowthCombatEnv(gym.Env[int, int]):
    """Lockstep combat strategy environment backed by a local Overgrowth process."""

    metadata: dict[str, Any] = {}

    def __init__(self, port: int = 47852, timeout: float = 120.0):
        super().__init__()
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Discrete(8)
        self._timeout = timeout
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", port))
        self._listener.listen(1)
        self._listener.settimeout(timeout)
        self._connection: socket.socket | None = None
        self._reader = None
        self._started = False
        self._episode_active = False
        self._character_id: int | None = None

    def _connect(self) -> None:
        if self._connection is not None:
            return

        connection, address = self._listener.accept()
        if not ipaddress.ip_address(address[0]).is_loopback:
            connection.close()
            raise ConnectionError("refusing non-loopback game connection")

        connection.settimeout(self._timeout)
        reader = connection.makefile("rb")
        greeting = read_line(reader)
        if greeting != f"HELLO {PROTOCOL_VERSION}":
            reader.close()
            connection.close()
            raise ValueError(f"unsupported game protocol greeting: {greeting!r}")

        self._connection = connection
        self._reader = reader

    def _send(self, *parts: object) -> None:
        if self._connection is None:
            raise RuntimeError("game is not connected")
        try:
            self._connection.sendall(encode_line(*parts))
        except OSError as exc:
            raise ConnectionError("failed to send command to Overgrowth") from exc

    def _receive_observation(self) -> tuple[int, int, float, bool]:
        if self._reader is None:
            raise RuntimeError("game is not connected")
        line = read_line(self._reader)
        if line.startswith("ERROR "):
            raise RuntimeError(f"Overgrowth training bridge: {line[6:]}")
        observation = parse_observation(line)
        if self._character_id is not None and observation[0] != self._character_id:
            raise ValueError(
                f"game changed training character from {self._character_id} to {observation[0]}"
            )
        self._character_id = observation[0]
        return observation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._connect()
        if self._started:
            self._send("RESET")
            if read_line(self._reader) != "RESET_OK":
                raise ValueError("game did not acknowledge episode reset")
            self._character_id = None

        self._send("START")
        character_id, observation, _, terminal = self._receive_observation()
        if terminal:
            raise ValueError("game returned a terminal initial observation")
        self._started = True
        self._episode_active = True
        return observation, {"character_id": character_id}

    def step(self, action: int):
        if not self._episode_active or self._character_id is None:
            raise RuntimeError("reset() must be called before step() and after episode end")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be in [0, {ACTION_COUNT})")

        self._send("ACTION", self._character_id, int(action))
        character_id, observation, reward, terminated = self._receive_observation()
        self._episode_active = not terminated
        return observation, reward, terminated, False, {"character_id": character_id}

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._send("ABORT")
            except (ConnectionError, RuntimeError, ValueError):
                pass
            if self._reader is not None:
                self._reader.close()
                self._reader = None
            self._connection.close()
            self._connection = None
        self._listener.close()
