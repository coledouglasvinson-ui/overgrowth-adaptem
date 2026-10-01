from __future__ import annotations

import socket
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env import OvergrowthCombatEnv
from protocol import read_line


class EnvironmentTests(unittest.TestCase):
    def test_lockstep_reset_and_step(self):
        environment = OvergrowthCombatEnv(port=0, timeout=3.0)
        port = environment._listener.getsockname()[1]
        failures: list[BaseException] = []

        def fake_game() -> None:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=3.0) as game:
                    reader = game.makefile("rb")
                    game.sendall(b"HELLO 1\n")
                    self.assertEqual(read_line(reader), "START")
                    game.sendall(b"OBS 9 3 0 0\n")
                    self.assertEqual(read_line(reader), "ACTION 9 2")
                    game.sendall(b"OBS 9 6 0.25 1\n")
                    self.assertEqual(read_line(reader), "RESET")
                    game.sendall(b"RESET_OK\n")
                    self.assertEqual(read_line(reader), "START")
                    game.sendall(b"OBS 10 1 0 0\n")
                    reader.close()
            except Exception as exc:
                failures.append(exc)

        game_thread = threading.Thread(target=fake_game, daemon=True)
        game_thread.start()
        try:
            observation, info = environment.reset(seed=1)
            self.assertEqual((observation, info), (3, {"character_id": 9}))
            self.assertEqual(
                environment.step(2),
                (6, 0.25, True, False, {"character_id": 9}),
            )
            self.assertEqual(environment.reset()[0], 1)
        finally:
            environment.close()
            game_thread.join(timeout=3.0)

        self.assertFalse(game_thread.is_alive())
        if failures:
            raise failures[0]


if __name__ == "__main__":
    unittest.main()
