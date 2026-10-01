from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import encode_line, parse_observation, read_line


class ProtocolTests(unittest.TestCase):
    def test_line_encoding_and_reading(self):
        encoded = encode_line("ACTION", 17, 2)
        self.assertEqual(encoded, b"ACTION 17 2\n")
        self.assertEqual(read_line(io.BytesIO(encoded)), "ACTION 17 2")

    def test_observation_parsing(self):
        self.assertEqual(parse_observation("OBS 17 3 -0.25 1"), (17, 3, -0.25, True))

    def test_rejects_invalid_observations(self):
        for line in ("OBS 17 8 0 0", "OBS -1 0 0 0", "OBS 17 0 nan 0", "OBS 17 0 0 2"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_observation(line)

    def test_rejects_oversized_or_unterminated_messages(self):
        with self.assertRaises(ValueError):
            encode_line("x" * 128)
        with self.assertRaises(ValueError):
            read_line(io.BytesIO(b"x" * 129))


if __name__ == "__main__":
    unittest.main()
