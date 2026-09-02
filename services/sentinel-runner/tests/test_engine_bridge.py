import sys
import tempfile
import unittest
from pathlib import Path

from sentinel_runner.engine_bridge import load_engine


class EngineBridgeTests(unittest.TestCase):
    def test_engine_can_import_companion_modules_from_its_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            companion_name = "eidos_runner_test_companion"
            (root / f"{companion_name}.py").write_text("VALUE = 'loaded'\n", encoding="utf-8")
            engine_path = root / "EIDOS_BRAIN_UNIFIED_v0_4.7.02.py"
            engine_path.write_text(
                f"import {companion_name}\n"
                "ENGINE_VERSION = '0.4.7.02'\n"
                f"COMPANION_VALUE = {companion_name}.VALUE\n"
                "def run_stream_once(*args, **kwargs):\n"
                "    return {}\n",
                encoding="utf-8",
            )
            try:
                engine = load_engine(engine_path, root / "artifacts")
                self.assertEqual(engine.COMPANION_VALUE, "loaded")
                self.assertNotIn(str(root), sys.path)
            finally:
                sys.modules.pop(companion_name, None)


if __name__ == "__main__":
    unittest.main()
