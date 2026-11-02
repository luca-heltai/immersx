"""Ensure benchmark builds cannot silently bypass their shared compiler cache."""

import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import compiler_cache


class CompilerCache(unittest.TestCase):
    def test_effective_disable_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"CCACHE_DIR": str(root / "cache"), "CXXFLAGS": ""}), \
                    patch("pipeline.shutil.which", return_value="/usr/bin/ccache"), \
                    patch("pipeline.subprocess.check_output", return_value="true\n"):
                with self.assertRaisesRegex(ValueError, "disabled"):
                    compiler_cache(root / "source", root / "source/build")

    def test_missing_cache_is_rejected(self):
        with patch("pipeline.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "required"):
                compiler_cache(Path("source"), Path("source/build"))

    def test_paths_with_spaces_remain_single_compiler_arguments(self):
        with tempfile.TemporaryDirectory(prefix="benchmark cache ") as directory:
            root = Path(directory)
            source, build = root / "source", root / "source/build"
            with patch.dict(os.environ, {"CCACHE_DIR": str(root / "cache"), "CXXFLAGS": ""}), \
                    patch("pipeline.shutil.which", return_value="/usr/bin/ccache"), \
                    patch("pipeline.subprocess.check_output", return_value="false\n") as effective:
                flags, env = compiler_cache(source, build)
            options = shlex.split(next(f.split("=", 1)[1] for f in flags
                                       if f.startswith("-DCMAKE_CXX_FLAGS=")))
            self.assertEqual(len(options), 6)
            for path, name in ((source, "source"), (build, "build")):
                for kind in ("file", "debug", "macro"):
                    self.assertIn(f"-f{kind}-prefix-map={path}=/immersx/{name}", options)
            self.assertEqual(env["CCACHE_DIR"], str(root / "cache"))
            self.assertEqual(env["CCACHE_BASEDIR"], str(source))
            self.assertEqual(env["CCACHE_NOHASHDIR"], "true")
            self.assertEqual(effective.call_args.kwargs["env"], env)


if __name__ == "__main__":
    unittest.main()
