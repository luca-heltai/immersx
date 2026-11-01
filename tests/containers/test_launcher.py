"""Exercise argument forwarding without requiring a container runtime."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class Entrypoint(unittest.TestCase):
    def test_help_dispatch_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            bindir = Path(directory)
            entry = bindir / "immersx"
            shutil.copyfile(ROOT / "scripts/container_entrypoint.sh", entry)
            app = bindir / "poisson"
            app.write_text('#!/bin/bash\nprintf "<%s>\\n" "$@"\nexit 23\n')
            app.chmod(0o755)
            for args in ([], ["--help"], ["-h"]):
                result = subprocess.run(["bash", str(entry), *args],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0)
                self.assertIn("  poisson\n", result.stdout)
                self.assertNotIn("  immersx\n", result.stdout)
            result = subprocess.run(["bash", str(entry), "poisson", "input with spaces.prm", "--flag"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 23)
            self.assertEqual(result.stdout, "<input with spaces.prm>\n<--flag>\n")
            for name in ("unknown", "../poisson", "/bin/sh", "immersx"):
                result = subprocess.run(["bash", str(entry), name], capture_output=True)
                self.assertEqual(result.returncode, 127)


class Launcher(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "case with spaces"
        self.work.mkdir()
        self.log = self.root / "arguments.json"
        self.env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                        CONTAINER_TEST_LOG=str(self.log))
        for runtime in ("docker", "podman", "apptainer"):
            path = self.root / runtime
            path.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                            'with open(os.environ["CONTAINER_TEST_LOG"], "w") as out:\n'
                            '    json.dump(sys.argv[1:], out)\n'
                            'sys.exit(int(os.environ.get("CONTAINER_TEST_EXIT", "0")))\n')
            path.chmod(0o755)

    def run_launcher(self, runtime, *arguments, image="example:test"):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/run_container.sh"), "--runtime", runtime,
             "--image", image, "--workdir", str(self.work), *arguments],
            cwd="/tmp", env=self.env, capture_output=True, text=True)

    def test_docker_and_podman_forward_application_arguments(self):
        for runtime in ("docker", "podman"):
            result = self.run_launcher(runtime, "poisson", "file with spaces.prm", "--flag")
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(self.log.read_text())
            self.assertEqual(args[-4:], ["example:test", "poisson", "file with spaces.prm", "--flag"])
            self.assertIn(f"type=bind,source={self.work},target=/work", args)
            self.assertIn(f"{os.getuid()}:{os.getgid()}", args)
            self.assertEqual(args[args.index("--entrypoint") + 1], "/opt/immersx/bin/immersx")

    def test_no_application_and_mpi(self):
        self.assertEqual(self.run_launcher("docker").returncode, 0)
        args = json.loads(self.log.read_text())
        self.assertEqual(args[-1], "example:test")
        self.assertEqual(self.run_launcher("docker", "--ranks", "2", "poisson", "in.prm").returncode, 0)
        args = json.loads(self.log.read_text())
        self.assertEqual(args[-8:], ["mpirun", "example:test", "--oversubscribe", "-n", "2",
                                    "/opt/immersx/bin/immersx", "poisson", "in.prm"])

    def test_apptainer_oci_and_sif(self):
        for image, expected in (("example:test", "docker://example:test"),
                                ("docker://example:test", "docker://example:test"),
                                ("/tmp/local image.sif", "/tmp/local image.sif")):
            result = self.run_launcher("apptainer", "poisson", "in.prm", image=image)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(self.log.read_text())
            self.assertEqual(args[-4:], [expected, "/opt/immersx/bin/immersx", "poisson", "in.prm"])
            self.assertIn(f"{self.work}:/work", args)
            self.assertIn("--cleanenv", args)

    def test_custom_command_and_runtime_failure(self):
        result = self.run_launcher("docker", "--exec", "bash", "-c", "printf '%s' 'a b'")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.log.read_text())
        self.assertEqual(args[-4:], ["bash", "example:test", "-c", "printf '%s' 'a b'"])
        self.env["CONTAINER_TEST_EXIT"] = "42"
        self.assertEqual(self.run_launcher("apptainer", "poisson").returncode, 42)

    def test_invalid_options_fail_before_invoking_runtime(self):
        for arguments in (("--ranks", "0"), ("--ranks", "x"), ("--image",), ("--exec",), ("--unknown",)):
            self.assertNotEqual(self.run_launcher("docker", *arguments).returncode, 0)
            self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
