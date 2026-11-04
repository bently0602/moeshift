#!/usr/bin/env python3
"""Offline first-run tests using the same GGUF tools installed by Homebrew."""
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from io import BytesIO, StringIO
import json
import os
import pty
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import termios
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/moe"))
import onboarding as setup
from gguf import GGUFWriter


class OnboardingTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        self.source = self.home / "Files with spaces"
        self.source.mkdir()
        self.config = self.home / "config.json"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, HOME=str(self.home)))
        self.mock("onboarding.TEMPLATE", ROOT / "configs/moeshift.json")
        self.log = self.stack.enter_context(redirect_stdout(StringIO()))

    def mock(self, *args, **kwargs):
        return self.stack.enter_context(patch(*args, **kwargs))

    def test_arrow_menu_and_terminal_restoration(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        stream = self.stack.enter_context(os.fdopen(slave))
        input_flags = termios.ICANON | termios.ECHO | termios.ISIG
        previous = termios.tcgetattr(slave)[3] & input_flags
        with patch("sys.stdin", stream):
            for keys, selected in ((b"\n", 0), (b"2\x1b[B\x1b[B\x1bOA\n", 1), (b"\x1b[A\n", 2)):
                os.write(master, keys)
                self.assertEqual(setup.choose("Model", ["Flash", "Qwen", "Manual"]), selected)
                self.assertEqual(termios.tcgetattr(slave)[3] & input_flags, previous)
            with patch("os.read", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                setup.choose("Model", ["Flash", "Qwen"])
            self.assertEqual(termios.tcgetattr(slave)[3] & input_flags, previous)
        self.assertIn("› Manual", self.log.getvalue())

    def test_verification_spinner_clears_on_error(self):
        second_frame = threading.Event()

        class Terminal(StringIO):
            def isatty(self):
                return True

            def write(self, text):
                if "⠙" in text:
                    second_frame.set()
                return super().write(text)

        terminal = Terminal()
        with redirect_stdout(terminal), patch.dict(os.environ, TERM="xterm"):
            with self.assertRaisesRegex(ValueError, "invalid experts"), setup.checking_experts():
                self.assertTrue(second_frame.wait(2))
                raise ValueError("invalid experts")
        self.assertTrue(terminal.getvalue().endswith("\r\033[K"))

    def fixture(self, index):
        preset = setup.PRESETS[index]
        split = len(preset.files) == 2
        base = setup.FLASH if split else Path(preset.files[0][0]).stem
        writer = GGUFWriter(self.source / f"{base}.gguf", preset.arch, split_max_tensors=1 if split else 0)
        writer.add_uint32(preset.arch + ".expert_count", 2)
        writer.add_uint32(preset.arch + ".expert_used_count", 1)
        data = np.arange(2048, dtype=np.float32).reshape(2, 32, 32)
        for layer in range(2):
            writer.add_tensor(f"blk.{layer}.ffn_gate_exps.weight", data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        files = tuple((remote, (self.source / Path(remote).name).stat().st_size) for remote, _ in preset.files)
        fixture = replace(preset, files=files, expert_bytes=data.nbytes * 2)
        presets = list(setup.PRESETS)
        presets[index] = fixture
        self.mock("onboarding.PRESETS", presets)
        return fixture

    def test_offline_split_missing_then_separate_shard(self):
        preset = self.fixture(0)
        first, second = [self.source / Path(remote).name for remote, _ in preset.files]
        moved = self.home / second.name
        second.rename(moved)
        self.mock("onboarding.can_resolve", return_value=False)
        self.mock("onboarding.download", side_effect=AssertionError("offline"))
        with patch("onboarding.choose", return_value=0), patch("builtins.input", side_effect=[shlex.quote(str(first)), str(moved)]):
            self.assertTrue(setup.wizard(self.config, False))
        args = json.loads(self.config.read_text())["models"][preset.alias]["args"]
        self.assertTrue(Path(args["model"]).samefile(first))
        self.assertTrue(Path(args["model"]).with_name(second.name).samefile(moved))
        self.assertIn("Still needed: " + second.name, self.log.getvalue())

    def test_adjacent_cache_links_preserve_names(self):
        preset = self.fixture(0)
        snapshot = self.home / "snapshot"
        snapshot.mkdir()
        for i, (remote, _) in enumerate(preset.files):
            original = self.source / Path(remote).name
            blob = original.rename(self.source / str(i))
            (snapshot / original.name).symlink_to(blob)
        with patch("builtins.input", return_value=str(snapshot / Path(preset.files[1][0]).name)):
            first = setup.local_files(preset, self.home)
        dest = setup.prepare(preset, first, self.home)
        self.assertTrue((dest / "manifest.json").exists())
        (dest / "layer_000.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "unexpected sizes"):
            setup.prepare(preset, first, self.home)

    def test_complete_managed_qwen_skips_source_and_download(self):
        preset = self.fixture(1)
        managed = self.home / ".moeshift/models" / preset.alias
        managed.mkdir(parents=True)
        name = preset.files[0][0]
        shutil.copyfile(self.source / name, managed / name)
        self.mock("onboarding.can_resolve", return_value=True)
        self.mock("onboarding.download", side_effect=AssertionError("unnecessary download"))
        with patch("onboarding.choose", return_value=1):
            self.assertTrue(setup.wizard(self.config, False))
        config = json.loads(self.config.read_text())
        self.assertEqual(list(config["models"]), [preset.alias])
        args = config["models"][preset.alias]["args"]
        self.assertEqual(args["model"], str(managed / name))
        self.assertEqual(args["moe-experts"], str(managed / "experts"))

    def test_wrong_preset_and_space(self):
        preset = self.fixture(1)
        path = self.source / preset.files[0][0]
        with self.assertRaisesRegex(ValueError, "architecture"):
            setup.validate_shard(path, replace(preset, arch="wrong"), 0)
        with patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(10, 8, 2)):
            with self.assertRaisesRegex(ValueError, "free space"):
                setup.prepare(preset, path, self.home)
        self.assertFalse((self.home / "experts").exists())

    def test_manual_and_repair(self):
        self.mock("onboarding.can_resolve", return_value=False)
        self.config.write_bytes(b"broken")
        with patch("onboarding.choose", return_value=0), self.assertRaises(SystemExit):
            setup.wizard(self.config, True)
        self.assertEqual(self.config.read_bytes(), b"broken")
        with patch("onboarding.choose", side_effect=[1, 2]):
            self.assertFalse(setup.wizard(self.config, True))
        self.assertEqual(json.loads(self.config.read_text()), json.loads(setup.TEMPLATE.read_text()))
        self.assertEqual(next(self.home.glob("config.json.backup-*")).read_bytes(), b"broken")
        self.assertIn(str(self.config), self.log.getvalue())
        self.assertIn("https://github.com/davide-troiani/moeshift/blob/main/configs/README.md", self.log.getvalue())

    def test_download_resume_and_ignored_range(self):
        target = self.home / "model.gguf"
        partial = target.with_suffix(".gguf.part")
        for status in (206, 200):
            partial.write_bytes(b"abc")
            response = BytesIO(b"def" if status == 206 else b"abcdef")
            response.status = status
            response.headers = {"Content-Range": "bytes 3-5/6"}
            with patch("onboarding.urlopen", return_value=response) as request:
                setup.direct_download("https://example.org/model", target, 6)
            self.assertEqual(request.call_args.args[0].headers["Range"], "bytes=3-")
            self.assertEqual(target.read_bytes(), b"abcdef")
            self.assertIn("ETA", self.log.getvalue())

    def test_hf_download_and_direct_fallback(self):
        preset = self.fixture(0)
        managed = self.home / "library"
        managed.mkdir()
        fake = self.home / "hf"
        fake.write_text(f"""#!{sys.executable}
import shutil, sys
from pathlib import Path
remote = Path(sys.argv[3])
if remote.name.endswith("00002-of-00002.gguf"):
    sys.exit(1)
out = Path(sys.argv[-1]) / remote
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(Path({str(self.source)!r}) / remote.name, out)
print("reconstructing file: 50%|###| 4.10kB / 8.20kB", flush=True)
""")
        fake.chmod(0o755)
        body = (self.source / Path(preset.files[1][0]).name).read_bytes()
        response = BytesIO(body)
        response.status = 200
        with patch("shutil.which", return_value=str(fake)), \
             patch("onboarding.urlopen", return_value=response), patch("onboarding.choose", return_value=0), \
             patch("onboarding.transfer_status", wraps=setup.transfer_status) as progress:
            setup.download(preset, managed)
        self.assertTrue(any(call.args[0] == 4100 for call in progress.call_args_list))
        self.assertIn("ETA", self.log.getvalue())
        for remote, _ in preset.files:
            self.assertEqual((managed / Path(remote).name).read_bytes(), (self.source / Path(remote).name).read_bytes())

    def test_temporary_hf_install_failure_cleanup(self):
        preset = self.fixture(1)
        dest = self.home / "library"
        dest.mkdir()
        payload = next(self.source.glob("*.gguf")).read_bytes()
        response = BytesIO(payload)
        response.status = 200
        with patch("shutil.which", return_value=None), patch("onboarding.choose", side_effect=[0, 0]), \
             patch("subprocess.run", side_effect=[None, setup.subprocess.CalledProcessError(1, "pip")]) as run, \
             patch("onboarding.urlopen", return_value=response):
            setup.download(preset, dest)
        temporary = Path(run.call_args_list[0].args[0][-1])
        self.assertFalse(temporary.exists())
        self.assertEqual((dest / preset.files[0][0]).read_bytes(), payload)

    def test_interrupted_transfer_keeps_partial(self):
        target = self.home / "model.gguf"
        response = BytesIO(b"ab")
        response.status = 200
        with patch("onboarding.urlopen", return_value=response), self.assertRaisesRegex(ValueError, "interrupted"):
            setup.direct_download("https://example.org/model", target, 4)
        self.assertFalse(target.exists())
        self.assertEqual(target.with_suffix(".gguf.part").read_bytes(), b"ab")


if __name__ == "__main__":
    unittest.main()
