"""Interactive first-run setup for the installed MoEShift server."""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import textwrap
import tty
import time
from collections import deque
from contextlib import contextmanager
from itertools import cycle
from dataclasses import dataclass
from http.client import HTTPException
from urllib.request import Request, urlopen

import experts


@dataclass(frozen=True)
class Preset:
    alias: str
    quantization: str
    repo: str
    arch: str
    files: tuple[tuple[str, int], ...]
    expert_bytes: int

    total_gb: int


TEMPLATE = Path(__file__).resolve().parents[2] / "moeshift.json"
FLASH = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
PRESETS = (
    Preset("qwen-3.8-flash-next-rco-gsq-iq3-xxs", "IQ3_XXS",
           "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "qwen4exp",
           ((f"IQ3_XXS/{FLASH}-00001-of-00002.gguf", 47_039_860_096),
            (f"IQ3_XXS/{FLASH}-00002-of-00002.gguf", 28_800_138_432)), 42_912_972_800, 120),
    Preset("qwen-3.6-35b-a3b-q6-xl", "UD-Q6_K_XL",
           "unsloth/Qwen3.6-35B-A3B-GGUF", "qwen35moe",
           (("Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf", 31_843_777_504),), 29_154_607_104, 61),
)


def heading(text):
    width = max(12, min(66, shutil.get_terminal_size().columns - 2))
    border = "─" * width
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.getenv("TERM") != "dumb":
        text = f"\033[1;36m{text}\033[0m"
        border = f"\033[2;36m{border}\033[0m"
    print(f"\n{border}\n{text}", flush=True)


def choose(prompt, choices):
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    selected = 0
    rows = 0
    ansi = sys.stdout.isatty() and os.getenv("TERM") != "dumb"
    print(f"{prompt}  (↑/↓ to move, Enter to select)")
    try:
        tty.setcbreak(fd, termios.TCSANOW)
        while True:
            if rows and ansi:
                print(f"\033[{rows}A\r\033[J", end="")
            width = max(5, shutil.get_terminal_size().columns - 1)
            lines = []
            for index, label in enumerate(choices):
                prefix = "  › " if index == selected else "    "
                option = textwrap.wrap(label, width=width, initial_indent=prefix, subsequent_indent="    ")
                if index == selected and ansi and "NO_COLOR" not in os.environ:
                    option = [f"\033[1;36m{line}\033[0m" for line in option]
                lines.extend(option)
            print("\n".join(lines), flush=True)
            rows = len(lines)

            prefix = b""
            while True:
                key = os.read(fd, 1)
                if key in (b"", b"\x04"):
                    raise EOFError
                if key in (b"\r", b"\n"):
                    return selected
                if prefix in (b"\x1b[", b"\x1bO") and key in (b"A", b"B"):
                    selected = (selected + (1 if key == b"B" else -1)) % len(choices)
                    break
                if key == b"\x1b":
                    prefix = key
                elif prefix == b"\x1b" and key in (b"[", b"O"):
                    prefix += key
                else:
                    prefix = b""
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, previous)


def can_resolve():
    # getaddrinfo() has no portable deadline. Bound the system resolver in a child.
    try:
        result = subprocess.run([sys.executable, "-c",
                                 "import socket; socket.getaddrinfo('huggingface.co', 443)"],
                                stderr=subprocess.DEVNULL, timeout=4)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def check_space(folder, needed):
    free = shutil.disk_usage(folder).free
    if needed > free:
        raise ValueError(f"{folder} needs {needed / 1e9:.2f} GB of free space; {free / 1e9:.2f} GB available.")


def validate_shard(path, preset, index):
    remote, size = preset.files[index]
    if path.stat().st_size != size:
        raise ValueError(f"Expected {Path(remote).name} ({size:,} bytes).")
    try:
        reader = experts.GGUFReader(path)
        if index == 0 and experts.scalar(reader, "general.architecture") != preset.arch:
            raise ValueError("unexpected model architecture")
        if (experts.scalar(reader, "split.count", 1) != len(preset.files)
                or experts.scalar(reader, "split.no", 0) != index):
            raise ValueError("unexpected GGUF shard metadata")
    except (ValueError, IndexError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid GGUF {path}: {exc}") from exc


def pasted_paths(line):
    # Finder escapes spaces when dragging. A literal unquoted path also works.
    if not line.strip():
        return []
    literal = Path(line.strip()).expanduser()
    if literal.exists():
        return [literal.absolute()]
    return [Path(word).expanduser().absolute() for word in shlex.split(line)]


def local_files(preset, folder, found=None):
    expected = [Path(remote).name for remote, _ in preset.files]
    found = dict(found or {})
    print("Drop a GGUF or its folder here, or paste its path. Press Return without a path to go back.")
    while len(found) != len(expected):
        try:
            line = input("Path › ")
            if not line.strip():
                return None
            paths = pasted_paths(line)
            for path in paths:
                if not path.is_dir() and path.name not in expected:
                    raise ValueError(f"Expected {' or '.join(expected)}.")
                parent = path if path.is_dir() else path.parent
                for i, name in enumerate(expected):
                    candidate = parent / name
                    if path.is_dir() and not candidate.is_file():
                        candidate = parent / preset.files[i][0]
                    if name not in found and candidate.is_file():
                        validate_shard(candidate, preset, i)
                        found[name] = candidate
        except (OSError, ValueError) as exc:
            print(exc)
        missing = [name for name in expected if name not in found]
        if missing:
            print("Still needed: " + ", ".join(missing))
    paths = [found[name] for name in expected]
    print("Using your GGUF files in place. Keep these paths available:")
    for path in paths:
        print(f"  {path}")
    if len({path.parent for path in paths}) == 1:
        return paths[0]
    # The native loader expects adjacent shard names. Give it links, not copies.
    digest = hashlib.sha256(os.fsencode("\0".join(map(str, paths)))).hexdigest()[:16]
    links = folder / f"sources-{digest}"
    links.mkdir(exist_ok=True)
    for path in paths:
        link = links / path.name
        if not link.is_symlink():
            link.symlink_to(path)
        elif link.readlink() != path:
            raise ValueError(f"Conflicting shard link: {link}")
    return links / paths[0].name


def transfer_status(done, size, rate):
    fraction = done / size
    columns = shutil.get_terminal_size().columns
    width = max(4, min(22, columns - 53))
    filled = min(width, int(width * fraction))
    eta = time.strftime("%H:%M:%S", time.gmtime((size - done) / rate)) if rate > 0 else "…"
    bar = "━" * filled + "─" * (width - filled)
    scale, unit = (1e9, "GB") if size >= 1e9 else (1e6, "MB")
    counts = f"{done / scale:.2f}/{size / scale:.2f} {unit}  " if columns >= 65 else ""
    print(f"\r{bar} {fraction:5.1%}  {counts}{rate / 1e6:.1f} MB/s  ETA {eta}   ", end="", flush=True)


def direct_download(url, target, size, check=None):
    partial = target.with_suffix(".gguf.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset != size:
        headers = {"Range": f"bytes={offset}-"} if 0 < offset < size else {}
        with urlopen(Request(url, headers=headers), timeout=45) as response:
            if response.status == 206:
                if not headers or response.headers.get("Content-Range") != f"bytes {offset}-{size - 1}/{size}":
                    raise ValueError("Unexpected download range; partial file preserved.")
            elif response.status == 200:
                offset = 0  # The server may ignore Range.
            else:
                raise ValueError(f"Unexpected HTTP status {response.status}")
            done = offset
            start = last = time.monotonic()
            with partial.open("ab" if offset else "wb") as output:
                while block := response.read(1 << 20):
                    if done + len(block) > size:
                        raise ValueError("Download exceeds the expected GGUF size.")
                    output.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if now - last >= 0.25 or done == size:
                        transfer_status(done, size, (done - offset) / max(now - start, 0.001))
                        last = now
            print()
    if partial.stat().st_size != size:
        raise ValueError("Download interrupted. Run moeshift again to resume.")
    if check is not None:
        try:
            check(partial)
        except ValueError:
            partial.unlink()  # A complete but invalid transfer cannot be resumed.
            raise
    partial.rename(target)


def hf_download(hf, repo, remote, stage, size):
    # hf-xet reports reconstructed bytes and transfer speed on separate terminal rows.
    # Use completed file bytes for the ETA, independent of compression or cache hits.
    master, slave = pty.openpty()
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "0", "HF_XET_CACHE": str(stage / ".cache/xet")}
    recent = deque(maxlen=5)
    started = time.monotonic()
    baseline = None
    done = 0
    transfer_status(0, size, 0)
    with os.fdopen(master, encoding="utf-8", errors="replace") as output, os.fdopen(slave, "wb") as terminal:
        # A fresh PTY has zero columns. Give hf room to render byte counts.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))
        with subprocess.Popen([hf, "download", repo, remote, "--local-dir", str(stage)],
                              stdout=terminal, stderr=terminal, env=env) as child:
            terminal.close()
            try:
                for line in output:
                    line = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", line).strip()
                    percent = re.search(r"(\d{1,3}(?:\.\d+)?)%\s*\|", line)
                    if not percent:
                        if line:
                            recent.append(line)
                        continue
                    count = int(size * float(percent[1]) / 100)
                    units = re.search(r"([\d.]+)\s*([kMGT]?)B?\s*/\s*[\d.]+\s*[kMGT]?B", line, re.I)
                    if units:
                        count = int(float(units[1]) * 1000 ** " KMGT".index(units[2].upper() or " "))
                    done = max(done, min(count, size))
                    now = time.monotonic()
                    if baseline is None:
                        baseline, started = done, now
                    transfer_status(done, size, (done - baseline) / max(now - started, 0.001))
            except OSError as exc:
                if exc.errno != errno.EIO:  # Darwin may report EIO instead of EOF on a closed PTY.
                    raise
            finally:
                if sys.exc_info()[0] is not None and child.poll() is None:
                    child.terminate()
            status = child.wait()
    if status == 0:
        transfer_status(size, size, (size - (baseline or 0)) / max(time.monotonic() - started, 0.001))
    print()
    if status:
        print("\n".join(recent))
    return status == 0


def download(preset, folder):
    pending = [(i, remote, size) for i, (remote, size) in enumerate(preset.files)
               if not (folder / Path(remote).name).exists()]
    parts = {i: (folder / Path(remote).name).with_suffix(".gguf.part") for i, remote, _ in pending}
    hf = shutil.which("hf")
    # Install outside the Homebrew environment, clean up on both success and failure.
    with tempfile.TemporaryDirectory(prefix="moeshift-hf-") as tmp:
        if not hf and any(not part.exists() for part in parts.values()):
            if choose("Download method", ["Use accelerated hf (temporary install)", "Direct HTTPS"]) == 0:
                try:
                    print("Installing hf for this download…", flush=True)
                    subprocess.run([sys.executable, "-m", "venv", tmp], check=True)
                    subprocess.run([f"{tmp}/bin/python", "-m", "pip", "install",
                                    "--quiet", "--disable-pip-version-check", "--no-cache-dir", "huggingface_hub"], check=True)
                    hf = f"{tmp}/bin/hf"
                except subprocess.CalledProcessError:
                    if choose("Could not install hf.", ["Continue with HTTPS", "Exit"]) == 1:
                        raise SystemExit(1)
        needed = 0 if (folder / "experts").exists() else preset.expert_bytes
        for i, remote, size in pending:
            stage = folder / ".hf" / Path(remote).name
            if parts[i].exists():
                retained = parts[i].stat().st_size
            elif hf and stage.exists():
                # A private hf local-dir per shard isolates its resumable transfer.
                retained = max((p.stat().st_blocks * 512 for p in stage.rglob("*.incomplete")), default=0)
                if (stage / remote).is_file():
                    retained = (stage / remote).stat().st_size
            else:
                retained = 0
            needed += size - retained if retained <= size else size
        check_space(folder, needed)
        for i, remote, size in pending:
            target = folder / Path(remote).name
            print(f"Downloading {target.name}", flush=True)
            if hf and not parts[i].exists():
                stage = folder / ".hf" / target.name
                marker = stage / ".moeshift-download"
                if not stage.exists():
                    stage.mkdir(parents=True)
                    marker.touch()
                if stage.is_symlink() or not marker.is_file():
                    raise ValueError(f"Cannot use an existing unmanaged folder for hf: {stage}")
                if hf_download(hf, preset.repo, remote, stage, size):
                    downloaded = stage / remote
                    validate_shard(downloaded, preset, i)
                    downloaded.rename(target)
                    shutil.rmtree(stage)
                    continue
                if choose("hf could not finish.", ["Continue with HTTPS", "Exit and retry later"]) == 1:
                    raise SystemExit(1)
                hf = None
                # hf and HTTPS partials have different formats. Retain the hf cache for a later retry.
                check_space(folder, sum(s for _, r, s in pending if not (folder / Path(r).name).exists())
                            + (0 if (folder / "experts").exists() else preset.expert_bytes))
            direct_download(f"https://huggingface.co/{preset.repo}/resolve/main/{remote}?download=true",
                            target, size, lambda part: validate_shard(part, preset, i))


@contextmanager
def checking_experts():
    label = "Checking expert files…"
    if not sys.stdout.isatty() or os.getenv("TERM") == "dumb":
        print(label, flush=True)
        yield
        return

    stopped = threading.Event()

    def animate():
        for frame in cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            print(f"\r{frame}  {label}", end="", flush=True)
            if stopped.wait(0.1):
                break

    animation = threading.Thread(target=animate)
    animation.start()
    try:
        yield
    finally:
        stopped.set()
        animation.join()
        print("\r\033[K", end="", flush=True)


def prepare(preset, model, folder):
    dest = folder / "experts"
    if not dest.exists():
        # Ctrl-C cleans up immediately. A killed process leaves only its marked staging folder.
        for old in folder.glob(".extract-*"):
            if old.is_dir() and not old.is_symlink() and (old / ".moeshift-extraction").is_file():
                shutil.rmtree(old)
        check_space(folder, preset.expert_bytes)
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=folder) as tmp:
            (Path(tmp) / ".moeshift-extraction").touch()
            staging = Path(tmp) / "experts"
            experts.extract(argparse.Namespace(model=model, out_dir=staging, force=False, layers=None), verbose=False,
                            on_layer=lambda done, total: print(f"\rPreparing expert layers  {done}/{total}",
                                                               end="", flush=True))
            print()
            with checking_experts():
                verify(preset, model, staging)
            staging.rename(dest)
    else:
        with checking_experts():
            verify(preset, model, dest)
    return dest


def verify(preset, model, folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    paths = experts.resolve_model_paths(model)
    if manifest["source"]["model_files"] != list(map(str, paths)):
        raise ValueError(f"Existing experts in {folder} belong to different GGUF paths.")
    experts.verify(argparse.Namespace(model=model, experts=folder, layers=None, metadata_only=True), verbose=False)
    sizes = {}
    for entry in manifest["entries"]:
        name = f"layer_{entry['layer']:03d}.bin"
        if entry["repacked_file"] != name:
            raise ValueError(f"Unexpected expert file {entry['repacked_file']}")
        sizes[name] = sizes.get(name, 0) + entry["exact_byte_length"]
    if sum(sizes.values()) != preset.expert_bytes or any((folder / n).stat().st_size != s for n, s in sizes.items()):
        raise ValueError("Expert files have unexpected sizes.")


def save_config(path, config, original):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".config-", delete=False) as out:
        staging = Path(out.name)
    try:
        staging.write_text(json.dumps(config, indent=2) + "\n")
        if original is None:
            os.link(staging, path)  # First launch never clobbers an existing config.
        else:
            if path.read_bytes() != original:
                raise ValueError("Configuration changed during setup; leaving it untouched.")
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix="config.json.backup-", delete=False) as backup:
                backup.write(original)
            print(f"Original configuration saved to {backup.name}")
            os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def select_source(preset, folder, online):
    heading("02 / 03    Model files")
    print(f"Library   {folder}\nSource    https://huggingface.co/{preset.repo}")
    managed = [folder / Path(remote).name for remote, _ in preset.files]
    found = {}
    conflicts = []
    for i, path in enumerate(managed):
        if os.path.lexists(path):
            try:
                validate_shard(path, preset, i)
                found[path.name] = path
            except (ValueError, OSError) as exc:
                conflicts.append(path)
                print(f"Cannot reuse {path}: {exc}")
    if len(found) == len(managed):
        print("Found complete GGUF files in your library.")
        return managed[0]
    if conflicts:
        print("Local files can still be used. Move the conflicting library files to enable downloads.")
    if online and not conflicts:
        source = choose("Where are your model files?",
                        ["Use files on this Mac", "Download from Hugging Face", "Back to model selection"])
        if source == 2:
            return None
        if source == 1:
            download(preset, folder)
            return managed[0]
    return local_files(preset, folder, found)


def wizard(config_path, repair):
    heading("◆  MoEShift   /   First-run setup")
    original = None
    if repair:
        if choose(f"Reconfigure {config_path}? The original will be backed up.", ["Keep it and exit", "Reconfigure"]) == 0:
            raise SystemExit(1)
        original = config_path.read_bytes()
    elif config_path.exists():
        raise ValueError("A configuration already exists. Run moeshift again.")
    config = json.loads(TEMPLATE.read_text())
    while True:
        heading("01 / 03    Choose your model")
        online = can_resolve()
        if not online:
            print("Cannot resolve huggingface.co. Local files work offline. Check your DNS, VPN or firewall.")
        choices = [f"{p.repo} · {p.quantization} · requires {p.total_gb} GB on disk" for p in PRESETS]
        choices.append("Manual configuration")
        if not online:
            choices.append("Retry connection")
        selection = choose("Choose your setup", choices)
        if selection == len(PRESETS) + 1:
            continue
        if selection == len(PRESETS):
            save_config(config_path, config, original)
            heading("✓  Configuration created")
            print(f"Configuration file: {config_path}")
            print("Manual setup guide: https://github.com/davide-troiani/moeshift/blob/main/configs/README.md")
            print("Set model and moe-experts in your profile, save the file, then run moeshift.")
            return False
        preset = PRESETS[selection]
        folder = Path.home() / ".moeshift/models" / preset.alias
        folder.mkdir(parents=True, exist_ok=True)
        model = select_source(preset, folder, online)
        if model is not None:
            break
    heading("03 / 03    Prepare and check experts")
    dest = prepare(preset, model, folder)
    profile = config["models"][preset.alias]
    profile["args"].update({"model": str(model), "moe-experts": str(dest)})
    config["models"] = {preset.alias: profile}
    save_config(config_path, config, original)
    print(f"Configuration saved to {config_path}\nStarting the local server…", flush=True)
    return True


def main():
    path, binary, action, *overrides = sys.argv[1:]
    config = Path(path).absolute()
    config.parent.mkdir(parents=True, exist_ok=True)
    # One library per HOME even if two launches use different XDG_CONFIG_HOME values.
    library = Path.home() / ".moeshift"
    library.mkdir(exist_ok=True)
    with (library / ".setup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another MoEShift setup is already running.")
        start = wizard(config, action == "repair")
    if start:
        os.execv(binary, [binary, "--config", str(config), *overrides])


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nSetup cancelled. Run moeshift to continue.")
    except (OSError, HTTPException, ValueError, KeyError, subprocess.SubprocessError) as exc:
        sys.exit(f"moeshift: {exc}")
