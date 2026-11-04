#!/usr/bin/env python3
"""Integration checks for the native LES router, using a lightweight fake worker."""

import concurrent.futures
import http.client
import http.server
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fake_worker():
    args = {}
    values = iter([arg for arg in sys.argv[2:] if arg != "--worker"])
    for key in values:
        args[key] = True if key == "--no-mmap" else next(values)
    if args["--model"].endswith("fail.gguf"):
        return 1
    log_path = os.environ["MOESHIFT_ROUTER_TEST_LOG"]

    def log(event):
        with open(log_path, "a") as output:
            output.write(json.dumps({"event": event, "pid": os.getpid(), "time": time.monotonic()}) + "\n")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log("request_start")
            self.send_response(200)
            if body.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"first":true}\n\n')
                self.wfile.flush()
                time.sleep(0.6)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "pid": os.getpid(), "args": args, "path": self.path,
                    "env": {key: value for key, value in os.environ.items() if key.startswith(("LLAMA_", "GGML_", "LES_", "MB_"))},
                }).encode())
            log("request_end")

    server = http.server.HTTPServer(("127.0.0.1", int(args["--port"])), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    log("start")
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
        log("stop")
    return 0


class RouterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="les router ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = self.root / "events.jsonl"
        worker = self.root / "fake worker"
        worker.write_text(
            f"#!{sys.executable}\nimport runpy, sys\nsys.argv.insert(1, '--fake-worker')\n"
            f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n"
        )
        worker.chmod(0o755)
        self.port = free_port()
        example = json.loads((Path(__file__).resolve().parents[1] / "configs/moeshift.json").read_text())
        profile = example["models"]["qwen-3.8-flash-next-rco-gsq-iq3-xxs"]
        self.config = {"binary": str(worker), "host": "127.0.0.1", "port": self.port, "models": {}}
        for name, ctx in (("a", 2048), ("b", 4096), ("fail", 2048)):
            value = json.loads(json.dumps(profile))
            value["args"].update({"alias": name, "model": f"{name}.gguf", "ctx-size": ctx})
            value["env"]["LLAMA_MOE_DECODE_WINDOW"] = ctx // 1024
            value["env"]["LLAMA_MOE_PROTECTED_SLOTS"] = 0
            self.config["models"][name] = value
        self.path = self.root / "server.json"
        self.path.write_text(json.dumps(self.config))
        self.log = open(self.root / "router.log", "w+")
        self.addCleanup(self.log.close)
        env = dict(os.environ, MOESHIFT_ROUTER_TEST_LOG=str(self.events), LLAMA_MOE_F_NOCACHE="1",
                   LLAMA_MOE_PROTECTED_POLICY="ignored", LLAMA_ARG_THREADS="999",
                   LLAMA_LES_DECODE_BANK="511", LLAMA_KV_GROW="0",
                   LLAMA_GRAPH_REUSE_DISABLE="1", LLAMA_QWEN_SEPARATE_PREFILL_BANKS="1",
                   GGML_METAL_FUSION_DISABLE="1", LES_BLOCK_DIAG="1", MB_DUMP_INP_EMBD="1")
        self.proc = subprocess.Popen([BINARY, "--config", str(self.path)], env=env,
                                     stdout=self.log, stderr=self.log)
        self.addCleanup(self.stop)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if self.request("GET", "/health")[0] == 200:
                    return
            except OSError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.05)
        self.log.seek(0)
        self.fail("Router failed to start: " + self.log.read())

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                raise

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request(method, path, None if body is None else json.dumps(body),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def infer(self, name, path="/v1/chat/completions"):
        status, body = self.request("POST", path, {"model": name, "messages": []})
        self.assertEqual(status, 200, body)
        return body

    def test_runtime_environment_cannot_override_profile(self):
        worker = self.infer("a")
        for key in ("LLAMA_LES_DECODE_BANK", "LLAMA_KV_GROW", "LLAMA_GRAPH_REUSE_DISABLE",
                    "LLAMA_QWEN_SEPARATE_PREFILL_BANKS", "GGML_METAL_FUSION_DISABLE",
                    "LES_BLOCK_DIAG", "MB_DUMP_INP_EMBD"):
            self.assertNotIn(key, worker["env"])
        self.assertEqual(worker["env"]["LLAMA_MOE_DECODE_WINDOW"], "2")

    def test_reject_parallel_and_external_host(self):
        for args in (("--parallel", "2"), ("--host", "0.0.0.0")):
            result = subprocess.run([BINARY, "--config", str(self.path), *args],
                                    capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)

    def test_worker_rejects_speculative_options(self):
        for option in ("--spec-type", "--model-draft", "--parallel"):
            result = subprocess.run([BINARY, "--worker", option, "none"],
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"Unsupported worker option", result.stderr)

    def test_cli_overrides(self):
        self.stop()
        self.proc = subprocess.Popen(
            [BINARY, "--config", str(self.path), "--ctx-size", "8192",
             "--moe-protected-slots", "23", "--moe-decode-window", "12"],
            env=dict(os.environ, MOESHIFT_ROUTER_TEST_LOG=str(self.events)),
            stdout=self.log, stderr=self.log)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if self.request("GET", "/health")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.05)
        response = self.infer("a")
        self.assertEqual(response["args"]["--ctx-size"], "8192")
        self.assertEqual(response["env"]["LLAMA_MOE_PROTECTED_SLOTS"], "23")

    def test_routing_environment_reuse_failure_recovery_and_cleanup(self):
        self.assertFalse(self.events.exists(), "Models must load lazily")
        self.assertEqual([v["id"] for v in self.request("GET", "/v1/models")[1]["data"]], ["a", "b", "fail"])
        self.assertEqual(self.request("POST", "/v1/chat/completions", {})[0], 400)
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"model": "unknown"})[0], 404)
        a = self.infer("a")
        self.assertEqual(a["pid"], self.infer("a")["pid"])
        self.assertEqual(self.request("POST", "/v1/completions", {"model": "unknown"})[0], 404)
        self.assertEqual(a["pid"], self.infer("a")["pid"])
        self.assertEqual(a["env"]["LLAMA_MOE_F_NOCACHE"], "0")
        self.assertNotIn("LLAMA_ARG_THREADS", a["env"])
        self.assertNotIn("LLAMA_MOE_PROTECTED_POLICY", a["env"])
        self.assertNotIn("--parallel", a["args"])
        self.assertIs(a["args"]["--no-mmap"], True)
        self.assertEqual(a["args"]["--ctx-size"], "2048")
        b = self.infer("b", "/v1/completions")
        self.assertNotEqual(a["pid"], b["pid"])
        self.assertEqual(b["args"]["--ctx-size"], "4096")
        self.assertEqual(b["path"], "/v1/completions")
        self.assertEqual(b["env"]["LLAMA_MOE_DECODE_WINDOW"], "4")
        self.assertEqual(self.request("POST", "/v1/completions", {"model": "fail"})[0], 503)
        recovered = self.infer("a")
        os.kill(recovered["pid"], signal.SIGKILL)
        time.sleep(0.1)
        restarted = self.infer("a")
        self.assertNotEqual(recovered["pid"], restarted["pid"])
        self.stop()
        self.assertEqual(self.proc.returncode, 0)
        with self.assertRaises(ProcessLookupError):
            os.kill(restarted["pid"], 0)

    def test_stream_finishes_before_switch(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        self.addCleanup(conn.close)
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": "a", "stream": True}),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(response.getheader("Content-Type").startswith("text/event-stream"))
        self.assertEqual(response.readline(), b'data: {"first":true}\n')
        self.assertEqual(response.readline(), b"\n")
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(self.infer, "b")
            time.sleep(0.1)
            self.assertFalse(pending.done())
            self.assertIn(b"data: [DONE]", response.read())
            b = pending.result(timeout=15)
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        first_end = next(i for i, event in enumerate(events) if event["event"] == "request_end")
        next_start = next(i for i, event in enumerate(events) if event["event"] == "start" and event["pid"] == b["pid"])
        self.assertLess(first_end, next_start)

    def test_shutdown_during_stream(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        self.addCleanup(conn.close)
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": "a", "stream": True}),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.readline(), b'data: {"first":true}\n')
        events = [json.loads(line) for line in self.events.read_text().splitlines()]
        pid = next(event["pid"] for event in events if event["event"] == "start")
        self.stop()
        self.assertEqual(self.proc.returncode, 0)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_logs_identify_moeshift(self):
        self.infer("a")
        self.stop()
        self.log.seek(0)
        output = self.log.read()
        self.assertIn("moeshift", output)
        self.assertIn(f"moeshift: Ready. Listening on http://127.0.0.1:{self.port}/v1 with a b fail", output)
        self.assertNotIn("moeshift models:", output)
        self.assertNotIn("LES router", output)
        for line in output.splitlines():
            if line.strip(): self.assertIn("moeshift", line)

    def test_reject_removed_configuration(self):
        self.stop()
        for category, key, value in (("args", "parallel", 1), ("env", "LLAMA_MOE_F_NOCACHE", 0),
                                     ("env", "LLAMA_MOE_PROTECTED_POLICY", "demand-lrfu")):
            with self.subTest(key=key):
                profile = self.config["models"]["a"][category]
                profile[key] = value
                self.path.write_text(json.dumps(self.config))
                result = subprocess.run([BINARY, "--config", str(self.path)], capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                self.assertIn(key.encode(), result.stderr)
                del profile[key]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-worker":
        sys.exit(fake_worker())
    BINARY = str(Path(sys.argv.pop(1)).resolve())
    unittest.main()
