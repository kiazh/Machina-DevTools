#!/usr/bin/env python3
"""Scripted LSP server over stdio for transport/lifecycle tests (pure stdlib).

The server speaks real Content-Length framed JSON-RPC on stdin/stdout; stderr
carries diagnostics/floods. Behavior is selected with ``--mode`` and a few
numeric flags so a single script can exercise handshake ordering, slow and
out-of-order replies, crashes, oversized/garbage output, stderr floods,
server-initiated requests and refusal to exit.

Custom test channel: the server emits ``$/fake/progress`` notifications with a
``event`` param so tests can synchronize on events instead of sleeping, and it
answers a few synthetic requests:

- ``record-log`` -> received-message log, initialize params/count
- ``handshake-log`` -> received-method order
- ``env-keys`` -> sorted ``os.environ`` keys of this process
- ``fetch-responses`` -> responses this server received from the client
- ``echo``/``ping`` -> ``{"method": ..., "params": ...}``
- ``slow`` -> same echo after ``--delay`` seconds (on a thread)
- ``hang`` -> no reply
- ``boom`` -> process exits immediately with code 9
- ``error`` -> JSON-RPC error with ``params.code``/``params.message``
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

MODES = (
    "normal",
    "slow-initialize",
    "out-of-order",
    "garbage",
    "stderr-flood",
    "refuse-exit",
    "oversized",
    "server-requests",
    "crash-after-initialize",
)


def encode(message: dict) -> bytes:
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: %d\r\n\r\n%s" % (len(body), body)


class FakeServer:
    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.mode = options.mode
        self.records: list[tuple[str, str, object]] = []
        self.initialize_params: dict | None = None
        self.initialize_count = 0
        self.server_responses: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._stdin_buffer = bytearray()
        self._body_remaining: int | None = None
        self._shutdown_seen = False

    # -- output --------------------------------------------------------------

    def send(self, message: dict) -> None:
        data = encode(message)
        with self._lock:
            os.write(1, data)

    def notify(self, method: str, params: object = None) -> None:
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)

    def progress(self, event: str, **extra: object) -> None:
        self.notify("$/fake/progress", {"event": event, **extra})

    def record(self, kind: str, method: str, params: object) -> None:
        with self._lock:
            self.records.append((kind, method, params))

    # -- input framing -------------------------------------------------------

    def read_message(self) -> dict | None:
        while True:
            if self._body_remaining is not None:
                if len(self._stdin_buffer) < self._body_remaining:
                    chunk = os.read(0, 65536)
                    if not chunk:
                        return None
                    self._stdin_buffer.extend(chunk)
                    continue
                body = bytes(self._stdin_buffer[: self._body_remaining])
                del self._stdin_buffer[: self._body_remaining]
                self._body_remaining = None
                return json.loads(body.decode("utf-8"))
            end = self._stdin_buffer.find(b"\r\n\r\n")
            if end == -1:
                chunk = os.read(0, 65536)
                if not chunk:
                    return None
                self._stdin_buffer.extend(chunk)
                continue
            header = bytes(self._stdin_buffer[:end])
            del self._stdin_buffer[: end + 4]
            for line in header.split(b"\r\n"):
                name, _, value = line.partition(b":")
                if name.strip().lower() == b"content-length":
                    self._body_remaining = int(value.strip())
            if self._body_remaining is None:
                raise SystemExit("fake server: missing Content-Length")

    # -- dispatch ------------------------------------------------------------

    def serve(self) -> int:
        self._install_signal_handling()
        if self.options.dump_env:
            self._dump_env()
        while True:
            message = self.read_message()
            if message is None:
                return 0
            self.handle(message)

    def handle(self, message: dict) -> None:
        method = message.get("method")
        params = message.get("params")
        if method is None:
            if "id" in message:
                self.server_responses[str(message["id"])] = message
                self._maybe_report_responses_collected()
            return
        if "id" in message:
            self.record("request", method, params)
            self.handle_request(message["id"], method, params)
        else:
            self.record("notification", method, params)
            self.handle_notification(method, params)

    def handle_request(self, request_id: object, method: str, params: object) -> None:
        if method == "initialize":
            self.initialize_count += 1
            self.initialize_params = params
            if self.mode == "slow-initialize":
                time.sleep(self.options.delay)
            self.send({"jsonrpc": "2.0", "id": request_id, "result": self.initialize_result()})
            return
        if method == "shutdown":
            self._shutdown_seen = True
            if self.mode != "refuse-exit":
                self.send({"jsonrpc": "2.0", "id": request_id, "result": None})
            return
        if method == "record-log":
            self.send({"jsonrpc": "2.0", "id": request_id, "result": self.record_log()})
            return
        if method == "handshake-log":
            self.send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": [{"kind": k, "method": m} for k, m, _ in self.records],
                }
            )
            return
        if method == "env-keys":
            self.send({"jsonrpc": "2.0", "id": request_id, "result": sorted(os.environ.keys())})
            return
        if method == "fetch-responses":
            self.send({"jsonrpc": "2.0", "id": request_id, "result": dict(self.server_responses)})
            return
        if method == "slow":
            self.progress("slow-received")
            threading.Thread(
                target=self._reply_slow, args=(request_id, method, params), daemon=True
            ).start()
            return
        if method == "hang":
            self.progress("hang-received")
            return
        if method == "boom":
            self.progress("boom-received")
            os._exit(9)
        if method == "error":
            code = -32603
            text = "injected error"
            if isinstance(params, dict):
                code = int(params.get("code", code))
                text = str(params.get("message", text))
            self.send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": code, "message": text, "data": {"injected": True}},
                }
            )
            return
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"method": method, "params": params},
            }
        )

    def _reply_slow(self, request_id: object, method: str, params: object) -> None:
        time.sleep(self.options.delay)
        self.progress("slow-replied")
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"method": method, "params": params},
            }
        )

    def handle_notification(self, method: str, params: object) -> None:
        if method == "initialized":
            self.after_initialized()
            return
        if method == "exit":
            if self.mode != "refuse-exit":
                raise SystemExit(0)
            return
        if method == "window/logMessage" or method == "window/showMessage":
            return

    def after_initialized(self) -> None:
        if self.mode == "crash-after-initialize":
            os._exit(9)
        if self.mode == "garbage":
            os.write(1, b'{"jsonrpc":"2.0","id":99,"method":"no framing"}\n')
        elif self.mode == "stderr-flood":
            threading.Thread(target=self._flood_stderr, daemon=True).start()
        elif self.mode == "oversized":
            os.write(1, b"Content-Length: 999999999\r\n\r\n")
        elif self.mode == "server-requests":
            self._send_server_requests()

    # -- mode helpers --------------------------------------------------------

    def initialize_result(self) -> dict:
        return {
            "capabilities": {
                "textDocumentSync": {
                    "openClose": True,
                    "change": 1,
                    "save": {"includeText": self.options.save_include_text},
                },
                "completionProvider": {},
                "hoverProvider": True,
            },
            "serverInfo": {
                "name": "fake-lsp",
                "version": "1.0",
                "machinaInitializeCount": self.initialize_count,
            },
        }

    def record_log(self) -> dict:
        return {
            "records": [{"kind": k, "method": m, "params": p} for k, m, p in self.records],
            "initializeParams": self.initialize_params,
            "initializeCount": self.initialize_count,
        }

    def _dump_env(self) -> None:
        lines = "".join(f"ENVKEY: {key}\n" for key in sorted(os.environ))
        os.write(2, (lines + "ENVKEYS-DONE\n").encode("utf-8"))

    def _flood_stderr(self) -> None:
        try:
            os.write(2, b"\x1b[31mFLOOD-START\x1b[0m\n")
            block = b"flood-chunk-" + b"x" * 8180
            written = 0
            while written < self.options.flood_bytes:
                os.write(2, block)
                written += len(block)
            os.write(2, b"FLOOD-END\n")
        except OSError:
            return
        self.progress("flood-done")

    def _send_server_requests(self) -> None:
        self.notify(
            "window/logMessage",
            {"type": 3, "message": "FAKE-LOG\x1b[31m-RED\x07 tail"},
        )
        self.send(
            {
                "jsonrpc": "2.0",
                "id": "srv-1",
                "method": "workspace/applyEdit",
                "params": {"edit": {"changes": {}}},
            }
        )
        self.send(
            {
                "jsonrpc": "2.0",
                "id": "srv-2",
                "method": "workspace/configuration",
                "params": {"items": [{"section": "fake"}]},
            }
        )
        self.send({"jsonrpc": "2.0", "id": "srv-3", "method": "machina/unknown", "params": {}})
        self.send(
            {
                "jsonrpc": "2.0",
                "id": "srv-4",
                "method": "window/showMessageRequest",
                "params": {"type": 1, "message": "pick one"},
            }
        )
        self.send(
            {
                "jsonrpc": "2.0",
                "id": "srv-5",
                "method": "client/registerCapability",
                "params": {"registrations": []},
            }
        )
        self.progress("server-requests-sent")

    def _maybe_report_responses_collected(self) -> None:
        if self.mode == "server-requests" and len(self.server_responses) >= 5:
            self.progress("responses-collected")

    def _install_signal_handling(self) -> None:
        if self.mode == "refuse-exit":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scripted fake LSP server")
    parser.add_argument("--mode", choices=MODES, default="normal")
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--flood-bytes", type=int, default=2_000_000)
    parser.add_argument("--dump-env", action="store_true")
    parser.add_argument("--no-save-include-text", dest="save_include_text", action="store_false")
    parser.set_defaults(save_include_text=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(sys.argv[1:] if argv is None else argv)
    return FakeServer(options).serve()


if __name__ == "__main__":
    sys.exit(main())
