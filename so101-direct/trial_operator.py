"""Operator input for one supervised trial over a private SSH pipe."""
import threading
import time


class ConsoleOperator:
    timeout = 4.0

    def __init__(self, stream, emit):
        self.changed = threading.Condition()
        self.phase = "checking"
        self.emit = emit
        self.last_beat = 0.
        self.stopped = self.closed = self.started = self.supported = False
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream):
        try:
            with stream:
                for line in stream:
                    self.receive(line.strip())
        finally:
            with self.changed:
                self.closed = self.stopped = True
                self.changed.notify_all()

    def receive(self, token):
        with self.changed:
            if token == "PING":
                self.last_beat = time.monotonic()
            elif token == "STOP":
                self.stopped = True
            elif token == "START" and self.phase == "ready" and not self.stop_requested():
                self.started = True
            elif token == "RELEASE" and self.phase == "holding":
                self.supported = True
            self.changed.notify_all()

    def transition(self, phase):
        with self.changed:
            self.phase = phase
            self.emit("TRIAL_PHASE " + phase)

    def stop_requested(self):
        with self.changed:
            return self.stopped or self.closed or time.monotonic() - self.last_beat > self.timeout

    def confirm_start(self):
        with self.changed:
            self.transition("ready")
            while not self.started and not self.stop_requested():
                self.changed.wait(.25)
            if self.stop_requested():
                return False
            self.transition("arming")
            return True

    def confirm_support(self):
        with self.changed:
            self.transition("holding")
            while not self.supported and not self.closed:
                self.changed.wait()
            if not self.supported:
                raise EOFError("控制连接断开，请托稳后切断从臂电源")
