"""The browser runner must never confuse a stop or an old click with permission to release."""
import os
from pathlib import Path
import queue
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'act_shadow'))
from trial_operator import ConsoleOperator


class ConsoleInputs(unittest.TestCase):
    def setUp(self):
        rd, wr = os.pipe()
        self.input = os.fdopen(wr,'w',buffering=1)
        self.events = queue.Queue()
        self.op = ConsoleOperator(os.fdopen(rd),self.events.put)
        self.addCleanup(self.input.close)

    def test_distinct_gates(self):
        self.op.receive('PING')
        self.op.receive('START')
        self.op.receive('RELEASE')
        def experiment():
            if self.op.confirm_start():
                self.op.confirm_support()
        thread = threading.Thread(target=experiment,daemon=True)
        thread.start()
        self.assertEqual(self.events.get(timeout=1),'TRIAL_PHASE ready')
        self.assertFalse(self.op.started)
        self.op.receive('START')
        self.assertEqual(self.events.get(timeout=1),'TRIAL_PHASE arming')
        self.assertEqual(self.events.get(timeout=1),'TRIAL_PHASE holding')
        self.op.receive('STOP')
        self.assertTrue(self.op.stop_requested())
        self.assertFalse(self.op.supported)
        self.input.write('RELEASE\n')
        thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_closed_link_and_stale_lease(self):
        self.op.receive('PING')
        self.op.last_beat -= 10
        self.assertTrue(self.op.stop_requested())
        self.assertFalse(self.op.confirm_start())
        self.input.close()
        with self.assertRaises(EOFError):
            self.op.confirm_support()
        self.assertFalse(self.op.supported)


if __name__ == '__main__':
    unittest.main()
