"""Run live_demo.ipynb headless (nbclient) with a dummy camera and fake motors.

Checks the decoupled notebook end to end (TASKS M0-3, M0-4, M0-6): load a
model from the dropdown, drive, make the display slow on purpose, stop, and
press the end button -- both threads must end within 1 s with the motors at 0.

Skipped when nbclient / ipykernel / ipywidgets are not installed (the robot).
"""
import json
import os
import tempfile
import unittest

import torch

# Shared test helpers (sets the import path)
from _helpers import ROOT

# nbclient and friends are PC-only test dependencies
try:
    import nbformat
    from nbclient import NotebookClient
    import ipykernel  # noqa: F401
    import ipywidgets  # noqa: F401
    # Everything needed is there
    HAVE_NB = True
# Missing on the robot: skip
except ImportError:
    # Tests below are skipped
    HAVE_NB = False

# Code appended to the notebook: drive, slow display, stop, end; prints a JSON result
SCENARIO = r'''
import json, time
torch.set_num_threads(1)
res = {}
model_dd.value = model_dd.options[0][1]
t0 = time.time()
while drive.model is None and time.time() - t0 < 60:
    time.sleep(0.05)
res['loaded'] = drive.model is not None
time.sleep(1.0)
on_drive()
time.sleep(1.5)
res['driving'] = drive.driving
res['button_driving'] = drive_btn.description
res['left'] = robot.left_motor.value
drive.periods.values.clear()
time.sleep(3.0)
res['p95_fast'] = drive.periods.percentile(95)
res['display_fps_fast'] = drive.display_stats.get('fps')
fast_render = display_worker.render
def slow_render(item, quality):
    fast_render(item, quality)
    time.sleep(0.5)
display_worker.render = slow_render
time.sleep(2.0)
drive.periods.values.clear()
time.sleep(3.0)
res['p95_slow'] = drive.periods.percentile(95)
res['display_fps_slow'] = drive.display_stats.get('fps')
res['quality_slow'] = drive.display_stats.get('quality')
res['still_driving'] = drive.driving
on_drive()
time.sleep(1.5)
res['button_after_stop'] = drive_btn.description
res['status_after_stop'] = drive_status.value
t0 = time.time()
on_end()
res['end_seconds'] = time.time() - t0
res['end_button'] = end_btn.description
res['motors'] = [robot.left_motor.value, robot.right_motor.value]
res['threads_alive'] = [t.name for t in drive._threads if t.is_alive()] + (
    ['display'] if display_worker._thread.is_alive() else [])
res['stats'] = stats.value
res['display_error'] = display_worker.error
print('RESULT ' + json.dumps(res))
'''


# Headless notebook run
@unittest.skipUnless(HAVE_NB, "nbclient / ipykernel / ipywidgets not installed")
class LiveDemoNotebookTest(unittest.TestCase):
    # Full scenario in one kernel
    def test_drive_slow_display_and_end(self):
        # Checkpoint builder of the project
        from loader import build_model, write_meta
        # Temporary models and log folders
        with tempfile.TemporaryDirectory() as tmp:
            # Folder listed by the dropdown
            models = os.path.join(tmp, "models_trt")
            # Create it
            os.makedirs(models)
            # Small ResNet-18 without downloads
            net = build_model("resnet18", pretrained=False)
            # Constant prediction the controller accepts: zero weights, bias = target ahead-right
            with torch.no_grad():
                # No dependence on the input
                net.fc.weight.zero_()
                # Fixed target point
                net.fc.bias.copy_(torch.tensor([0.1, -0.3]))
            # Checkpoint path
            path = os.path.join(models, "const.pth")
            # Save the weights
            torch.save(net.state_dict(), path)
            # Minimal metadata (no sanity data -> 'missing', driving allowed)
            write_meta(path, {"label": "const test model", "preprocessing": {"method": "lane_marking"}})
            # The notebook
            nb = nbformat.read(os.path.join(ROOT, "live_demo.ipynb"), as_version=4)
            # Settings cell: dummy camera, temporary folders
            cfg = nb.cells[2]
            # Replace the three settings
            for old, new in (("MODELS_DIR = 'models_trt'", "MODELS_DIR = %r" % models),
                             ("CAMERA_SOURCE = 'jetbot'", "CAMERA_SOURCE = 'dummy'"),
                             ("LOG_DIR = 'runs'", "LOG_DIR = %r" % os.path.join(tmp, "runs"))):
                # The setting must exist exactly like this
                self.assertIn(old, cfg.source)
                # Replace it
                cfg.source = cfg.source.replace(old, new)
            # Append the scenario
            nb.cells.append(nbformat.v4.new_code_cell(SCENARIO))
            # Run every cell in project4/ (the notebook imports from '.')
            NotebookClient(nb, timeout=180, kernel_name="python3",
                           resources={"metadata": {"path": ROOT}}).execute()
        # Printed output of the scenario cell
        out = "".join(o.get("text", "") for o in nb.cells[-1].outputs if o.get("output_type") == "stream")
        # The JSON line
        line = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
        # It was printed
        self.assertTrue(line, out)
        # Parsed result
        res = json.loads(line[0][len("RESULT "):])
        # Print it for the HANDOFF record
        print("\n[live_demo] %s" % {k: v for k, v in res.items() if k not in ("stats", "status_after_stop")})
        # Model loaded and driving started (red stop button, motors running)
        self.assertTrue(res["loaded"] and res["driving"], res)
        # Button shows the driving state ("stop" in Chinese)
        self.assertEqual(res["button_driving"], "\u505c\u6b62")
        # Motors received values
        self.assertGreater(res["left"], 0.0)
        # Slow display: still driving, display FPS dropped, quality lowered
        self.assertTrue(res["still_driving"], res)
        # Display rate fell to about 2 per second
        self.assertLess(res["display_fps_slow"], 3.0)
        # JPEG quality was lowered
        self.assertLess(res["quality_slow"], 80)
        # Control period barely changed (lenient bound; the unit test is strict)
        self.assertLess(res["p95_slow"], res["p95_fast"] * 1.25 + 2.0)
        # After stop: green start button ("start driving") and a normal stop text
        self.assertEqual(res["button_after_stop"], "\u958b\u59cb\u884c\u99db")
        # Manual stop reason ("manual stop") shown
        self.assertIn("\u624b\u52d5\u505c\u6b62", res["status_after_stop"])
        # End: both threads ended within 1 s, motors zero
        self.assertLess(res["end_seconds"], 1.0)
        # No thread left
        self.assertEqual(res["threads_alive"], [])
        # Motors at zero
        self.assertEqual(res["motors"], [0.0, 0.0])
        # End button reports "camera released" (all threads ended)
        self.assertEqual(res["end_button"], "\u76f8\u6a5f\u5df2\u91cb\u653e")
        # Timing line shown (TASKS M0-1)
        self.assertIn("p95", res["stats"])
        # The display thread never raised
        self.assertIsNone(res["display_error"])


# Run the tests directly
if __name__ == "__main__":
    # unittest entry point
    unittest.main()
