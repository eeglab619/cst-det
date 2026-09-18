import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from protocol import split_indices
from release import require_core
from metrics import detection_metrics
ROOT = Path(__file__).resolve().parents[1]

class FrameworkChecks(unittest.TestCase):

    def test_cli(self):
        result = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--describe'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('after', result.stdout)
        blocked = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--data', 'missing.npz'], capture_output=True, text=True)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn('NotImplementedError', blocked.stderr)
        self.assertNotIn('FileNotFoundError', blocked.stderr)

    def test_no_block_leakage(self):
        ids = list(range(1, 21)) * 2
        groups = split_indices(ids)
        self.assertEqual({ids[i] for i in groups['train']}, set(range(1, 13)))
        self.assertEqual({ids[i] for i in groups['validation']}, set(range(13, 17)))
        self.assertEqual({ids[i] for i in groups['test']}, set(range(17, 21)))
        self.assertFalse(set(groups['train']) & set(groups['test']))
        for invalid in [[], [0, 13, 20], [1.5, 13, 20], [1, 2, 3], [1, 13, 21]]:
            with self.assertRaises(ValueError):
                split_indices(invalid)

    def test_fail_closed(self):
        with self.assertRaisesRegex(NotImplementedError, 'accepted'):
            require_core()
        from pipeline import run_experiment
        with self.assertRaises(NotImplementedError):
            run_experiment('missing.npz', {})

    def test_metrics(self):
        result = detection_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result['accuracy'], 0.75)
        self.assertEqual(result['sensitivity'], 1)
        self.assertEqual(result['specificity'], 0.5)
        self.assertIsNone(detection_metrics([1], [1])['balanced_accuracy'])
        with self.assertRaises(ValueError):
            detection_metrics([0], [2])

    def test_fbcca_helper(self):
        from models.fbcca import FBCCA
        t = np.arange(512) / 256
        eeg = np.stack([np.sin(2 * np.pi * 10 * t + p) for p in np.linspace(0, 1, 8)])
        (_, index, scores) = FBCCA(list(range(8, 16)), fs=256).predict(eeg)
        self.assertEqual(scores.shape, (8,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertIn(index, range(8))

    def test_core_bodies_are_absent(self):
        path = ROOT / CORE_FILE
        classes = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)}
        for name in CORE_NAMES:
            methods = [n for n in classes[name].body if isinstance(n, ast.FunctionDef)]
            self.assertTrue(methods)
            for method in methods:
                self.assertEqual(len(method.body), 1)
                self.assertIsInstance(method.body[0], ast.Raise)
                self.assertEqual(ast.unparse(method.body[0].exc), 'NotImplementedError(RELEASE_NOTICE)')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed; static stub checks still run')
    def test_core_runtime(self):
        import importlib
        module = importlib.import_module(CORE_MODULE)
        for (name, args) in CORE_ARGS.items():
            with self.assertRaisesRegex(NotImplementedError, 'accepted'):
                getattr(module, name)(*args)

    def test_segments_and_shaping(self):
        from segments import find_events, slice_signal_by_events
        from postprocess import temporal_shaping
        events = find_events(np.array([1, 1, 0, 1]), 32)
        np.testing.assert_array_equal(events, [[0, 64], [96, 128]])
        pieces = slice_signal_by_events(np.zeros((8, 128)), events)
        self.assertEqual([p.shape for p in pieces[0]], [(8, 64), (8, 32)])
        self.assertEqual(find_events(np.zeros(4)).shape, (0, 2))
        with self.assertRaises(NotImplementedError):
            temporal_shaping([0, 1, 0])
CORE_FILE = 'models/cst_det.py'
CORE_MODULE = 'models.cst_det'
CORE_NAMES = ['SpatiotemporalFeatures', 'LearnableSmoothing']
CORE_ARGS = {'SpatiotemporalFeatures': [], 'LearnableSmoothing': [], 'CSTDet': []}
if __name__ == '__main__':
    unittest.main()
