import tempfile
import unittest
import numpy as np
from experiments.historical.openmm_cpu_rng import historical_openmm_native as adapter


class AdapterChecks(unittest.TestCase):
    def test_native_roundtrip_and_contract(self):
        import openmm
        import importlib.metadata

        cfg = {
            "package_version": importlib.metadata.version("openmm"),
            "git_revision": openmm.version.git_revision,
            "platform": "CPU",
            "seed": 42,
        }
        with tempfile.TemporaryDirectory() as path:
            s = adapter.create(cfg, path)
            adapter.advance_one(s)
            before = adapter.observe(s)
            self.assertEqual(int(before["step"]), 1)
            self.assertEqual(before["position"].shape, (10, 3))
            self.assertEqual(before["box"].shape, (3, 3))
            adapter.save(s, path)
            after = adapter.observe(s)
            loaded = adapter.restore(cfg, path, path)
            restored = adapter.observe(loaded)
            self.assertEqual(set(before), set(adapter.UNITS))
            for field in before:
                self.assertTrue(np.array_equal(before[field], after[field]), field)
                self.assertTrue(np.array_equal(before[field], restored[field]), field)
            self.assertEqual(adapter.collect_outputs(path), {})

    def test_profile_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            adapter.create(
                {
                    "package_version": "0.0.0",
                    "git_revision": "invalid",
                    "platform": "CPU",
                    "seed": 42,
                },
                ".",
            )


if __name__ == "__main__":
    unittest.main()
