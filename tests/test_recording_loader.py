import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from bioloc import load_recording, local_to_gps_coordinates, read_position_csv


class RecordingLoaderTest(unittest.TestCase):
    def test_current_dated_layout_and_headerless_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            positions = root / "positions.csv"
            positions.write_text(
                "49.0000,14.0000,mic-b\n"
                "49.0001,14.0001,mic-a\n"
            )
            for recorder_id, amplitude in (("mic-a", 0.1), ("mic-b", 0.2)):
                folder = root / recorder_id / "20260102"
                folder.mkdir(parents=True)
                sf.write(
                    folder / f"device_20260102_030405_SYNC.WAV",
                    np.full(1000, amplitude, dtype=np.float32),
                    1000,
                )
            recording = load_recording(
                root,
                positions,
                ["mic-a", "mic-b"],
                recording_date="2026-01-02",
                recording_time=30405,
                start_seconds=0.1,
                duration_seconds=0.2,
            )
            self.assertEqual(recording.microphone_names, ("mic-a", "mic-b"))
            self.assertEqual(recording.signals.shape, (2, 200))
            self.assertEqual(recording.sample_rate, 1000)
            self.assertAlmostEqual(float(np.mean(recording.signals[0])), 0.1, places=3)
            self.assertAlmostEqual(float(np.mean(recording.signals[1])), 0.2, places=3)
            self.assertEqual(recording.microphone_positions.shape, (2, 2))
            map_info = recording.metadata["map_info"]
            self.assertEqual(map_info["environment_margin_m"], 100.0)
            np.testing.assert_allclose(map_info["plot_min"], [0.0, 0.0])
            np.testing.assert_allclose(
                map_info["plot_max"],
                [map_info["environment_size"], map_info["environment_size"]],
            )
            recovered_gps = local_to_gps_coordinates(
                recording.microphone_positions,
                map_info,
            )
            np.testing.assert_allclose(
                recovered_gps,
                np.asarray([[49.0001, 14.0001], [49.0, 14.0]]),
                atol=1e-10,
            )

    def test_undated_layout_and_headered_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            positions = root / "positions.csv"
            with positions.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["recorder_id", "latitude", "longitude"])
                writer.writerow(["north", 50.0, 4.0])
                writer.writerow(["south", 49.999, 4.0])
            for recorder_id in ("north", "south"):
                folder = root / recorder_id
                folder.mkdir()
                sf.write(folder / "recording_SYNC.wav", np.zeros(400), 8000)
            recording = load_recording(
                root,
                positions,
                ["south", "north"],
                recording_date=None,
            )
            self.assertEqual(recording.microphone_names, ("south", "north"))
            self.assertEqual(recording.signals.shape, (2, 400))
            self.assertLess(
                recording.microphone_positions[0, 1],
                recording.microphone_positions[1, 1],
            )

    def test_missing_and_duplicate_position_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "positions.csv"
            path.write_text("1,2,mic\n3,4,mic\n")
            with self.assertRaisesRegex(ValueError, "Duplicate recorder ID"):
                read_position_csv(path)


if __name__ == "__main__":
    unittest.main()
