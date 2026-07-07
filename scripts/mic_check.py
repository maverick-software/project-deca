"""WS6 host mic diagnostic: list input devices and measure live RMS per device.

The 2026-07-06 probe showed the intake stream OPEN and DELIVERING but at
~1e-05 RMS during speech -- a host problem (wrong default device / Windows
mic privacy / muted input), not an intake problem. This tool answers: which
device index actually hears the room? Set DECADIC_AUDIO_DEVICE to that index.

Usage:
    python scripts/mic_check.py            # list devices + test the default
    python scripts/mic_check.py --all      # test every input device (3s each)
    python scripts/mic_check.py --device 5 # test one device index
Speak continuously while it runs.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _test_device(sd, device: int | None, seconds: float = 3.0) -> float | None:
    try:
        with sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", device=device
        ) as stream:
            frames = int(seconds * 16000)
            buf, _ = stream.read(frames)
        return float(np.sqrt(np.mean(np.square(buf.astype(np.float64)))))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
        print(f"    device={device}: OPEN FAILED ({type(exc).__name__}: {exc})")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()

    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        print(f"sounddevice unavailable: {exc}")
        return 1

    devices = sd.query_devices()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    print(f"default input device index: {default_in}\n")
    inputs = [
        (i, d) for i, d in enumerate(devices) if int(d.get("max_input_channels", 0)) > 0
    ]
    for i, d in inputs:
        mark = " <-- default" if i == default_in else ""
        print(f"  [{i}] {d['name']} (in={d['max_input_channels']}){mark}")

    targets: list[int | None]
    if args.device is not None:
        targets = [args.device]
    elif args.all:
        targets = [i for i, _ in inputs]
    else:
        targets = [None]  # the default device

    print("\nSPEAK CONTINUOUSLY -- measuring 3s RMS per device:")
    verdicts = []
    for dev in targets:
        rms = _test_device(sd, dev)
        if rms is None:
            continue
        label = "default" if dev is None else str(dev)
        alive = rms > 0.003
        verdicts.append((label, rms, alive))
        print(f"    device={label}: rms={rms:.5f} {'HEARS YOU' if alive else 'silent/dead'}")
    live = [v for v in verdicts if v[2]]
    weak = [v for v in verdicts if not v[2] and v[1] > 2e-4]  # signal, but faint
    if live or weak:
        best = max(live or weak, key=lambda v: v[1])
        if best[0] != "default":
            print(f"\nUse: $env:DECADIC_AUDIO_DEVICE = \"{best[0]}\"")
        # Recommend a capture gain targeting ~0.08 RMS speech (2026-07-06: a
        # live device delivering 0.002 RMS speech sat below the silence gate).
        gain = max(1.0, min(50.0, 0.08 / max(1e-4, best[1])))
        if gain > 1.5:
            print(f"Use: $env:DECADIC_AUDIO_GAIN = \"{gain:.0f}\"  "
                  f"(measured speech rms {best[1]:.4f}; target ~0.08)")
    else:
        print("\nNo device heard you. Check Windows Settings > Privacy & security > "
              "Microphone > 'Let desktop apps access your microphone', and input levels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
