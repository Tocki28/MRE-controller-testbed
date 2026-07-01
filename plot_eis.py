"""Capture one EIS sweep from STM32 UART and plot Nyquist + Bode panels.

Usage:
  python3.11 plot_eis.py                         # read from board
  python3.11 plot_eis.py --port /dev/ttyACM0     # specify port
  python3.11 plot_eis.py --demo                  # synthetic data (no board needed)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testbed.eis_analysis import SweepParser

DEFAULT_PORT = "/dev/tty.usbmodem142103"
BAUD = 115200
OUT = "eis_sweep.png"
ANNOTATION = "STM32 Nucleo G431KB + TLV9061 | 1 Hz – 4.6 kHz"


def _demo_sweep() -> dict:
    """Synthetic Randles+Warburg sweep — testbed room-temp Fe/Ca/Mg/Al solution."""
    from testbed.randles_model import randles_impedance
    freq = np.logspace(0, np.log10(4600), 32)
    omega = 2 * np.pi * freq
    # Rs=100Ω, Rct=50Ω, Cdl=100µF, sigma=5 (Warburg)
    Z = randles_impedance(omega, Rs=100.0, Rct=50.0, Cdl=1e-4, sigma=5.0)
    return {"freq": list(freq), "Z_re": list(Z.real), "Z_im": list(Z.imag)}


def _read_sweep_from_serial(port: str) -> dict:
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed — run: pip install pyserial", file=sys.stderr)
        sys.exit(1)

    sp = SweepParser()
    print(f"Opening {port} at {BAUD} baud …")
    try:
        with serial.Serial(port, BAUD, timeout=30) as ser:
            print("Waiting for complete sweep …")
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace")
                sp.parse_line(line)  # # DBG lines ignored by parser
                if sp.is_complete_after(line):
                    sweep = sp.get_sweep()
                    if sweep and len(sweep["freq"]) >= 10:
                        break
                    sp = SweepParser()
                    print("Partial sweep discarded, waiting for next …")
                    continue
    except serial.SerialException as exc:
        print(f"ERROR: Cannot open {port!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    return sweep


def _plot(sweep: dict, title_suffix: str) -> None:
    freq  = np.asarray(sweep["freq"])
    Z_re  = np.asarray(sweep["Z_re"])
    Z_im  = np.asarray(sweep["Z_im"])
    Z_abs = np.hypot(Z_re, Z_im)
    phase = np.degrees(np.arctan2(Z_im, Z_re))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")
    ax_nyq, ax_mag, ax_ph = axes

    ax_nyq.plot(Z_re, -Z_im, "o-", color="#2196F3", markersize=5, linewidth=1.5)
    ax_nyq.set_aspect("equal")
    ax_nyq.set(title=f"Nyquist{title_suffix}", xlabel="Re(Z) / Ω", ylabel="−Im(Z) / Ω")
    ax_nyq.grid(True, alpha=0.4)

    ax_mag.loglog(freq, Z_abs, "o-", color="#4CAF50", markersize=5, linewidth=1.5)
    ax_mag.set(title=f"|Z| vs Frequency{title_suffix}", xlabel="Frequency / Hz", ylabel="|Z| / Ω")
    ax_mag.grid(True, which="both", alpha=0.4)

    ax_ph.semilogx(freq, phase, "o-", color="#9C27B0", markersize=5, linewidth=1.5)
    ax_ph.set(title=f"Phase vs Frequency{title_suffix}", xlabel="Frequency / Hz", ylabel="Phase / °")
    ax_ph.grid(True, which="both", alpha=0.4)

    fig.text(0.99, 0.01, ANNOTATION, ha="right", va="bottom",
             fontsize=7, color="#888888", transform=fig.transFigure)

    plt.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=os.environ.get("EIS_PORT", DEFAULT_PORT))
    ap.add_argument("--demo", action="store_true", help="Use synthetic Randles data (no board needed)")
    args = ap.parse_args()

    if args.demo:
        print("Demo mode — generating synthetic Randles+Warburg sweep …")
        sweep = _demo_sweep()
        _plot(sweep, " (synthetic)")
    else:
        sweep = _read_sweep_from_serial(args.port)
        _plot(sweep, "")


if __name__ == "__main__":
    main()
