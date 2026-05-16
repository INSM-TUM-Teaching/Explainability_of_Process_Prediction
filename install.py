#!/usr/bin/env python3

import os
import platform
import re
import subprocess
import sys

def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def pip(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "pip", *args], check=check)

def _nvidia_cuda_version() -> str | None:
    """
    Query nvidia-smi for the installed CUDA version.
    Returns a string like '12.4' or None when no NVIDIA GPU is present.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _rocm_available() -> bool:
    """Return True when an AMD GPU with ROCm is detected."""
    for cmd in (["rocm-smi", "--version"], ["rocminfo"]):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False

def _cuda_wheel(cuda_version: str) -> str:
    """
    Map a CUDA version string to the closest PyTorch wheel index.
    Supported builds: cu118, cu121, cu124.
    """
    major, minor = (int(x) for x in cuda_version.split(".")[:2])
    if major > 12 or (major == 12 and minor >= 4):
        whl, label = "cu124", "CUDA 12.4"
    elif major == 12:
        whl, label = "cu121", "CUDA 12.1"
    else:  # CUDA 11.x or older
        whl, label = "cu118", "CUDA 11.8"
    print(f"       → Using PyTorch build for {label}  (detected CUDA {cuda_version})")
    return f"https://download.pytorch.org/whl/{whl}"


def torch_install_args() -> list[str]:
    """
    Return the argument list for `pip install <args>` that installs the
    right PyTorch variant for the current platform.
    """
    system = platform.system()
    machine = platform.machine().lower()

    packages = ["torch", "torchvision", "torchaudio"]

    if system == "Darwin":
        if "arm" in machine or "aarch" in machine:
            print("[INFO] macOS (Apple Silicon) detected.")
            print("       PyTorch ships with MPS (Metal Performance Shaders) support.")
        else:
            print("[INFO] macOS (Intel) detected.")
            print("       PyTorch will run on CPU (no MPS on Intel Macs).")

    cuda = _nvidia_cuda_version()
    if cuda:
        print(f"[INFO] {system}: NVIDIA GPU detected (CUDA {cuda}).")
        index_url = _cuda_wheel(cuda)
        return [*packages, "--index-url", index_url]

    if system == "Linux" and _rocm_available():
        print("[INFO] Linux: AMD GPU with ROCm detected.")
        return [
            *packages,
            "--index-url",
            "https://download.pytorch.org/whl/rocm6.2",
        ]

    print(f"[INFO] {system}: No GPU / CUDA detected — installing CPU-only PyTorch.")
    return [*packages, "--index-url", "https://download.pytorch.org/whl/cpu"]


def main() -> None:
    print("=" * 62)
    print("  Project Install")
    print(f"  OS      : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python  : {sys.version.split()[0]}")
    print("=" * 62)
    print("\n[1/4] Upgrading pip …")
    pip("install", "--upgrade", "pip")
    print("\n[2/4] Installing PyTorch …")
    args = torch_install_args()
    pip("install", *args)
    print("\n[3/4] Installing torch-geometric …")
    pip("install", "torch-geometric")
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    print(f"\n[4/4] Installing project requirements ({req}) …")
    pip("install", "-r", req)

    print("\n" + "=" * 62)
    print("  Installation complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()
