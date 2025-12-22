"""Saving helpers for masks and images."""

import os
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

from .config import GS_EXE


def save_bg_jpeg(bg_bgr: np.ndarray, out_path: str, quality: int = 40):
    # MRC-style: BG can be aggressively compressed because sharp stuff is in FG
    cv2.imwrite(
        out_path,
        bg_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    )


def save_fg_jpeg(fg_bgr: np.ndarray, out_path: str, quality: int = 40):
    # MRC-style: BG can be aggressively compressed because sharp stuff is in FG
    cv2.imwrite(
        out_path,
        fg_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality), cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    )


def save_mask_jbig2_via_gs(mask_black_fg: np.ndarray, out_path: str):
    """
    Save binary mask as JBIG2 using Ghostscript (Windows).
    Uses PBM (P4) intermediate.
    BLACK (0) = FG => '1' bit in PBM (black).
    """
    if not os.path.exists(GS_EXE):
        raise RuntimeError(
            "Ghostscript not found. Update GS_EXE path to gswin64c.exe.\n"
            f"Current: {GS_EXE}"
        )

    bilevel = (mask_black_fg == 0).astype(np.uint8)  # 1 for FG/black

    with tempfile.TemporaryDirectory() as tmp:
        pbm_path = os.path.join(tmp, "mask.pbm")

        h, w = bilevel.shape
        with open(pbm_path, "wb") as f:
            f.write(f"P4\n{w} {h}\n".encode("ascii"))
            packed = np.packbits(bilevel, axis=1)
            f.write(packed.tobytes())

        cmd = [
            GS_EXE,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=jbig2",
            f"-sOutputFile={out_path}",
            pbm_path
        ]
        subprocess.check_call(cmd)


def save_mask_tiff_g4(mask_black_fg: np.ndarray, out_path: str):
    """
    Save binary mask as TIFF CCITT Group 4 (1-bit, lossless).
    BLACK (0) = FG
    """
    # Convert to 1-bit image: FG=black(0), BG=white(255)
    bilevel = (mask_black_fg == 0).astype(np.uint8) * 255

    img = Image.fromarray(bilevel, mode="L").convert("1")

    img.save(
        out_path,
        format="TIFF",
        compression="group4"
    )
