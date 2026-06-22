"""Universal visualization of processed StandardFrameData output.

``render`` builds a per-frame figure that auto-detects whichever modalities a
processed ``.npz`` carries; ``visualize_processed`` is the CLI that turns a
processed dataset folder into one MP4 per scene.
"""

from standard_e2e.visualization.render import figure_to_bgr, render_frame

__all__ = ["render_frame", "figure_to_bgr"]
