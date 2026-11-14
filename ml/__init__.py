"""ClearCart ML package.

macOS note: PyTorch ships its own libomp and XGBoost links Homebrew's libomp. With both
loaded and multi-threaded OpenMP, the process segfaults (exit 139) on the first MiniLM
encode. Pinning OpenMP to one thread before either library is imported avoids it; the
dataset is small so the speed cost is negligible.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
