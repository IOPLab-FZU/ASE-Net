"""Compatibility entry point. Prefer test.py, evaluation.py and visualization.py."""

from evaluation import Inference, feature_gene, pre_caption
from option.options import options
from test import main
from utils.data_paths import change_path
from visualization import (
    Image_Token_Selection1,
    Image_Token_Selection,
    Text_Token_Visualize,
    rank_token_visualization,
    rank_token_visualization2,
)

__all__ = [
    "Inference",
    "feature_gene",
    "pre_caption",
    "main",
    "change_path",
    "Image_Token_Selection1",
    "Image_Token_Selection",
    "Text_Token_Visualize",
    "rank_token_visualization",
    "rank_token_visualization2",
]

if __name__ == "__main__":
    main(options().opt)
