"""Reality View: free-viewpoint ego-surround rendering from six cameras.

DA3 metric depth -> point cloud -> splat into any orbit camera; a one-step MeanFlow DiT fills what no camera saw;
real pixels are kept where available and a stylised ego car is drawn on top.
"""
__version__ = "0.1.0"
