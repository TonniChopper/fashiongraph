"""Vision stack: fashion embedder, garment segmenter, visual index.

Heavy deps (torch, open_clip, transformers) are imported lazily inside each
module so importing ``fg.vision`` stays cheap and the rest of the app runs
without them installed.
"""
