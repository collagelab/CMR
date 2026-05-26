"""
Retriever implementations used inside the core package.

We keep these inside the package so that console entrypoints like
`train-baseline` can import them reliably, without depending on the
top-level `data/` directory layout.
"""

