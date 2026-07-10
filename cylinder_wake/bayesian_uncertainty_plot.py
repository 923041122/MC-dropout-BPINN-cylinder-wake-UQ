"""Compatibility entry point for the manuscript uncertainty evaluation.

The previous standalone defaults (dropout=0.05, MC=30 and a legacy checkpoint)
were removed. This wrapper now delegates to benchmark_evaluate.py, whose formal
defaults are dropout=0.002 and 50 MC samples.
"""
from benchmark_evaluate import main

if __name__ == "__main__":
    main()
