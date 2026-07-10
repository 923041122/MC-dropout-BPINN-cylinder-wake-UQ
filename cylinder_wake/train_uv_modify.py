"""Compatibility entry point for the formal MC-dropout B-PINN training run."""
import sys
from benchmark_train import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--method", "bpinn_dropout"])
    main()
