"""Run a small, reproducible training and inference smoke test."""
import sys
from training_cli import main

if __name__ == '__main__':
    main(argv=['--smoke', *sys.argv[1:]])
