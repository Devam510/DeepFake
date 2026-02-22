#!/usr/bin/env python
"""
DeepFake Detection System - CLI Entry Point

This wrapper resolves the RuntimeWarning about module execution order
by providing a clean entry point for CLI usage.
"""

import sys
import os

# Ensure proper module resolution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.detection_api import handle_analyze_request
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="DeepFake Detection API")
    parser.add_argument("file", help="Media file to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--no-provenance", action="store_true", help="Skip provenance check"
    )

    args = parser.parse_args()

    request_data = {"file_path": args.file, "check_provenance": not args.no_provenance}

    result = handle_analyze_request(request_data)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
