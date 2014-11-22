#!/usr/bin/env python

import argparse
from web import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs Topmodel Server")
    parser.add_argument(
        "--remote", "-r", action="store_true", default=False, help="Use data from S3")
    parser.add_argument("--development", "-d", action="store_true",
                        default=False, help="Run topmodel in development mode with autoreload")
    args = parser.parse_args()
    app.local = not args.remote
    app.run(port=9191, host="0.0.0.0",
            debug=True, use_reloader=args.development)
