# Databricks notebook source

import subprocess, sys

subprocess.run(["rm", "-rf", "/tmp/insurance-calibration"], check=True)
r = subprocess.run(
    ["git", "clone", "https://github.com/burning-cost/insurance-calibration.git",
     "/tmp/insurance-calibration"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    dbutils.notebook.exit("CLONE FAILED: " + r.stderr[:500])

r2 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/tmp/insurance-calibration[dev]", "-q"],
    capture_output=True, text=True,
)
if r2.returncode != 0:
    dbutils.notebook.exit("PIP FAILED: " + r2.stderr[:500])

result = subprocess.run(
    [sys.executable, "-m", "pytest", "/tmp/insurance-calibration/tests/",
     "-v", "--tb=short", "--no-header"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-calibration",
)
output = result.stdout + "\nRC=" + str(result.returncode)
# notebook.exit captures at most ~1MB; trim from start if needed
dbutils.notebook.exit(output[-5000:] if len(output) > 5000 else output)
