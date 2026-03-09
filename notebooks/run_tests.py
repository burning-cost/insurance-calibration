# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-calibration: Run Tests
# MAGIC
# MAGIC Installs the library from GitHub and runs the full test suite.

# COMMAND ----------

# MAGIC %pip install "numpy>=1.24" "scipy>=1.12" "polars>=0.20" "matplotlib>=3.7" "pytest>=7.0" "pytest-cov" git+https://github.com/burning-cost/insurance-calibration.git

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/root/.pyenv/versions/3.11.0/lib/python3.11/site-packages/insurance_calibration/../../../",
        "-v", "--tb=short", "-x",
    ],
    capture_output=True,
    text=True,
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)

# COMMAND ----------

# Alternative: clone the repo and run pytest on the local test files
import subprocess, sys, os

# Clone the repository
subprocess.run(["rm", "-rf", "/tmp/insurance-calibration"], check=True)
subprocess.run(
    ["git", "clone", "https://github.com/burning-cost/insurance-calibration.git",
     "/tmp/insurance-calibration"],
    check=True,
)

# Install in editable mode
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance-calibration"],
    check=True, capture_output=True,
)

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest", "/tmp/insurance-calibration/tests/", "-v", "--tb=short"],
    capture_output=True,
    text=True,
    cwd="/tmp/insurance-calibration",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
