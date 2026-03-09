"""
Run insurance-calibration tests on Databricks via the Jobs API.
Execute this with: python run_databricks_tests.py
"""
import os
import time
import json

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

w = WorkspaceClient()

NOTEBOOK_CONTENT = '''
import subprocess, sys, os

subprocess.run(["rm", "-rf", "/tmp/insurance-calibration"], check=True)
result_clone = subprocess.run(
    ["git", "clone", "https://github.com/burning-cost/insurance-calibration.git",
     "/tmp/insurance-calibration"],
    capture_output=True, text=True,
)
print("Clone:", result_clone.returncode)
if result_clone.returncode != 0:
    print(result_clone.stderr)
    raise RuntimeError("Clone failed")

pip_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance-calibration[dev]",
     "--quiet"],
    capture_output=True, text=True,
)
print("Install:", pip_result.returncode)
if pip_result.returncode != 0:
    print(pip_result.stderr[-1000:])

result = subprocess.run(
    [sys.executable, "-m", "pytest", "/tmp/insurance-calibration/tests/", "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd="/tmp/insurance-calibration",
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])
print("Return code:", result.returncode)
if result.returncode != 0:
    raise RuntimeError(f"Tests failed with return code {result.returncode}")
'''

# Upload as a Python script to DBFS
import base64
encoded = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()

# Use the script via a Spark Python task
run_config = {
    "run_name": "insurance-calibration-tests",
    "tasks": [
        {
            "task_key": "run-tests",
            "spark_python_task": {
                "python_file": "dbfs:/tmp/ic_test_runner.py",
            },
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 0,
                "spark_conf": {"spark.databricks.cluster.profile": "singleNode"},
                "custom_tags": {"ResourceClass": "SingleNode"},
            },
        }
    ],
}

# Write script to DBFS
import requests

host = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {"Authorization": f"Bearer {token}"}

# Upload script via DBFS API
upload_resp = requests.post(
    f"{host}/api/2.0/dbfs/put",
    headers=headers,
    json={
        "path": "/tmp/ic_test_runner.py",
        "contents": encoded,
        "overwrite": True,
    },
)
print("Upload:", upload_resp.status_code, upload_resp.text[:200])

# Submit the run
run_resp = requests.post(
    f"{host}/api/2.1/jobs/runs/submit",
    headers=headers,
    json=run_config,
)
print("Submit:", run_resp.status_code)
run_data = run_resp.json()
print("Response:", json.dumps(run_data, indent=2))

if "run_id" not in run_data:
    print("Failed to submit job")
    exit(1)

run_id = run_data["run_id"]
print(f"\nRun ID: {run_id}")
print(f"Monitor at: {host}/#job/runs/{run_id}")

# Poll for completion
print("\nPolling for completion...")
for i in range(60):
    time.sleep(20)
    status_resp = requests.get(
        f"{host}/api/2.1/jobs/runs/get",
        headers=headers,
        params={"run_id": run_id},
    )
    run_info = status_resp.json()
    state = run_info.get("state", {})
    life_cycle = state.get("life_cycle_state", "UNKNOWN")
    result_state = state.get("result_state", "")
    print(f"  [{i*20}s] {life_cycle} {result_state}")

    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        print(f"\nFinal state: {life_cycle} / {result_state}")

        # Get output
        output_resp = requests.get(
            f"{host}/api/2.1/jobs/runs/get-output",
            headers=headers,
            params={"run_id": run_id},
        )
        output_data = output_resp.json()
        logs = output_data.get("logs", "")
        print("\n--- LOGS ---")
        print(logs[-5000:] if len(logs) > 5000 else logs)

        error = output_data.get("error", "")
        if error:
            print("\n--- ERROR ---")
            print(error)

        break
else:
    print("Timed out waiting for job")
