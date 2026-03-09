"""
Run insurance-calibration tests on Databricks serverless via notebook task.
"""
import os
import time
import json
import requests
import base64
import tempfile

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ[key.strip()] = val.strip()

host = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Notebook content as Jupyter format
notebook_cells = [
    {
        "cell_type": "code",
        "source": (
            "import subprocess, sys\n"
            "subprocess.run(['rm', '-rf', '/tmp/insurance-calibration'], check=True)\n"
            "result = subprocess.run(\n"
            "    ['git', 'clone', 'https://github.com/burning-cost/insurance-calibration.git',\n"
            "     '/tmp/insurance-calibration'],\n"
            "    capture_output=True, text=True,\n"
            ")\n"
            "print('Clone rc:', result.returncode)\n"
            "print(result.stdout[:500])\n"
            "if result.returncode != 0:\n"
            "    print(result.stderr)\n"
            "    raise RuntimeError('Clone failed')\n"
        ),
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "code",
        "source": (
            "import subprocess, sys\n"
            "pip_result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pip', 'install', '-e', '/tmp/insurance-calibration[dev]', '-q'],\n"
            "    capture_output=True, text=True,\n"
            ")\n"
            "print('Install rc:', pip_result.returncode)\n"
            "if pip_result.stderr:\n"
            "    print(pip_result.stderr[-500:])\n"
        ),
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "code",
        "source": (
            "import subprocess, sys\n"
            "result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pytest', '/tmp/insurance-calibration/tests/', '-v', '--tb=short'],\n"
            "    capture_output=True, text=True,\n"
            "    cwd='/tmp/insurance-calibration',\n"
            ")\n"
            "print(result.stdout)\n"
            "if result.stderr:\n"
            "    print('STDERR:', result.stderr[-1000:])\n"
            "print('Return code:', result.returncode)\n"
            "assert result.returncode == 0, f'Tests failed: rc={result.returncode}'\n"
        ),
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    },
]

notebook_json = {
    "nbformat": 4,
    "nbformat_minor": 2,
    "metadata": {"language_info": {"name": "python"}},
    "cells": notebook_cells,
}

notebook_content = json.dumps(notebook_json)
encoded = base64.b64encode(notebook_content.encode()).decode()

# Upload notebook to workspace
workspace_path = "/Workspace/Users/pricing.frontier@gmail.com/insurance_calibration_tests"

# First ensure directory exists
import_resp = requests.post(
    f"{host}/api/2.0/workspace/import",
    headers=headers,
    json={
        "path": workspace_path,
        "format": "JUPYTER",
        "language": "PYTHON",
        "content": encoded,
        "overwrite": True,
    },
)
print("Import:", import_resp.status_code, import_resp.text[:300])

if import_resp.status_code != 200:
    # Try source format
    source_content = "\n# COMMAND ----------\n".join(
        cell["source"] for cell in notebook_cells
    )
    source_encoded = base64.b64encode(source_content.encode()).decode()
    import_resp = requests.post(
        f"{host}/api/2.0/workspace/import",
        headers=headers,
        json={
            "path": workspace_path,
            "format": "SOURCE",
            "language": "PYTHON",
            "content": source_encoded,
            "overwrite": True,
        },
    )
    print("Import (SOURCE):", import_resp.status_code, import_resp.text[:300])

# Submit as a serverless notebook task
run_config = {
    "run_name": "insurance-calibration-tests",
    "tasks": [
        {
            "task_key": "run-tests",
            "notebook_task": {
                "notebook_path": workspace_path,
                "source": "WORKSPACE",
            },
            "libraries": [],
        }
    ],
    "queue": {"enabled": True},
}

run_resp = requests.post(
    f"{host}/api/2.1/jobs/runs/submit",
    headers=headers,
    json=run_config,
)
print("Submit:", run_resp.status_code)
run_data = run_resp.json()
print("Response:", json.dumps(run_data, indent=2))

if "run_id" not in run_data:
    print("Failed to submit. Full response:", json.dumps(run_data))
    exit(1)

run_id = run_data["run_id"]
print(f"\nRun ID: {run_id}")

# Poll for completion
print("\nPolling for completion...")
for i in range(80):
    time.sleep(15)
    status_resp = requests.get(
        f"{host}/api/2.1/jobs/runs/get",
        headers=headers,
        params={"run_id": run_id},
    )
    run_info = status_resp.json()
    state = run_info.get("state", {})
    life_cycle = state.get("life_cycle_state", "UNKNOWN")
    result_state = state.get("result_state", "")
    print(f"  [{i*15}s] {life_cycle} {result_state}")

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
        notebook_output = output_data.get("notebook_output", {})

        if notebook_output:
            print("\n--- NOTEBOOK OUTPUT ---")
            result_text = notebook_output.get("result", "")
            print(result_text[-5000:] if len(result_text) > 5000 else result_text)

        if logs:
            print("\n--- LOGS ---")
            print(logs[-3000:] if len(logs) > 3000 else logs)

        error = output_data.get("error", "")
        if error:
            print("\n--- ERROR ---")
            print(error)

        break
else:
    print("Timed out after 20 minutes")
