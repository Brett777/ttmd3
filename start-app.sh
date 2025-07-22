#!/usr/bin/env sh

echo "Starting FastAPI server..."

# Use the PORT environment variable provided by DataRobot, with a fallback to 8080 for local dev.
APP_PORT=${PORT:-8080}

# Use the BASE_PATH environment variable for the --root-path, which is crucial for deployment in DataRobot.
# This tells FastAPI that it's running behind a proxy at a sub-path.
UVICORN_ROOT_PATH=${BASE_PATH:-/}

# Quick diagnostic to understand the environment
echo "PATH is: $PATH"
echo "Checking for uvicorn in common locations:"
ls -la /root/.local/bin/uvicorn 2>/dev/null || echo "Not in /root/.local/bin/"
ls -la /usr/local/bin/uvicorn 2>/dev/null || echo "Not in /usr/local/bin/"
which uvicorn 2>/dev/null || echo "uvicorn not found in PATH"

# Run the server using uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port ${APP_PORT} --root-path ${UVICORN_ROOT_PATH}