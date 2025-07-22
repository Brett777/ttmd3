#!/usr/bin/env sh

# add set -e so if pip install fails, the build will be marked as failed
set -e

echo "=== BUILD-APP.SH STARTING ==="
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la

# Install Python dependencies
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt, installing Python dependencies..."
    echo "Current PATH: $PATH"
    echo "Which pip: $(which pip)"
    echo "Pip version: $(pip --version)"
    
    # Try different installation approaches
    echo "Running: pip install --system -r requirements.txt"
    pip install --system -r requirements.txt || {
        echo "System install failed, trying without flags..."
        pip install -r requirements.txt
    }
    
    echo "Checking if uvicorn was installed:"
    echo "Searching for uvicorn executable:"
    which uvicorn || echo "uvicorn not in PATH"
    find /usr -name uvicorn -type f 2>/dev/null | head -10
    find /root -name uvicorn -type f 2>/dev/null | head -10
    
    echo "Python packages installed:"
    pip list | grep uvicorn || echo "uvicorn not in pip list"
else
    echo "No requirements.txt found. Skipping pip install."
fi

echo "=== PYTHON DEPENDENCIES COMPLETE ==="

# Install and build the React frontend
cd frontend

echo "Installing React dependencies from package.json..."
npm install

echo "Building React app..."
# Use double quotes to allow shell variable expansion for BASE_PATH
# Also, ensure the path is correctly formatted as /path/
npm run build -- --base="/$BASE_PATH/"

echo "=== BUILD-APP.SH COMPLETE ==="