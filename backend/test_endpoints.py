"""
Test script for document processing endpoints.
Run this after starting the FastAPI server to verify endpoints are working.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8080"
SESSION_ID = "test-session-123"

def test_models_endpoint():
    """Test the models endpoint."""
    print("Testing /api/models...")
    response = requests.get(f"{BASE_URL}/api/models")
    if response.status_code == 200:
        models = response.json()["models"]
        print(f"✓ Found {len(models)} models")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        return False

def test_document_upload():
    """Test document upload endpoint."""
    print("\nTesting /api/documents/upload...")
    
    # Create a simple test text file
    test_content = "This is a test document for RAG-Ultra processing."
    test_filename = "test_document.txt"
    
    files = {
        'file': (test_filename, test_content, 'text/plain')
    }
    data = {
        'extractionModel': 'gpt-4o',
        'session_id': SESSION_ID
    }
    
    response = requests.post(f"{BASE_URL}/api/documents/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Upload successful: Job ID = {result['jobId']}")
        return result['jobId']
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")
        return None

def test_job_status(job_id):
    """Test job status endpoint."""
    print(f"\nTesting /api/documents/process/{job_id}/status...")
    
    response = requests.get(
        f"{BASE_URL}/api/documents/process/{job_id}/status",
        params={'session_id': SESSION_ID}
    )
    
    if response.status_code == 200:
        status = response.json()
        print(f"✓ Status: {status['status']} - {status['progress']['message']}")
        return status
    else:
        print(f"✗ Failed: {response.status_code}")
        return None

def test_list_documents():
    """Test list documents endpoint."""
    print(f"\nTesting /api/documents/{SESSION_ID}...")
    
    response = requests.get(f"{BASE_URL}/api/documents/{SESSION_ID}")
    
    if response.status_code == 200:
        documents = response.json()["documents"]
        print(f"✓ Found {len(documents)} documents")
        return documents
    else:
        print(f"✗ Failed: {response.status_code}")
        return []

def main():
    """Run all tests."""
    print("Document Processing Endpoints Test")
    print("=" * 50)
    
    # Test models endpoint
    if not test_models_endpoint():
        print("Models endpoint failed. Is the server running?")
        return
    
    # Test document upload
    job_id = test_document_upload()
    if not job_id:
        print("Upload failed.")
        return
    
    # Test job status
    status = test_job_status(job_id)
    if not status:
        print("Status check failed.")
        return
    
    # Wait a bit for processing
    print("\nWaiting for processing to complete...")
    time.sleep(5)
    
    # Check status again
    status = test_job_status(job_id)
    
    # List documents
    documents = test_list_documents()
    
    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main() 