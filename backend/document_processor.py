"""
Document processing module for RAG-Ultra integration.
Handles document upload, processing with progress tracking, and job management.
"""

import os
import sys
import uuid
import asyncio
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
import logging
from functools import partial, wraps

# Add the backend directory to Python path to find rag_ultra
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import RAG-Ultra modules
from rag_ultra import process_document as rag_ultra_process_document
from rag_ultra.document_loader import convert_document_to_text
from rag_ultra.image_converter import convert_document_pages_to_images
from rag_ultra.metadata_generator import generate_batch_metadata
from rag_ultra.document_summary import generate_document_summary, extract_document_details, assemble_document_metadata
from rag_ultra.utils import get_file_details

logger = logging.getLogger(__name__)

class ProcessingJob:
    """Represents a document processing job."""
    def __init__(self, job_id: str, document_id: str, filename: str, progress_callback: Callable[[str, int, int, Optional[str]], None]):
        self.id = job_id
        self.document_id = document_id
        self.filename = filename
        self.status = 'queued'  # queued, processing, completed, error
        self.progress = {
            'stage': 'Queued',
            'current': 0,
            'total': 0,
            'message': 'Waiting to start...'
        }
        self.logs: List[str] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.error: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        # Attributes not saved in cache previously
        self.temp_file_path: Optional[str] = None
        self.extraction_model: Optional[str] = None
        self.progress_callback = progress_callback
    
    def save_to_cache(self, session_id: str):
        """Saves the entire job state to a cache file (noop in diskless env)."""
        self.add_log("Skipping job save to cache in diskless environment.")
        pass

    @classmethod
    def load_from_cache(cls, job_id: str, session_id: str) -> Optional['ProcessingJob']:
        """Loads a job from a cache file (noop in diskless env)."""
        logger.info(f"Skipping job load from cache for job {job_id} (diskless env).")
        return None
    
    def add_log(self, message: str):
        """Add a log entry with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{timestamp} - {message}")
        logger.info(f"Job {self.id}: {message}")
    
    def update_progress(self, stage: str, current: int, total: int, message: Optional[str] = None):
        """Update job progress."""
        self.progress = {
            'stage': stage,
            'current': current,
            'total': total,
            'message': message or f"{stage} ({current}/{total})"
        }
        self.add_log(self.progress['message'])
        # Note: We don't save to cache here to avoid excessive I/O during rapid updates
        # The job will be saved at appropriate intervals by the calling code
        
        # DEBUG: Log before calling callback
        logger.info(f"[CALLBACK TRACE] Job {self.id}: About to call progress_callback with stage='{stage}', current={current}, total={total}")
        
        if hasattr(self, 'progress_callback') and self.progress_callback:
            logger.info(f"[CALLBACK TRACE] Job {self.id}: progress_callback exists, calling it now")
            self.progress_callback(stage, current, total, message)
        else:
            logger.error(f"[CALLBACK TRACE] Job {self.id}: NO PROGRESS CALLBACK! Cannot update session progress!")
    
    def complete(self, result: Dict[str, Any]):
        """Mark job as completed."""
        self.status = 'completed'
        self.result = result
        self.end_time = datetime.now()
        self.add_log("Processing completed successfully")
        # After completing, save the result to the cache.
        # We need the session_id, which isn't stored on the job.
        # This will be called from DocumentProcessor, which has the session_id.
    
    def fail(self, error: str):
        """Mark job as failed."""
        self.status = 'error'
        self.error = error
        self.end_time = datetime.now()
        self.add_log(f"Processing failed: {error}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API response and caching."""
        return {
            'id': self.id,
            'documentId': self.document_id,
            'filename': self.filename,
            'status': self.status,
            'progress': self.progress,
            'logs': self.logs,
            'startTime': self.start_time.isoformat(),
            'endTime': self.end_time.isoformat() if self.end_time else None,
            'error': self.error,
            'metadata': self.result,
            'temp_file_path': self.temp_file_path,
            'extraction_model': self.extraction_model,
        }


class DocumentProcessor:
    """Manages document processing for a session."""
    
    def __init__(self, session_id: str, max_concurrent_jobs: int = 1):
        self.session_id = session_id
        self.max_concurrent_jobs = max_concurrent_jobs
        self.jobs: Dict[str, ProcessingJob] = {}
        self.processing_queue: List[str] = []
        self.active_jobs: List[str] = []
        self._lock = asyncio.Lock()
        # Store a reference to the session progress update function
        self._update_session_progress = None
    
    def set_progress_callback(self, callback):
        """Set the session progress update callback function."""
        self._update_session_progress = callback
    
    def update_progress(self, stage: str, current: int, total: int, message: Optional[str] = None):
        """Update progress in the session progress store."""
        if self._update_session_progress:
            # Extract job_id from the message or use a default
            # This is a simple callback that will be called by ProcessingJob
            pass
    
    async def create_job(self, document_id: str, filename: str) -> ProcessingJob:
        """Create a new processing job."""
        job_id = document_id # Use the document_id as the job_id for simplicity
        
        logger.info(f"[CALLBACK TRACE] Creating job {job_id} for {filename}")
        
        # Create progress callback for this specific job
        def progress_callback(stage: str, current: int, total: int, message: Optional[str] = None):
            logger.info(f"[CALLBACK TRACE] Progress callback called for job {job_id}: {stage} ({current}/{total})")
            if self._update_session_progress:
                logger.info(f"[CALLBACK TRACE] Calling session progress update for job {job_id}")
                # Get the job to access its logs
                job = self.jobs.get(job_id)
                logs = job.logs if job else []
                self._update_session_progress(
                    self.session_id, job_id, 'processing', 
                    stage, current, total, logs, None
                )
            else:
                logger.error(f"[CALLBACK TRACE] No session progress updater available!")
        
        job = ProcessingJob(job_id, document_id, filename, progress_callback)
        logger.info(f"[CALLBACK TRACE] Job {job_id} created with progress callback")
        
        async with self._lock:
            self.jobs[job_id] = job
        
        # Save the initial state to cache immediately
        job.save_to_cache(self.session_id)
        
        # Initialize in session progress
        if self._update_session_progress:
            logger.info(f"[CALLBACK TRACE] Initializing session progress for job {job_id} as 'queued'")
            self._update_session_progress(
                self.session_id, job_id, 'queued',
                'Waiting to start...', 0, 0, job.logs, None
            )
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by ID, loading from cache if not in memory."""
        logger.info(f"[CALLBACK TRACE] get_job called for {job_id}")
        
        job = self.jobs.get(job_id)
        if not job:
            logger.info(f"[CALLBACK TRACE] Job {job_id} not in memory, trying cache")
            # If not in memory, try loading from cache
            job = ProcessingJob.load_from_cache(job_id, self.session_id)
            if job:
                logger.info(f"[CALLBACK TRACE] Job {job_id} loaded from cache")
                # Add to in-memory store if loaded
                async with self._lock:
                    self.jobs[job_id] = job
            else:
                logger.info(f"[CALLBACK TRACE] Job {job_id} not found in cache either")
        else:
            logger.info(f"[CALLBACK TRACE] Job {job_id} found in memory")
        
        # CRITICAL FIX: Ensure the job has a valid progress callback
        if job and hasattr(job, 'progress_callback'):
            logger.info(f"[CALLBACK TRACE] Resetting progress callback for job {job_id}")
            # Create progress callback for this specific job if it doesn't have one or if it's lost
            def progress_callback(stage: str, current: int, total: int, message: Optional[str] = None):
                logger.info(f"[CALLBACK TRACE] Reset callback invoked for job {job_id}: {stage} ({current}/{total})")
                if self._update_session_progress:
                    # Get the job to access its logs
                    current_job = self.jobs.get(job_id)
                    logs = current_job.logs if current_job else []
                    self._update_session_progress(
                        self.session_id, job_id, 'processing', 
                        stage, current, total, logs, None
                    )
                else:
                    logger.error(f"[CALLBACK TRACE] No session progress updater in reset callback!")
            
            # Always set/reset the progress callback to ensure it's valid
            job.progress_callback = progress_callback
            logger.info(f"[CALLBACK TRACE] Progress callback reset complete for job {job_id}")
        
        return job
    
    async def start_job_processing(self, job_id: str):
        """Adds a job to the processing queue and starts the queue processor."""
        job = await self.get_job(job_id)
        if not job:
            logger.error(f"Cannot start processing, job {job_id} not found.")
            return

        # Ensure job is in a state that can be started
        if job.status != 'queued':
            logger.warning(f"Job {job_id} is not in 'queued' state (state: {job.status}), cannot start processing.")
            return

        async with self._lock:
            if job_id not in self.processing_queue and job_id not in self.active_jobs:
                self.processing_queue.append(job_id)
                logger.info(f"Added job {job_id} to processing queue.")
            else:
                logger.info(f"Job {job_id} is already in queue or being processed.")
        
        # Trigger the queue processor to check if it can start this job
        asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Processes jobs from the queue if there is capacity."""
        async with self._lock:
            # Check if we can start more jobs
            while (len(self.active_jobs) < self.max_concurrent_jobs and 
                   self.processing_queue):
                job_id = self.processing_queue.pop(0)
                self.active_jobs.append(job_id)
                
                # Start processing in background
                asyncio.create_task(self._process_job(job_id))
    
    async def _process_job(self, job_id: str):
        """Process a single job."""
        job = await self.get_job(job_id) # Use get_job to ensure it's loaded
        if not job:
            logger.error(f"Could not process job {job_id}, as it could not be found or loaded.")
            return
        
        temp_file_path = None # Ensure it's defined for the finally block
        try:
            job.status = 'processing'
            job.add_log(f"Started processing {job.filename}")
            job.save_to_cache(self.session_id)  # Save the processing status change
            
            # Update session progress to processing immediately
            if self._update_session_progress:
                self._update_session_progress(
                    self.session_id, job_id, 'processing',
                    'Starting processing...', 0, 100, job.logs, None
                )
            
            # Small delay to ensure the status update is propagated
            await asyncio.sleep(0.1)
            
            # Get the stored attributes
            temp_file_path = getattr(job, 'temp_file_path', None)
            extraction_model = getattr(job, 'extraction_model', 'gpt-4.1-mini')
            
            if not temp_file_path or not os.path.exists(temp_file_path):
                error_msg = f"Document file not found at path: {temp_file_path}"
                job.fail(error_msg)
                job.save_to_cache(self.session_id) # Save failure state
                
                # Update session progress with error
                if self._update_session_progress:
                    self._update_session_progress(
                        self.session_id, job_id, 'error',
                        'File not found', 0, 0, job.logs, error_msg
                    )
                return

            # Get the API key for the selected model
            api_key = await self._get_api_key_for_model(extraction_model)

            # Create a thread-safe callback for progress updates
            loop = asyncio.get_running_loop()

            def thread_safe_progress_callback(stage: str, current: int, total: int, message: Optional[str] = None):
                # This function will be called from the background thread.
                # We need to use run_coroutine_threadsafe to schedule the
                # update on the main event loop.
                async def update_in_loop():
                    job.update_progress(stage, current, total, message)
                
                asyncio.run_coroutine_threadsafe(update_in_loop(), loop)

            # Use functools.partial to prepare the synchronous function with its arguments
            blocking_process_func = partial(
                rag_ultra_process_document,
                document_path=temp_file_path,
                model=extraction_model,
                api_key=api_key,
                filename=job.filename,
                callbacks={ # Pass the thread-safe callback
                    'progress': thread_safe_progress_callback
                }
            )

            # Run the blocking function in a separate thread
            result_metadata = await asyncio.to_thread(blocking_process_func)

            job.complete(result_metadata)
            job.save_to_cache(self.session_id)
            
            # Final update to session progress
            if self._update_session_progress:
                self._update_session_progress(
                    self.session_id, job_id, 'completed',
                    'Processing completed', 100, 100, job.logs, None
                )

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
            job.fail(str(e))
            job.save_to_cache(self.session_id)
            
            # Update session progress with error
            if self._update_session_progress:
                self._update_session_progress(
                    self.session_id, job_id, 'error',
                    'Processing failed', 0, 0, job.logs, str(e)
                )
            
        finally:
            # Clean up active job tracking
            async with self._lock:
                if job_id in self.active_jobs:
                    self.active_jobs.remove(job_id)
            
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    job.add_log(f"Cleaned up temporary file: {temp_file_path}")
                except OSError as e:
                    job.add_log(f"Error cleaning up temporary file {temp_file_path}: {e}")
            
            # Save final job state
            job.save_to_cache(self.session_id)
            
            # Trigger queue again to process next item
            await self._process_queue()

    async def _get_api_key_for_model(self, model: str) -> Optional[str]:
        """Determine the correct API key environment variable for a given model."""
        model_lower = model.lower()
        if 'gpt' in model_lower or 'openai' in model_lower:
            return os.environ.get('OPENAI_API_KEY')
        elif 'claude' in model_lower or 'anthropic' in model_lower:
            return os.environ.get('ANTHROPIC_API_KEY')
        elif 'grok' in model_lower or 'xai' in model_lower:
            return os.environ.get('XAI_API_KEY')
        elif 'deepseek' in model_lower:
            return os.environ.get('DEEPSEEK_API_KEY')
        elif 'cohere' in model_lower:
            return os.environ.get('COHERE_API_KEY')
        return None

    async def _simulate_processing(self, job: ProcessingJob):
        """Simulates document processing for testing purposes."""
        total_steps = 10
        for i in range(total_steps):
            job.update_progress("Simulating work", i + 1, total_steps)
            await asyncio.sleep(1)
        
        result = {"summary": f"This is a simulated summary for {job.filename}"}
        job.complete(result)
        job.save_to_cache(self.session_id)

    async def process_document(self, file_path: str, filename: str, model: str, 
                             api_key: str, job_id: str) -> Dict[str, Any]:
        """
        Processes a document using the RAG-Ultra SDK and updates progress via callbacks.
        This version is a placeholder and needs to be implemented.
        """
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found for processing.")

        # This is a placeholder for the actual RAG-Ultra processing logic.
        # It needs to be replaced with the real implementation that uses callbacks.
        
        # Simulate the processing stages from the original plan
        await self._simulate_processing(job)
        
        # In a real implementation, you would return the metadata from RAG-Ultra.
        return job.result


# Global storage for document processors by session
SESSION_PROCESSORS: Dict[str, DocumentProcessor] = {}


def get_or_create_processor(session_id: str, progress_callback=None) -> DocumentProcessor:
    """Get or create a document processor for the session."""
    global SESSION_PROCESSORS
    
    logger.info(f"[CALLBACK TRACE] get_or_create_processor called for session {session_id}, callback provided: {progress_callback is not None}")
    
    if session_id not in SESSION_PROCESSORS:
        logger.info(f"[CALLBACK TRACE] Creating new processor for session {session_id}")
        SESSION_PROCESSORS[session_id] = DocumentProcessor(session_id)
        
    processor = SESSION_PROCESSORS[session_id]
    
    # Set the progress callback if provided
    if progress_callback:
        logger.info(f"[CALLBACK TRACE] Setting progress callback on processor for session {session_id}")
        processor.set_progress_callback(progress_callback)
    else:
        logger.warning(f"[CALLBACK TRACE] No progress callback provided for session {session_id}")
        
    return processor


async def cleanup_old_processors():
    """Clean up processors for inactive sessions (to be implemented)."""
    # TODO: Implement cleanup logic based on last activity time
    pass 