/**
 * Document Processor Hook
 * Combines document context with API operations
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { useDocuments } from '../contexts/DocumentContext';
import documentApi from '../services/documentApi';
import documentStorage from '../services/documentStorage';

export function useDocumentProcessor() {
  const { 
    documents, 
    processingJobs, 
    useDocumentContext, 
    selectedDocumentIds,
    addDocument,
    updateDocument,
    deleteDocument: removeDocument,
    addJob: addProcessingJob,
    updateJob: updateProcessingJob,
    deleteJob: deleteProcessingJob,
    toggleDocumentContext,
    setSelectedDocuments,
  } = useDocuments();
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const pollIntervalRef = useRef(null);
  
  // State for handling duplicate file uploads
  const [duplicateFiles, setDuplicateFiles] = useState([]); // Array of { file, existingDocument }
  const [uploadContinuation, setUploadContinuation] = useState(null); // { filesToUpload, extractionModel }
  
  // Get session ID from localStorage
  const getSessionId = useCallback(() => {
    let sessionId = localStorage.getItem('sessionId');
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      localStorage.setItem('sessionId', sessionId);
    }
    return sessionId;
  }, []);
  
  const pollSessionProgress = useCallback(async () => {
    if (pollIntervalRef.current) {
      console.log('Polling is already active.');
      return;
    }

    const sessionId = getSessionId();
    const pollInterval = setInterval(async () => {
      try {
        const sessionData = await documentApi.getSessionProgress(sessionId);
        const jobs = sessionData.jobs || {};
        
        console.log(`Session ${sessionId} progress:`, {
          jobCount: Object.keys(jobs).length,
          activeJobs: Object.keys(jobs).filter(jobId => 
            ['queued', 'processing'].includes(jobs[jobId]?.status)
          ).length,
        });
        
        for (const [jobId, progressData] of Object.entries(jobs)) {
          await updateProcessingJob(jobId, {
            status: progressData.status,
            progress: {
              stage: progressData.stage,
              current: progressData.current,
              total: progressData.total,
              message: progressData.stage || 'Processing...'
            },
            logs: progressData.logs || [],
            error: progressData.error,
          });
          
          const job = await documentStorage.getProcessingJob(jobId);
          if (job) {
            await documentStorage.updateProcessingJob(jobId, {
              ...job,
              status: progressData.status,
              progress: {
                stage: progressData.stage,
                current: progressData.current,
                total: progressData.total,
                message: progressData.stage || 'Processing...'
              },
              logs: progressData.logs || [],
              error: progressData.error,
            });
          }
          
          if (progressData.status === 'completed') {
            try {
              const fullStatus = await documentApi.getJobStatus(jobId, sessionId);
              if (fullStatus.metadata) {
                const job = await documentStorage.getProcessingJob(jobId);
                if (job && job.documentId) {
                  const documents = await documentStorage.getAllDocuments();
                  const doc = documents.find(d => d.id === job.documentId);
                  if (doc) {
                    const updatedDoc = {
                      status: 'completed',
                      processedDate: new Date().toISOString(),
                      metadata: fullStatus.metadata,
                    };
                    
                    await updateDocument(job.documentId, updatedDoc);
                    await documentStorage.updateDocument(job.documentId, {
                      ...doc,
                      ...updatedDoc,
                    });
                  }
                }
              }
            } catch (err) {
              console.error('Error getting final document metadata:', err);
            }
            
            await deleteProcessingJob(jobId);
            await documentApi.clearJobProgress(jobId, sessionId);
          }
        }
        
        const activeJobs = Object.values(jobs).filter(job => 
          ['queued', 'processing'].includes(job.status)
        );
        
        if (activeJobs.length === 0) {
          console.log('No active jobs, stopping session polling');
          clearInterval(pollInterval);
          pollIntervalRef.current = null;
        }
        
      } catch (err) {
        console.error('Error polling session progress:', err);
      }
    }, 2000);
    
    pollIntervalRef.current = pollInterval;
    return pollInterval;
  }, [updateProcessingJob, updateDocument, deleteProcessingJob, getSessionId]);

  /**
   * Main function to upload files, now handles batches and duplicates.
   */
  const uploadDocument = useCallback(async (files, extractionModel) => {
    setError(null);
    //setIsProcessing(true); // Don't set here, let the continuation handle it

    const filesToUpload = Array.from(files);
    const existingDocs = await documentStorage.getAllDocuments();
    const foundDuplicates = [];
    const newFiles = [];

    for (const file of filesToUpload) {
      const existingDoc = existingDocs.find(doc => doc.filename === file.name);
      if (existingDoc) {
        foundDuplicates.push({ file, existingDoc });
      } else {
        newFiles.push(file);
      }
    }
    
    const _uploadSingleFile = async (file, overwrite = false) => {
      try {
        const sessionId = getSessionId();
        const documentId = crypto.randomUUID();
        
        const uploadResult = await documentApi.uploadDocument(
          file,
          extractionModel,
          sessionId,
          documentId,
          overwrite
        );

        const newDocument = {
          id: documentId,
          filename: file.name,
          fileSize: file.size,
          uploadDate: new Date().toISOString(),
          status: 'processing',
          extractionModel,
          sessionId,
        };
        
        await addDocument(newDocument);
        await documentStorage.addDocument(newDocument);
        
        const job = {
          id: uploadResult.jobId,
          documentId: documentId,
          status: 'queued',
          progress: { stage: 'Queued', current: 0, total: 0, message: 'Waiting to start...' },
          logs: [],
          startTime: new Date().toISOString(),
        };
        
        await addProcessingJob(job);
        await documentStorage.addProcessingJob(job);
        
        await documentApi.startProcessing(uploadResult.jobId, sessionId);
        
        await updateProcessingJob(uploadResult.jobId, {
          status: 'processing',
          progress: { stage: 'Starting...', current: 0, total: 100, message: 'Starting processing...' }
        });
        
        pollSessionProgress();
      } catch (err) {
        console.error(`Error uploading file ${file.name}:`, err);
        setError(`Failed to upload ${file.name}.`);
      }
    };

    if (foundDuplicates.length > 0) {
      setDuplicateFiles(foundDuplicates);
      setUploadContinuation(() => async (choices) => {
        try {
          setIsProcessing(true);
          // First, upload all the new files that weren't duplicates
          for (const file of newFiles) {
            await _uploadSingleFile(file, false);
          }
          // Then, handle the duplicates based on user choices
          for (const duplicate of foundDuplicates) {
            if (choices[duplicate.file.name] === 'replace') {
              await _uploadSingleFile(duplicate.file, true);
            }
          }
        } finally {
          setIsProcessing(false);
          setDuplicateFiles([]);
          setUploadContinuation(null);
        }
      });
    } else {
      // No duplicates, just upload all files
      try {
        setIsProcessing(true);
        for (const file of newFiles) {
          await _uploadSingleFile(file, false);
        }
      } finally {
        setIsProcessing(false);
      }
    }
  }, [addDocument, addProcessingJob, getSessionId, pollSessionProgress, updateProcessingJob]);

  const handleConfirmDuplicateUpload = useCallback(async (choices) => {
    if (uploadContinuation) {
      await uploadContinuation(choices);
    }
  }, [uploadContinuation]);
  
  const cancelDuplicateUpload = useCallback(() => {
    setDuplicateFiles([]);
    setUploadContinuation(null);
    setIsProcessing(false);
  }, []);

  const startProcessing = useCallback(async (jobId, documentId) => {
    const sessionId = getSessionId();
    try {
      await documentApi.startProcessing(jobId, sessionId);
      await updateProcessingJob(jobId, { status: 'processing' });
      pollSessionProgress();
    } catch (err) {
      console.error('Error starting processing:', err);
      setError('Failed to start processing.');
    }
  }, [getSessionId, updateProcessingJob, pollSessionProgress]);
  
  const syncDocuments = async () => {
    const sessionId = getSessionId();
    const completedDocs = documents.filter(doc => doc.status === 'completed' && doc.metadata);
    
    if (completedDocs.length > 0) {
      console.log(`Syncing ${completedDocs.length} completed documents to backend...`);
      try {
        await documentApi.syncDocuments(sessionId, completedDocs);
        console.log('Document sync successful.');
      } catch (err) {
        console.error('Failed to sync documents:', err);
        setError('Could not sync documents with the server.');
      }
    }
  };

  const deleteDocument = useCallback(async (documentId) => {
    const sessionId = getSessionId();
    try {
      const job = processingJobs.find(j => j.documentId === documentId);
      await documentApi.deleteDocument(documentId, sessionId);
      if (job) {
        await documentApi.clearJobProgress(job.id, sessionId);
      }
      await removeDocument(documentId);
      await documentStorage.deleteDocument(documentId);
      if (job) {
        await deleteProcessingJob(job.id);
        await documentStorage.deleteProcessingJob(job.id);
      }
    } catch (err) {
      console.error('Error deleting document:', err);
      setError('Failed to delete document. Please try again.');
    }
  }, [processingJobs, removeDocument, deleteProcessingJob, getSessionId]);
  
  const clearAllDocuments = useCallback(async () => {
    try {
      const allDocs = await documentStorage.getAllDocuments();
      for (const doc of allDocs) {
        await deleteDocument(doc.id);
      }
    } catch (error) {
      console.error('Error clearing all documents:', error);
      throw error;
    }
  }, [deleteDocument]);

  useEffect(() => {
    const loadFromDb = async () => {
      const dbDocs = await documentStorage.getAllDocuments();
      const dbJobs = await documentStorage.getAllProcessingJobs();
      dbDocs.forEach(doc => addDocument(doc));
      dbJobs.forEach(job => addProcessingJob(job));
      syncDocuments();
    };
    
    loadFromDb();
  }, []);
  
  return {
    documents,
    processingJobs,
    useDocumentContext,
    selectedDocumentIds,
    isProcessing,
    error,
    uploadDocument,
    duplicateFiles,
    handleConfirmDuplicateUpload,
    cancelDuplicateUpload,
    startProcessing,
    deleteDocument,
    clearAllDocuments,
    toggleDocumentContext,
    setSelectedDocuments,
    startPolling: pollSessionProgress,
    syncDocuments,
  };
}

export default useDocumentProcessor; 