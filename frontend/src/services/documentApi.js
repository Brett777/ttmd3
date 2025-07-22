/**
 * Document API Service
 * Handles communication with backend document processing endpoints
 */

import axios from 'axios';

// Get base URL from environment or use default
const BASE_URL = import.meta.env.VITE_API_URL || '';

// Create axios instance with default config
const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Document API service
const documentApi = {
  /**
   * Upload a document for processing
   * @param {File} file - The file to upload
   * @param {string} extractionModel - Model to use for extraction
   * @param {string} sessionId - Session ID
   * @param {string} documentId - The client-generated unique ID for the document
   * @param {boolean} overwrite - Whether to overwrite an existing file with the same name
   * @returns {Promise<Object>} Upload response with job ID
   */
  async uploadDocument(file, extractionModel, sessionId, documentId, overwrite = false) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('extractionModel', extractionModel);
    formData.append('session_id', sessionId);
    formData.append('document_id', documentId);
    formData.append('overwrite', overwrite);
    
    const response = await api.post('api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    return response.data;
  },
  
  /**
   * Get processing job status
   * @param {string} jobId - Job ID
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Job status
   */
  async getJobStatus(jobId, sessionId) {
    const response = await api.get(`api/documents/process/${jobId}/status`, {
      params: { session_id: sessionId }
    });
    
    return response.data;
  },
  
  /**
   * Get progress for all jobs in a session (NEW - fast and lightweight)
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Session progress with all jobs
   */
  async getSessionProgress(sessionId) {
    const response = await api.get(`api/documents/progress/${sessionId}`);
    return response.data;
  },
  
  /**
   * Start processing a document
   * @param {string} jobId - Job ID
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Start response
   */
  async startProcessing(jobId, sessionId) {
    const response = await api.post(`api/documents/process/${jobId}/start`, {
      session_id: sessionId,
    });
    
    return response.data;
  },
  
  /**
   * List all documents for a session
   * @param {string} sessionId - Session ID
   * @returns {Promise<Array>} Array of documents
   */
  async listDocuments(sessionId) {
    const response = await api.get(`api/documents/${sessionId}`);
    return response.data.documents;
  },
  
  /**
   * Delete a document
   * @param {string} documentId - Document ID
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object>} Delete response
   */
  async deleteDocument(documentId, sessionId) {
    const response = await api.delete(`api/documents/${documentId}`, {
      params: { session_id: sessionId }
    });
    
    return response.data;
  },
  
  /**
   * Clears a job from the session's progress tracking on the backend.
   * @param {string} jobId - The ID of the job to clear.
   * @param {string} sessionId - The current session ID.
   * @returns {Promise<Object>} The response from the server.
   */
  async clearJobProgress(jobId, sessionId) {
    const response = await api.delete(`api/documents/progress/${jobId}`, {
      params: { session_id: sessionId }
    });
    return response.data;
  },
  
  /**
   * Poll job status until completion
   * @param {string} jobId - Job ID
   * @param {string} sessionId - Session ID
   * @param {Function} onProgress - Progress callback
   * @param {number} interval - Polling interval in ms
   * @returns {Promise<Object>} Final job status
   */
  async pollJobStatus(jobId, sessionId, onProgress, interval = 2000) {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getJobStatus(jobId, sessionId);
          
          if (onProgress) {
            onProgress(status);
          }
          
          if (status.status === 'completed') {
            resolve(status);
          } else if (status.status === 'error') {
            reject(new Error(status.error || 'Processing failed'));
          } else {
            // Continue polling
            setTimeout(poll, interval);
          }
        } catch (error) {
          reject(error);
        }
      };
      
      // Start polling
      poll();
    });
  },
  
  /**
   * Upload and process a document (complete flow)
   * @param {File} file - The file to upload
   * @param {string} extractionModel - Model to use for extraction
   * @param {string} sessionId - Session ID
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<Object>} Processed document
   */
  async uploadAndProcess(file, extractionModel, sessionId, onProgress) {
    try {
      // Step 1: Upload the file
      const documentId = crypto.randomUUID(); // Generate ID here for this specific flow
      const uploadResult = await this.uploadDocument(file, extractionModel, sessionId, documentId);
      const { jobId } = uploadResult;
      
      if (onProgress) {
        onProgress({
          status: 'uploaded',
          message: 'File uploaded successfully',
          jobId,
          documentId
        });
      }
      
      // Step 2: Start processing
      await this.startProcessing(jobId, sessionId);
      
      if (onProgress) {
        onProgress({
          status: 'processing',
          message: 'Processing started',
          jobId,
          documentId
        });
      }
      
      // Step 3: Poll for completion
      const finalStatus = await this.pollJobStatus(jobId, sessionId, (status) => {
        if (onProgress) {
          onProgress({
            ...status,
            documentId
          });
        }
      });
      
      // Step 4: Get the processed document
      const documents = await this.listDocuments(sessionId);
      const processedDoc = documents.find(doc => doc.id === documentId);
      
      if (!processedDoc) {
        throw new Error('Processed document not found');
      }
      
      return processedDoc;
      
    } catch (error) {
      console.error('Error in upload and process:', error);
      throw error;
    }
  },

  /**
   * Sync documents with backend for analysis
   * @param {string} sessionId - Session ID
   * @param {Array} documents - Array of document metadata objects
   * @returns {Promise<Object>} Sync response
   */
  async syncDocuments(sessionId, documents) {
    const response = await api.post('api/documents/sync', {
      session_id: sessionId,
      documents: documents
    });
    
    return response.data;
  }
};

export default documentApi; 