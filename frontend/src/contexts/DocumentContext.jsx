/**
 * Document Context
 * Provides global state management for documents and processing jobs
 */

import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';
import documentStorage from '../services/documentStorage';
import documentApi from '../services/documentApi';
import { v4 as uuidv4 } from 'uuid';

// Initial state
const initialState = {
  documents: [],
  processingJobs: [],
  selectedDocumentIds: [],
  isLoading: true,
  error: null,
  syncStatus: 'idle', // 'idle', 'syncing', 'synced', 'error'
  storageInfo: {
    used: 0,
    total: 0,
    percentage: 0,
    usedMB: '0',
    totalMB: 'Unknown'
  }
};

// Action types
const ActionTypes = {
  SET_DOCUMENTS: 'SET_DOCUMENTS',
  ADD_DOCUMENT: 'ADD_DOCUMENT',
  UPDATE_DOCUMENT: 'UPDATE_DOCUMENT',
  DELETE_DOCUMENT: 'DELETE_DOCUMENT',
  SET_PROCESSING_JOBS: 'SET_PROCESSING_JOBS',
  ADD_JOB: 'ADD_JOB',
  UPDATE_JOB: 'UPDATE_JOB',
  DELETE_JOB: 'DELETE_JOB',
  SET_SELECTED_DOCUMENTS: 'SET_SELECTED_DOCUMENTS',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_SYNC_STATUS: 'SET_SYNC_STATUS',
  SET_STORAGE_INFO: 'SET_STORAGE_INFO',
  CLEAR_ALL: 'CLEAR_ALL'
};

// Reducer
function documentReducer(state, action) {
  switch (action.type) {
    case ActionTypes.SET_DOCUMENTS:
      return { ...state, documents: action.payload };
      
    case ActionTypes.ADD_DOCUMENT:
      const docExists = state.documents.some(doc => doc.id === action.payload.id);
      if (docExists) {
        return {
          ...state,
          documents: state.documents.map(doc =>
            doc.id === action.payload.id ? { ...doc, ...action.payload } : doc
          )
        };
      }
      return { ...state, documents: [...state.documents, action.payload] };
      
    case ActionTypes.UPDATE_DOCUMENT:
      return {
        ...state,
        documents: state.documents.map(doc =>
          doc.id === action.payload.id ? { ...doc, ...action.payload.updates } : doc
        )
      };
      
    case ActionTypes.DELETE_DOCUMENT:
      return {
        ...state,
        documents: state.documents.filter(doc => doc.id !== action.payload),
        selectedDocumentIds: state.selectedDocumentIds.filter(id => id !== action.payload)
      };
      
    case ActionTypes.SET_PROCESSING_JOBS:
      return { ...state, processingJobs: action.payload };
      
    case ActionTypes.ADD_JOB:
      const jobExists = state.processingJobs.some(job => job.id === action.payload.id);
      if (jobExists) {
        return {
          ...state,
          processingJobs: state.processingJobs.map(job =>
            job.id === action.payload.id ? { ...job, ...action.payload } : job
          )
        };
      }
      return { ...state, processingJobs: [...state.processingJobs, action.payload] };
      
    case ActionTypes.UPDATE_JOB:
      console.log(`[Reducer] UPDATE_JOB for ${action.payload.id}:`, action.payload.updates);
      const updatedJobs = state.processingJobs.map(job =>
        job.id === action.payload.id ? { ...job, ...action.payload.updates } : job
      );
      console.log(`[Reducer] Updated jobs:`, updatedJobs);
      return {
        ...state,
        processingJobs: updatedJobs
      };
      
    case ActionTypes.DELETE_JOB:
      return {
        ...state,
        processingJobs: state.processingJobs.filter(job => job.id !== action.payload)
      };
      
    case ActionTypes.SET_SELECTED_DOCUMENTS:
      return { ...state, selectedDocumentIds: Array.from(new Set(action.payload)) };
      
    case ActionTypes.SET_LOADING:
      return { ...state, isLoading: action.payload };
      
    case ActionTypes.SET_ERROR:
      return { ...state, error: action.payload };
      
    case ActionTypes.SET_SYNC_STATUS:
      return { ...state, syncStatus: action.payload };
      
    case ActionTypes.SET_STORAGE_INFO:
      return { ...state, storageInfo: action.payload };
      
    case ActionTypes.CLEAR_ALL:
      return { ...initialState, isLoading: false };
      
    default:
      return state;
  }
}

// Create context
const DocumentContext = createContext(null);

// Provider component
export function DocumentProvider({ children }) {
  const [state, dispatch] = useReducer(documentReducer, initialState);
  
  // Load initial data from IndexedDB
  useEffect(() => {
    loadInitialData();
  }, []);
  
  // Update storage info periodically
  useEffect(() => {
    updateStorageInfo();
    const interval = setInterval(updateStorageInfo, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);
  
  // Get session ID (same logic as in App.jsx)
  const getSessionId = () => {
    let id = localStorage.getItem('sessionId');
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem('sessionId', id);
    }
    return id;
  };

  // Sync documents with backend
  const syncDocumentsWithBackend = async (documents) => {
    try {
      dispatch({ type: ActionTypes.SET_SYNC_STATUS, payload: 'syncing' });
      
      const sessionId = getSessionId();
      const completedDocs = documents.filter(doc => doc.status === 'completed');
      
      console.log(`Syncing ${completedDocs.length} completed documents with backend for session ${sessionId}`);
      
      const result = await documentApi.syncDocuments(sessionId, completedDocs);
      
      dispatch({ type: ActionTypes.SET_SYNC_STATUS, payload: 'synced' });
      console.log(`Successfully synced ${result.synced_count} documents with backend`);
      
    } catch (error) {
      console.error('Error syncing documents with backend:', error);
      dispatch({ type: ActionTypes.SET_SYNC_STATUS, payload: 'error' });
      dispatch({ type: ActionTypes.SET_ERROR, payload: `Failed to sync documents: ${error.message}` });
    }
  };

  // Load data from IndexedDB
  const loadInitialData = async () => {
    try {
      dispatch({ type: ActionTypes.SET_LOADING, payload: true });
      
      const [documents, jobs] = await Promise.all([
        documentStorage.getAllDocuments(),
        documentStorage.getAllProcessingJobs()
      ]);
      
      dispatch({ type: ActionTypes.SET_DOCUMENTS, payload: documents });
      dispatch({ type: ActionTypes.SET_PROCESSING_JOBS, payload: jobs });
      
      // Set all completed documents as selected by default
      const completedDocIds = documents
        .filter(doc => doc.status === 'completed')
        .map(doc => doc.id);
      dispatch({ type: ActionTypes.SET_SELECTED_DOCUMENTS, payload: completedDocIds });
      
      // Automatically sync completed documents with backend
      await syncDocumentsWithBackend(documents);
      
    } catch (error) {
      console.error('Error loading initial data:', error);
      dispatch({ type: ActionTypes.SET_ERROR, payload: error.message });
    } finally {
      dispatch({ type: ActionTypes.SET_LOADING, payload: false });
    }
  };
  
  // Update storage info
  const updateStorageInfo = async () => {
    try {
      const info = await documentStorage.checkStorageQuota();
      dispatch({ type: ActionTypes.SET_STORAGE_INFO, payload: info });
    } catch (error) {
      console.error('Error updating storage info:', error);
    }
  };
  
  // Actions
  const actions = {
    // Document actions
    addDocument: useCallback(async (document) => {
      try {
        const docWithId = { ...document, id: document.id || uuidv4() };
        await documentStorage.addDocument(docWithId);
        dispatch({ type: ActionTypes.ADD_DOCUMENT, payload: docWithId });
        
        // Auto-select if completed
        if (docWithId.status === 'completed') {
          dispatch({
            type: ActionTypes.SET_SELECTED_DOCUMENTS,
            payload: Array.from(new Set([...state.selectedDocumentIds, docWithId.id]))
          });
        }
        
        updateStorageInfo();
        return docWithId.id;
      } catch (error) {
        console.error('Error adding document:', error);
        throw error;
      }
    }, [state.selectedDocumentIds]),
    
    updateDocument: useCallback(async (id, updates) => {
      try {
        await documentStorage.updateDocument(id, updates);
        dispatch({ type: ActionTypes.UPDATE_DOCUMENT, payload: { id, updates } });
        
        // Auto-select if just completed
        if (updates.status === 'completed' && !state.selectedDocumentIds.includes(id)) {
          dispatch({
            type: ActionTypes.SET_SELECTED_DOCUMENTS,
            payload: Array.from(new Set([...state.selectedDocumentIds, id]))
          });
        }
        
        // If document was just completed, sync with backend
        if (updates.status === 'completed') {
          const updatedDocuments = await documentStorage.getAllDocuments();
          await syncDocumentsWithBackend(updatedDocuments);
        }
        
        updateStorageInfo();
      } catch (error) {
        console.error('Error updating document:', error);
        throw error;
      }
    }, [state.selectedDocumentIds]),
    
    deleteDocument: useCallback(async (id) => {
      try {
        await documentStorage.deleteDocument(id);
        dispatch({ type: ActionTypes.DELETE_DOCUMENT, payload: id });
        updateStorageInfo();
      } catch (error) {
        console.error('Error deleting document:', error);
        throw error;
      }
    }, []),
    
    // Job actions
    addJob: useCallback(async (job) => {
      try {
        const jobWithId = { ...job, id: job.id || uuidv4() };
        await documentStorage.addProcessingJob(jobWithId);
        dispatch({ type: ActionTypes.ADD_JOB, payload: jobWithId });
        return jobWithId.id;
      } catch (error) {
        console.error('Error adding job:', error);
        throw error;
      }
    }, []),
    
    updateJob: useCallback(async (id, updates) => {
      try {
        console.log(`[DocumentContext] Updating job ${id} with:`, updates);
        await documentStorage.updateProcessingJob(id, updates);
        dispatch({ type: ActionTypes.UPDATE_JOB, payload: { id, updates } });
      } catch (error) {
        console.error('Error updating job:', error);
        throw error;
      }
    }, []),
    
    deleteJob: useCallback(async (id) => {
      try {
        await documentStorage.deleteProcessingJob(id);
        dispatch({ type: ActionTypes.DELETE_JOB, payload: id });
      } catch (error) {
        console.error('Error deleting job:', error);
        throw error;
      }
    }, []),
    
    // Document sync actions
    manualSyncDocuments: useCallback(async () => {
      try {
        const documents = await documentStorage.getAllDocuments();
        await syncDocumentsWithBackend(documents);
      } catch (error) {
        console.error('Error manually syncing documents:', error);
        throw error;
      }
    }, []),
    
    setSelectedDocuments: useCallback((selectedIds) => {
      dispatch({ type: ActionTypes.SET_SELECTED_DOCUMENTS, payload: selectedIds });
    }, []),
    
    // Utility actions
    clearAllData: useCallback(async () => {
      try {
        await documentStorage.clearAllData();
        dispatch({ type: ActionTypes.CLEAR_ALL });
        updateStorageInfo();
      } catch (error) {
        console.error('Error clearing all data:', error);
        throw error;
      }
    }, []),
    
    exportData: useCallback(async () => {
      try {
        return await documentStorage.exportData();
      } catch (error) {
        console.error('Error exporting data:', error);
        throw error;
      }
    }, []),
    
    importData: useCallback(async (data) => {
      try {
        await documentStorage.importData(data);
        await loadInitialData();
      } catch (error) {
        console.error('Error importing data:', error);
        throw error;
      }
    }, []),
    
    refreshData: useCallback(async () => {
      await loadInitialData();
    }, []),
  };
  
  const value = {
    ...state,
    ...actions
  };
  
  return (
    <DocumentContext.Provider value={value}>
      {children}
    </DocumentContext.Provider>
  );
}

// Custom hook to use the document context
export function useDocuments() {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocuments must be used within a DocumentProvider');
  }
  return context;
}

export default DocumentContext; 