/**
 * Document Storage Service
 * Manages document storage in IndexedDB using Dexie.js
 */

import Dexie from 'dexie';

class DocumentStorageService extends Dexie {
  constructor() {
    super('RAGUltraDocuments');
    
    // Define database schema
    this.version(1).stores({
      documents: 'id, filename, uploadDate, status, extractionModel',
      processingJobs: 'id, documentId, status, startTime'
    });
    
    // Define table references
    this.documents = this.table('documents');
    this.processingJobs = this.table('processingJobs');
  }
  
  /**
   * Add a new document to storage
   * @param {Object} document - Document object with metadata
   * @returns {Promise<string>} Document ID
   */
  async addDocument(document) {
    try {
      const id = await this.documents.put(document);
      console.log('Document added/updated in IndexedDB:', id);
      return id;
    } catch (error) {
      console.error('Error adding document:', error);
      throw error;
    }
  }
  
  /**
   * Update an existing document
   * @param {string} id - Document ID
   * @param {Object} updates - Fields to update
   * @returns {Promise<number>} Number of updated records
   */
  async updateDocument(id, updates) {
    try {
      const count = await this.documents.update(id, updates);
      console.log('Document updated:', id);
      return count;
    } catch (error) {
      console.error('Error updating document:', error);
      throw error;
    }
  }
  
  /**
   * Get a document by ID
   * @param {string} id - Document ID
   * @returns {Promise<Object|undefined>} Document object or undefined
   */
  async getDocument(id) {
    try {
      return await this.documents.get(id);
    } catch (error) {
      console.error('Error getting document:', error);
      throw error;
    }
  }
  
  /**
   * Get all documents
   * @returns {Promise<Array>} Array of all documents
   */
  async getAllDocuments() {
    try {
      return await this.documents.toArray();
    } catch (error) {
      console.error('Error getting all documents:', error);
      throw error;
    }
  }
  
  /**
   * Get completed documents only
   * @returns {Promise<Array>} Array of completed documents
   */
  async getCompletedDocuments() {
    try {
      return await this.documents
        .where('status')
        .equals('completed')
        .toArray();
    } catch (error) {
      console.error('Error getting completed documents:', error);
      throw error;
    }
  }
  
  /**
   * Delete a document
   * @param {string} id - Document ID
   * @returns {Promise<void>}
   */
  async deleteDocument(id) {
    try {
      await this.documents.delete(id);
      console.log('Document deleted:', id);
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  }
  
  /**
   * Add a processing job
   * @param {Object} job - Processing job object
   * @returns {Promise<string>} Job ID
   */
  async addProcessingJob(job) {
    try {
      const id = await this.processingJobs.put(job);
      console.log('Processing job added/updated:', id);
      return id;
    } catch (error) {
      console.error('Error adding processing job:', error);
      throw error;
    }
  }
  
  /**
   * Update a processing job
   * @param {string} id - Job ID
   * @param {Object} updates - Fields to update
   * @returns {Promise<number>} Number of updated records
   */
  async updateProcessingJob(id, updates) {
    try {
      const count = await this.processingJobs.update(id, updates);
      console.log('Processing job updated:', id);
      return count;
    } catch (error) {
      console.error('Error updating processing job:', error);
      throw error;
    }
  }
  
  /**
   * Get all processing jobs
   * @returns {Promise<Array>} Array of all processing jobs
   */
  async getAllProcessingJobs() {
    try {
      return await this.processingJobs.toArray();
    } catch (error) {
      console.error('Error getting processing jobs:', error);
      throw error;
    }
  }
  
  /**
   * Get a processing job by ID
   * @param {string} id - Job ID
   * @returns {Promise<Object|undefined>} Job object or undefined
   */
  async getProcessingJob(id) {
    try {
      return await this.processingJobs.get(id);
    } catch (error) {
      console.error('Error getting processing job:', error);
      throw error;
    }
  }
  
  /**
   * Get active processing jobs
   * @returns {Promise<Array>} Array of active jobs
   */
  async getActiveProcessingJobs() {
    try {
      return await this.processingJobs
        .where('status')
        .anyOf(['queued', 'processing'])
        .toArray();
    } catch (error) {
      console.error('Error getting active jobs:', error);
      throw error;
    }
  }
  
  /**
   * Delete a processing job
   * @param {string} id - Job ID
   * @returns {Promise<void>}
   */
  async deleteProcessingJob(id) {
    try {
      await this.processingJobs.delete(id);
      console.log('Processing job deleted:', id);
    } catch (error) {
      console.error('Error deleting processing job:', error);
      throw error;
    }
  }
  
  /**
   * Check storage quota and usage
   * @returns {Promise<Object>} Storage info {used, total, percentage}
   */
  async checkStorageQuota() {
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        const used = estimate.usage || 0;
        const total = estimate.quota || 0;
        const percentage = total > 0 ? (used / total) * 100 : 0;
        
        return {
          used,
          total,
          percentage,
          usedMB: (used / (1024 * 1024)).toFixed(2),
          totalMB: (total / (1024 * 1024)).toFixed(2)
        };
      }
      
      // Fallback if Storage API not available
      return {
        used: 0,
        total: 0,
        percentage: 0,
        usedMB: '0',
        totalMB: 'Unknown'
      };
    } catch (error) {
      console.error('Error checking storage quota:', error);
      return {
        used: 0,
        total: 0,
        percentage: 0,
        usedMB: '0',
        totalMB: 'Unknown'
      };
    }
  }
  
  /**
   * Clear all data (use with caution!)
   * @returns {Promise<void>}
   */
  async clearAllData() {
    try {
      await this.documents.clear();
      await this.processingJobs.clear();
      console.log('All data cleared from IndexedDB');
    } catch (error) {
      console.error('Error clearing data:', error);
      throw error;
    }
  }
  
  /**
   * Get database size estimate
   * @returns {Promise<number>} Estimated size in bytes
   */
  async getDatabaseSize() {
    try {
      let totalSize = 0;
      
      // Estimate document sizes
      const documents = await this.documents.toArray();
      documents.forEach(doc => {
        // Rough estimate of document size
        totalSize += JSON.stringify(doc).length;
      });
      
      // Estimate job sizes
      const jobs = await this.processingJobs.toArray();
      jobs.forEach(job => {
        totalSize += JSON.stringify(job).length;
      });
      
      return totalSize;
    } catch (error) {
      console.error('Error calculating database size:', error);
      return 0;
    }
  }
  
  /**
   * Export all data (for backup)
   * @returns {Promise<Object>} All database data
   */
  async exportData() {
    try {
      const documents = await this.documents.toArray();
      const processingJobs = await this.processingJobs.toArray();
      
      return {
        version: 1,
        exportDate: new Date().toISOString(),
        data: {
          documents,
          processingJobs
        }
      };
    } catch (error) {
      console.error('Error exporting data:', error);
      throw error;
    }
  }
  
  /**
   * Import data (restore from backup)
   * @param {Object} data - Data to import
   * @returns {Promise<void>}
   */
  async importData(data) {
    try {
      if (data.version !== 1) {
        throw new Error('Incompatible data version');
      }
      
      // Clear existing data
      await this.clearAllData();
      
      // Import documents
      if (data.data.documents && data.data.documents.length > 0) {
        await this.documents.bulkAdd(data.data.documents);
      }
      
      // Import jobs
      if (data.data.processingJobs && data.data.processingJobs.length > 0) {
        await this.processingJobs.bulkAdd(data.data.processingJobs);
      }
      
      console.log('Data imported successfully');
    } catch (error) {
      console.error('Error importing data:', error);
      throw error;
    }
  }
}

// Create and export a singleton instance
const documentStorage = new DocumentStorageService();

// Open the database
documentStorage.open().catch(err => {
  console.error('Failed to open IndexedDB:', err);
});

export default documentStorage;

/**
 * Converts bytes to a human-readable file size string.
 * @param {number} bytes - The file size in bytes.
 * @param {boolean} si - True for metric (1000), false for binary (1024).
 * @param {number} dp - Decimal places.
 * @returns {string} Human-readable file size.
 */
export function humanFileSize(bytes, si = false, dp = 1) {
  const thresh = si ? 1000 : 1024;

  if (Math.abs(bytes) < thresh) {
    return bytes + ' B';
  }

  const units = si 
    ? ['kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'] 
    : ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB'];
  let u = -1;
  const r = 10**dp;

  do {
    bytes /= thresh;
    ++u;
  } while (Math.round(Math.abs(bytes) * r) / r >= thresh && u < units.length - 1);

  return bytes.toFixed(dp) + ' ' + units[u];
} 