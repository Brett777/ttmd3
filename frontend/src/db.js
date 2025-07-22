import Dexie from 'dexie';

export const db = new Dexie('ChatDatabase');

db.version(1).stores({
  conversations: 'id, title, createdAt, lastModified', // Primary key and indexed fields
  documents: 'id, filename, uploadDate, status', // from RAG feature
  processingJobs: 'id, documentId, status' // from RAG feature
});

db.version(2).stores({
  conversations: 'id, title, createdAt, lastModified',
  documents: 'id, filename, uploadDate, status',
  processingJobs: 'id, documentId, status',
  dataFiles: 'id' // Store datasets for "Talk to my Data"
}).upgrade(tx => {
    // This upgrade function will only run if the database is upgrading from a version < 2.
    // Dexie automatically handles creating the new 'dataFiles' table.
    // No specific data migration needed from v1 to v2 for this change.
    return tx.table('dataFiles').clear(); // Good practice to start with a clean slate for the new table.
}); 