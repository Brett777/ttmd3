/**
 * Storage Test Component
 * For testing IndexedDB storage functionality
 */

import React, { useEffect } from 'react';
import { Box, VStack, Text, Button, useToast } from '@chakra-ui/react';
import { useDocuments } from '../contexts/DocumentContext';
import documentStorage from '../services/documentStorage';

export function StorageTest() {
  const {
    documents,
    processingJobs,
    storageInfo,
    addDocument,
    updateDocument,
    deleteDocument,
    clearAllData
  } = useDocuments();
  
  const toast = useToast();
  
  useEffect(() => {
    // Test storage on mount
    testStorage();
  }, []);
  
  const testStorage = async () => {
    try {
      // Check if IndexedDB is available
      const quota = await documentStorage.checkStorageQuota();
      console.log('Storage quota:', quota);
      
      toast({
        title: 'Storage Available',
        description: `Using ${quota.usedMB} MB of ${quota.totalMB} MB`,
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      console.error('Storage test error:', error);
      toast({
        title: 'Storage Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    }
  };
  
  const addTestDocument = async () => {
    try {
      const testDoc = {
        filename: `test-doc-${Date.now()}.pdf`,
        fileSize: 1024 * 1024, // 1MB
        uploadDate: new Date().toISOString(),
        status: 'completed',
        extractionModel: 'gpt-4o',
        metadata: {
          document_details: {
            title: 'Test Document',
            total_pages: 10
          },
          pages: {},
          document_summary: {
            short_summary: 'This is a test document.'
          }
        }
      };
      
      const id = await addDocument(testDoc);
      console.log('Added test document with ID:', id);
      
      toast({
        title: 'Document Added',
        description: `Test document added successfully`,
        status: 'success',
        duration: 3000,
      });
    } catch (error) {
      console.error('Error adding test document:', error);
      toast({
        title: 'Error',
        description: error.message,
        status: 'error',
        duration: 5000,
      });
    }
  };
  
  return (
    <Box p={4} borderWidth="1px" borderRadius="lg">
      <VStack align="stretch" spacing={4}>
        <Text fontSize="lg" fontWeight="bold">Storage Test</Text>
        
        <Box>
          <Text>Documents: {documents.length}</Text>
          <Text>Processing Jobs: {processingJobs.length}</Text>
          <Text>Storage Used: {storageInfo.usedMB} MB / {storageInfo.totalMB} MB ({storageInfo.percentage.toFixed(2)}%)</Text>
        </Box>
        
        <Button onClick={addTestDocument} colorScheme="blue" size="sm">
          Add Test Document
        </Button>
        
        <Button onClick={clearAllData} colorScheme="red" size="sm">
          Clear All Data
        </Button>
        
        {documents.length > 0 && (
          <Box>
            <Text fontWeight="bold">Documents:</Text>
            {documents.map(doc => (
              <Box key={doc.id} p={2} borderWidth="1px" borderRadius="md" mt={2}>
                <Text fontSize="sm">{doc.filename}</Text>
                <Text fontSize="xs" color="gray.500">Status: {doc.status}</Text>
                <Button
                  size="xs"
                  colorScheme="red"
                  mt={1}
                  onClick={() => deleteDocument(doc.id)}
                >
                  Delete
                </Button>
              </Box>
            ))}
          </Box>
        )}
      </VStack>
    </Box>
  );
}

export default StorageTest; 