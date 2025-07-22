/**
 * Completed Documents Component
 * Displays recently processed documents
 */

import React from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Badge,
  IconButton,
  useColorModeValue,
  Icon,
  Tooltip,
} from '@chakra-ui/react';
import { FiFile, FiTrash2, FiCheck } from 'react-icons/fi';
import { useDocuments } from '../contexts/DocumentContext';
import { useDocumentProcessor } from '../hooks/useDocumentProcessor';

function DocumentItem({ document }) {
  const { deleteDocument } = useDocumentProcessor();
  
  // Colors
  const bgColor = useColorModeValue('gray.50', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };
  
  const handleDelete = async () => {
    if (window.confirm(`Are you sure you want to delete ${document.filename}?`)) {
      try {
        await deleteDocument(document.id);
      } catch (error) {
        console.error('Error deleting document:', error);
      }
    }
  };
  
  return (
    <Box
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="md"
      p={4}
      bg={bgColor}
      _hover={{
        borderColor: 'blue.300',
      }}
      transition="border-color 0.2s"
    >
      <HStack justify="space-between">
        <HStack spacing={3} flex={1}>
          <Icon as={FiFile} color="blue.500" />
          <VStack align="start" spacing={0} flex={1}>
            <Text fontWeight="medium" noOfLines={1}>
              {document.filename}
            </Text>
            <HStack spacing={2} fontSize="sm" color="gray.500">
              <Text>{formatFileSize(document.fileSize)}</Text>
              <Text>•</Text>
              <Text>{document.metadata?.document_details?.total_pages || 0} pages</Text>
              <Text>•</Text>
              <Text>{formatDate(document.processedDate || document.uploadDate)}</Text>
            </HStack>
          </VStack>
        </HStack>
        
        <HStack spacing={2}>
          <Badge colorScheme="green" variant="subtle">
            <HStack spacing={1}>
              <Icon as={FiCheck} />
              <Text>Completed</Text>
            </HStack>
          </Badge>
          
          <Tooltip label="Delete document">
            <IconButton
              icon={<FiTrash2 />}
              size="sm"
              variant="ghost"
              colorScheme="red"
              onClick={handleDelete}
            />
          </Tooltip>
        </HStack>
      </HStack>
    </Box>
  );
}

export function CompletedDocuments({ limit }) {
  const { documents } = useDocuments();
  
  // Filter completed documents and sort by date
  const completedDocs = documents
    .filter(doc => doc.status === 'completed')
    .sort((a, b) => {
      const dateA = new Date(a.processedDate || a.uploadDate);
      const dateB = new Date(b.processedDate || b.uploadDate);
      return dateB - dateA; // Most recent first
    })
    .slice(0, limit);
  
  if (completedDocs.length === 0) {
    return (
      <Box
        p={8}
        textAlign="center"
        borderWidth="1px"
        borderStyle="dashed"
        borderColor={useColorModeValue('gray.300', 'gray.600')}
        borderRadius="md"
      >
        <Text color="gray.500">
          No completed documents yet
        </Text>
      </Box>
    );
  }
  
  return (
    <VStack spacing={3} align="stretch">
      {completedDocs.map(doc => (
        <DocumentItem key={doc.id} document={doc} />
      ))}
    </VStack>
  );
}

export default CompletedDocuments; 