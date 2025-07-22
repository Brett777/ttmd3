/**
 * Explorer Tab Component
 * Document explorer with list view and detail viewer
 */

import React, { useState } from 'react';
import {
  Box,
  Flex,
  VStack,
  HStack,
  Text,
  Input,
  InputGroup,
  InputLeftElement,
  IconButton,
  useColorModeValue,
  Icon,
  Badge,
  Divider,
  Button,
  Progress,
  Tooltip,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
} from '@chakra-ui/react';
import { FiSearch, FiFile, FiTrash2, FiCalendar, FiFileText, FiHardDrive, FiMoreVertical, FiTrash } from 'react-icons/fi';
import { useDocuments } from '../contexts/DocumentContext';
import { useDocumentProcessor } from '../hooks/useDocumentProcessor';
import DocumentViewer from './DocumentViewer';

function DocumentListItem({ document, isSelected, onClick, onDelete }) {
  const bgColor = useColorModeValue(
    isSelected ? 'blue.50' : 'white',
    isSelected ? 'blue.900' : 'gray.800'
  );
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const hoverBg = useColorModeValue('gray.50', 'gray.700');
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };
  
  const handleDelete = (e) => {
    e.stopPropagation();
    if (window.confirm(`Are you sure you want to delete ${document.filename}?`)) {
      onDelete(document.id);
    }
  };
  
  return (
    <Box
      p={3}
      bg={bgColor}
      borderWidth="1px"
      borderColor={isSelected ? 'blue.400' : borderColor}
      borderRadius="md"
      cursor="pointer"
      onClick={onClick}
      _hover={{ bg: isSelected ? bgColor : hoverBg }}
      transition="all 0.2s"
    >
      <HStack justify="space-between">
        <VStack align="start" spacing={1} flex={1}>
          <HStack>
            <Icon as={FiFile} color="blue.500" />
            <Text fontWeight="medium" fontSize="sm" noOfLines={1}>
              {document.filename}
            </Text>
          </HStack>
          
          <HStack spacing={3} fontSize="xs" color="gray.500">
            <HStack spacing={1}>
              <Icon as={FiFileText} />
              <Text>{document.metadata?.document_details?.total_pages || 0} pages</Text>
            </HStack>
            <HStack spacing={1}>
              <Icon as={FiCalendar} />
              <Text>{formatDate(document.processedDate || document.uploadDate)}</Text>
            </HStack>
          </HStack>
        </VStack>
        
        <IconButton
          icon={<FiTrash2 />}
          size="sm"
          variant="ghost"
          colorScheme="red"
          onClick={handleDelete}
          aria-label="Delete document"
        />
      </HStack>
    </Box>
  );
}

export function ExplorerTab() {
  const { documents, storageInfo } = useDocuments();
  const { deleteDocument, clearAllDocuments } = useDocumentProcessor();
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Colors
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const sidebarBg = useColorModeValue('gray.50', 'gray.900');
  
  // Filter completed documents
  const completedDocs = documents.filter(doc => doc.status === 'completed');
  
  // Filter by search query
  const filteredDocs = completedDocs.filter(doc =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    doc.metadata?.document_details?.title?.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  // Sort by date (most recent first)
  const sortedDocs = [...filteredDocs].sort((a, b) => {
    const dateA = new Date(a.processedDate || a.uploadDate);
    const dateB = new Date(b.processedDate || b.uploadDate);
    return dateB - dateA;
  });
  
  // Get selected document
  const selectedDoc = selectedDocId ? documents.find(d => d.id === selectedDocId) : null;
  
  // Calculate storage percentage
  const storagePercentage = storageInfo && storageInfo.total > 0 
    ? (storageInfo.used / storageInfo.total) * 100 
    : 0;
  
  // Format storage size
  const formatStorageSize = (bytes) => {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
  };
  
  const handleDelete = async (docId) => {
    try {
      await deleteDocument(docId);
      if (selectedDocId === docId) {
        setSelectedDocId(null);
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };
  
  const handleClearAll = async () => {
    if (window.confirm('Are you sure you want to delete all documents? This action cannot be undone.')) {
      try {
        await clearAllDocuments();
        setSelectedDocId(null);
      } catch (error) {
        console.error('Error clearing documents:', error);
      }
    }
  };
  
  return (
    <Flex h="70vh">
      {/* Document List Sidebar */}
      <Box
        w="300px"
        bg={sidebarBg}
        borderRightWidth="1px"
        borderColor={borderColor}
        overflowY="auto"
        flexShrink={0}
      >
        <Box p={4}>
          {/* Search */}
          <InputGroup size="sm" mb={4}>
            <InputLeftElement pointerEvents="none">
              <Icon as={FiSearch} color="gray.400" />
            </InputLeftElement>
            <Input
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </InputGroup>
          
          {/* Document count and actions */}
          <HStack justify="space-between" mb={3}>
            <HStack>
              <Text fontSize="sm" fontWeight="medium">
                Documents
              </Text>
              <Badge colorScheme="blue" variant="subtle">
                {sortedDocs.length}
              </Badge>
            </HStack>
            
            {sortedDocs.length > 0 && (
              <Menu>
                <MenuButton
                  as={IconButton}
                  icon={<FiMoreVertical />}
                  size="xs"
                  variant="ghost"
                  aria-label="More actions"
                />
                <MenuList>
                  <MenuItem 
                    icon={<FiTrash />} 
                    onClick={handleClearAll}
                    color="red.500"
                  >
                    Clear all documents
                  </MenuItem>
                </MenuList>
              </Menu>
            )}
          </HStack>
          
          {/* Document list */}
          <VStack spacing={2} align="stretch">
            {sortedDocs.length === 0 ? (
              <Text fontSize="sm" color="gray.500" textAlign="center" py={8}>
                {searchQuery ? 'No documents match your search' : 'No documents yet'}
              </Text>
            ) : (
              sortedDocs.map(doc => (
                <DocumentListItem
                  key={doc.id}
                  document={doc}
                  isSelected={selectedDocId === doc.id}
                  onClick={() => setSelectedDocId(doc.id)}
                  onDelete={handleDelete}
                />
              ))
            )}
          </VStack>
          
          {/* Storage Usage */}
          {storageInfo && storageInfo.total > 0 && (
            <Box mt={6} pt={4} borderTopWidth="1px" borderColor={borderColor}>
              <HStack mb={2}>
                <Icon as={FiHardDrive} color="gray.500" />
                <Text fontSize="sm" fontWeight="medium">Storage Usage</Text>
              </HStack>
              <Tooltip 
                label={`${formatStorageSize(storageInfo.used)} of ${formatStorageSize(storageInfo.total)} used`}
                placement="top"
              >
                <Box>
                  <Progress 
                    value={storagePercentage} 
                    size="sm" 
                    colorScheme={storagePercentage > 90 ? 'red' : storagePercentage > 70 ? 'yellow' : 'blue'}
                    borderRadius="full"
                  />
                  <Text fontSize="xs" color="gray.500" mt={1}>
                    {formatStorageSize(storageInfo.used)} / {formatStorageSize(storageInfo.total)}
                  </Text>
                </Box>
              </Tooltip>
            </Box>
          )}
        </Box>
      </Box>
      
      {/* Document Viewer */}
      <Box flex={1} overflowY="auto">
        {selectedDoc ? (
          <DocumentViewer document={selectedDoc} />
        ) : (
          <Flex h="100%" align="center" justify="center">
            <VStack spacing={3}>
              <Icon as={FiFile} boxSize={12} color="gray.400" />
              <Text color="gray.500">
                {sortedDocs.length === 0 
                  ? 'No documents to display'
                  : 'Select a document to view details'
                }
              </Text>
            </VStack>
          </Flex>
        )}
      </Box>
    </Flex>
  );
}

export default ExplorerTab; 