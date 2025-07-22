/**
 * Document Selection Modal
 * Allows users to select which documents to use for chat context
 */

import React from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Button,
  VStack,
  HStack,
  Text,
  Checkbox,
  Badge,
  Box,
  useColorModeValue,
  Icon,
} from '@chakra-ui/react';
import { FiFile, FiCalendar } from 'react-icons/fi';
import { useDocuments } from '../contexts/DocumentContext';

export function DocumentSelectionModal({ isOpen, onClose }) {
  const { documents, selectedDocumentIds, setSelectedDocuments } = useDocuments();
  
  // Filter only completed documents
  const completedDocs = documents.filter(doc => doc.status === 'completed');
  
  // Colors
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const hoverBg = useColorModeValue('gray.50', 'gray.700');
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString();
  };
  
  const handleToggleDocument = (docId) => {
    if (selectedDocumentIds.includes(docId)) {
      setSelectedDocuments(selectedDocumentIds.filter(id => id !== docId));
    } else {
      setSelectedDocuments(Array.from(new Set([...selectedDocumentIds, docId])));
    }
  };
  
  const handleSelectAll = () => {
    if (selectedDocumentIds.length === completedDocs.length) {
      setSelectedDocuments([]);
    } else {
      setSelectedDocuments(completedDocs.map(doc => doc.id));
    }
  };
  
  return (
    <Modal isOpen={isOpen} onClose={onClose} size="lg">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Select Documents for Chat Context</ModalHeader>
        <ModalCloseButton />
        
        <ModalBody>
          {completedDocs.length === 0 ? (
            <Text textAlign="center" py={8} color="gray.500">
              No processed documents available. Upload and process documents first.
            </Text>
          ) : (
            <VStack align="stretch" spacing={3}>
              {/* Select All */}
              <HStack justify="space-between" pb={2} borderBottomWidth="1px" borderColor={borderColor}>
                <Text fontWeight="medium">
                  {selectedDocumentIds.length} of {completedDocs.length} selected
                </Text>
                <Button size="sm" variant="link" onClick={handleSelectAll}>
                  {selectedDocumentIds.length === completedDocs.length ? 'Deselect All' : 'Select All'}
                </Button>
              </HStack>
              
              {/* Document List */}
              {completedDocs.map(doc => (
                <Box
                  key={doc.id}
                  p={3}
                  borderWidth="1px"
                  borderColor={borderColor}
                  borderRadius="md"
                  _hover={{ bg: hoverBg }}
                  cursor="pointer"
                  onClick={() => handleToggleDocument(doc.id)}
                >
                  <HStack justify="space-between">
                    <HStack flex={1}>
                      <Checkbox
                        isChecked={selectedDocumentIds.includes(doc.id)}
                        onChange={() => handleToggleDocument(doc.id)}
                        onClick={(e) => e.stopPropagation()}
                      />
                      <VStack align="start" spacing={0} flex={1}>
                        <HStack>
                          <Icon as={FiFile} color="blue.500" />
                          <Text fontWeight="medium">{doc.filename}</Text>
                        </HStack>
                        <HStack spacing={3} fontSize="sm" color="gray.500">
                          <Text>{doc.metadata?.document_details?.total_pages || 0} pages</Text>
                          <HStack spacing={1}>
                            <Icon as={FiCalendar} boxSize={3} />
                            <Text>{formatDate(doc.processedDate || doc.uploadDate)}</Text>
                          </HStack>
                        </HStack>
                      </VStack>
                    </HStack>
                    <Badge colorScheme="green" variant="subtle">
                      {doc.extractionModel}
                    </Badge>
                  </HStack>
                </Box>
              ))}
            </VStack>
          )}
        </ModalBody>
        
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button colorScheme="blue" onClick={onClose}>
            Done
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}

export default DocumentSelectionModal; 