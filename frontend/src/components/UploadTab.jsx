/**
 * Upload Tab Component
 * Handles document upload, model selection, and processing queue
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Select,
  Button,
  useColorModeValue,
  useToast,
  FormControl,
  FormLabel,
  Icon,
  Divider,
} from '@chakra-ui/react';
import { FiUpload, FiFile } from 'react-icons/fi';
import { useDocumentProcessor } from '../hooks/useDocumentProcessor';
import ProcessingQueue from './ProcessingQueue';
import CompletedDocuments from './CompletedDocuments';
import DuplicateFileModal from './DuplicateFileModal';

// Available models for extraction (matching backend AVAILABLE_MODELS)
const EXTRACTION_MODELS = [
  { id: 'gpt-4.1-mini', name: 'ChatGPT 4.1 mini', provider: 'OpenAI' },
  { id: 'gpt-4.1-nano', name: 'ChatGPT 4.1 nano', provider: 'OpenAI' },
  { id: 'gpt-4o-mini', name: 'ChatGPT 4o mini', provider: 'OpenAI' },
  { id: 'gpt-4.1', name: 'ChatGPT 4.1', provider: 'OpenAI' },
  { id: 'gpt-4o', name: 'ChatGPT 4o', provider: 'OpenAI' },
  { id: 'claude-3-5-haiku-latest', name: 'Claude 3.5 Haiku', provider: 'Anthropic' },
  { id: 'claude-3-5-sonnet-latest', name: 'Claude 3.5 Sonnet', provider: 'Anthropic' },
  { id: 'claude-3-7-sonnet-latest', name: 'Claude 3.7 Sonnet', provider: 'Anthropic' },
  { id: 'grok-2', name: 'Grok 2', provider: 'xAI' },
  { id: 'grok-3', name: 'Grok 3', provider: 'xAI' },
];

export function UploadTab() {
  const [selectedModel, setSelectedModel] = useState('gpt-4.1-mini');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const toast = useToast();
  
  const { 
    uploadDocument, 
    isProcessing, 
    error,
    duplicateFiles,
    handleConfirmDuplicateUpload,
    cancelDuplicateUpload,
  } = useDocumentProcessor();
  
  // Colors
  const borderColor = useColorModeValue('gray.300', 'gray.600');
  const dropzoneBg = useColorModeValue('gray.50', 'gray.700');
  const dropzoneHoverBg = useColorModeValue('blue.50', 'blue.900');
  const dropzoneBorder = isDragging ? 'blue.400' : borderColor;
  
  // Handle file selection
  const handleFileSelect = useCallback(async (files) => {
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      'application/vnd.ms-powerpoint',
      'text/plain'
    ];
    
    const allowedExtensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.txt'];
    
    const validFiles = [];
    for (const file of files) {
      // Check file type
      const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExt)) {
        toast({
          title: 'Invalid file type',
          description: `${file.name} is not a supported file type. Supported: PDF, DOCX, PPTX, TXT`,
          status: 'error',
          duration: 5000,
        });
        continue;
      }
      
      // Check file size (20MB limit)
      if (file.size > 20 * 1024 * 1024) {
        toast({
          title: 'File too large',
          description: `${file.name} exceeds the 20MB size limit`,
          status: 'error',
          duration: 5000,
        });
        continue;
      }
      validFiles.push(file);
    }
    
    if (validFiles.length > 0) {
      try {
        // The uploadDocument hook now handles arrays and the duplicate check logic
        await uploadDocument(validFiles, selectedModel);
        // Toast notifications can be simplified as the hook handles individual file states
        toast({
          title: 'Upload process initiated',
          description: `Checking ${validFiles.length} file(s) for processing.`,
          status: 'info',
          duration: 3000,
        });
      } catch (err) {
        toast({
          title: 'Upload failed',
          description: err.message,
          status: 'error',
          duration: 5000,
        });
      }
    }
  }, [selectedModel, uploadDocument, toast]);
  
  // Drag and drop handlers
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) {
      setIsDragging(false);
    }
  }, []);
  
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);
  
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files);
    }
  }, [handleFileSelect]);
  
  // File input change handler
  const handleFileInputChange = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      handleFileSelect(files);
    }
    // Reset input
    e.target.value = '';
  }, [handleFileSelect]);
  
  return (
    <VStack spacing={6} align="stretch">
      {/* Duplicate File Modal */}
      <DuplicateFileModal
        isOpen={duplicateFiles.length > 0}
        duplicateFiles={duplicateFiles}
        onConfirm={handleConfirmDuplicateUpload}
        onCancel={cancelDuplicateUpload}
        isProcessing={isProcessing}
      />

      {/* Model Selection */}
      <FormControl>
        <FormLabel>Extraction Model</FormLabel>
        <Select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          size="md"
        >
          {Object.entries(
            EXTRACTION_MODELS.reduce((acc, model) => {
              const provider = model.provider;
              if (!acc[provider]) acc[provider] = [];
              acc[provider].push(model);
              return acc;
            }, {})
          ).map(([provider, models]) => (
            <optgroup key={provider} label={provider}>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </optgroup>
          ))}
        </Select>
      </FormControl>
      
      {/* Drop Zone */}
      <Box
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        cursor="pointer"
        borderWidth="2px"
        borderStyle="dashed"
        borderColor={dropzoneBorder}
        borderRadius="lg"
        bg={isDragging ? dropzoneHoverBg : dropzoneBg}
        p={12}
        textAlign="center"
        transition="all 0.2s"
        _hover={{
          borderColor: 'blue.400',
          bg: dropzoneHoverBg,
        }}
      >
        <VStack spacing={3}>
          <Icon as={FiUpload} boxSize={12} color="gray.400" />
          <Text fontSize="lg" fontWeight="medium">
            Drop files here or click to browse
          </Text>
          <Text fontSize="sm" color="gray.500">
            Supports PDF, DOCX, PPTX, TXT (max 20MB)
          </Text>
        </VStack>
      </Box>
      
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.docx,.doc,.pptx,.ppt,.txt"
        style={{ display: 'none' }}
        onChange={handleFileInputChange}
      />
      
      {/* Processing Queue */}
      <Box>
        <Text fontSize="lg" fontWeight="bold" mb={3}>
          Processing Queue
        </Text>
        <ProcessingQueue />
      </Box>
      
      <Divider />
      
      {/* Completed Documents */}
      <Box>
        <Text fontSize="lg" fontWeight="bold" mb={3}>
          Recently Processed
        </Text>
        <CompletedDocuments limit={5} />
      </Box>
    </VStack>
  );
}

export default UploadTab; 