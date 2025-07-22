/**
 * Processing Queue Component
 * Displays active and queued document processing jobs with detailed progress information
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Progress,
  Button,
  Badge,
  IconButton,
  useColorModeValue,
  Collapse,
  Icon,
  Tooltip,
  Flex,
  Spacer,
  Circle,
  useTheme,
} from '@chakra-ui/react';
import { 
  FiFile, 
  FiChevronDown, 
  FiChevronUp, 
  FiX, 
  FiClock,
  FiFileText,
  FiImage,
  FiCpu,
  FiCheckCircle,
  FiAlertCircle
} from 'react-icons/fi';
import { useDocuments } from '../contexts/DocumentContext';
import { useDocumentProcessor } from '../hooks/useDocumentProcessor';

// Stage configuration with icons and descriptions
const PROCESSING_STAGES = {
  'Extracting text from document...': {
    icon: FiFileText,
    color: 'blue',
    description: 'Reading document content and extracting text from all pages'
  },
  'Converting pages to images...': {
    icon: FiImage,
    color: 'green',
    description: 'Converting each page to high-quality images for AI analysis'
  },
  'Generating page metadata...': {
    icon: FiCpu,
    color: 'purple',
    description: 'AI is analyzing each page to extract insights and metadata'
  },
  'Extracting document details...': {
    icon: FiFileText,
    color: 'orange',
    description: 'Gathering document-level information and properties'
  },
  'Creating document summary...': {
    icon: FiCpu,
    color: 'teal',
    description: 'AI is generating comprehensive document summary'
  },
  'Finalizing...': {
    icon: FiCheckCircle,
    color: 'green',
    description: 'Completing processing and saving results'
  }
};

function StageIndicator({ stage, isActive, isCompleted }) {
  const theme = useTheme();
  const stageConfig = PROCESSING_STAGES[stage] || {
    icon: FiCpu,
    color: 'gray',
    description: stage
  };
  
  const iconColor = isCompleted 
    ? theme.colors.green[500]
    : isActive 
      ? theme.colors[stageConfig.color][500]
      : theme.colors.gray[400];
      
  const bgColor = isCompleted
    ? theme.colors.green[100]
    : isActive
      ? theme.colors[stageConfig.color][100]
      : theme.colors.gray[100];

  return (
    <Tooltip label={stageConfig.description} placement="top">
      <Circle
        size="40px"
        bg={bgColor}
        color={iconColor}
        border={isActive ? `2px solid ${iconColor}` : '2px solid transparent'}
        transition="all 0.3s ease"
      >
        <Icon as={stageConfig.icon} boxSize="20px" />
      </Circle>
    </Tooltip>
  );
}

function ProgressDetails({ progress, startTime }) {
  const [elapsedTime, setElapsedTime] = useState('');
  const [estimatedRemaining, setEstimatedRemaining] = useState('');
  
  useEffect(() => {
    const interval = setInterval(() => {
      if (startTime) {
        const start = new Date(startTime);
        const now = new Date();
        const elapsed = Math.floor((now - start) / 1000);
        
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        setElapsedTime(`${minutes}:${seconds.toString().padStart(2, '0')}`);
        
        // Calculate estimated remaining time
        if (progress?.total > 0 && progress?.current > 0) {
          const progressPercent = progress.current / progress.total;
          const totalEstimatedTime = elapsed / progressPercent;
          const remaining = Math.max(0, totalEstimatedTime - elapsed);
          const remainingMinutes = Math.floor(remaining / 60);
          const remainingSeconds = Math.floor(remaining % 60);
          setEstimatedRemaining(`~${remainingMinutes}:${remainingSeconds.toString().padStart(2, '0')}`);
        }
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [startTime, progress]);
  
  return (
    <HStack fontSize="xs" color="gray.500" spacing={4}>
      <HStack spacing={1}>
        <Icon as={FiClock} />
        <Text>Elapsed: {elapsedTime}</Text>
      </HStack>
      {estimatedRemaining && (
        <HStack spacing={1}>
          <Icon as={FiClock} />
          <Text>Remaining: {estimatedRemaining}</Text>
        </HStack>
      )}
    </HStack>
  );
}

function JobItem({ job }) {
  const [showLogs, setShowLogs] = useState(false);
  const { documents } = useDocuments();
  
  // Colors - Move all hooks to the top
  const bgColor = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const logsBgColor = useColorModeValue('gray.50', 'gray.800');
  const progressBarBg = useColorModeValue('gray.100', 'gray.600');
  const errorBg = useColorModeValue('red.50', 'red.900');
  
  // Find the document for this job
  const document = documents.find(doc => doc.id === job.documentId);
  const filename = document?.filename || 'Unknown file';
  
  // Calculate progress percentage
  const progressPercent = job.progress?.total > 0
    ? Math.round((job.progress.current / job.progress.total) * 100)
    : 0;
  
  // Status badge color
  const statusColor = {
    queued: 'gray',
    processing: 'blue',
    completed: 'green',
    error: 'red',
  }[job.status] || 'gray';
  
  // Get current stage info (FLEXIBLE MATCHING)
  const currentStage = job.progress?.stage || 'Waiting...';
  // Find the closest matching stage key
  const matchedStageKey = Object.keys(PROCESSING_STAGES).find(key => currentStage.startsWith(key));
  const stageConfig = PROCESSING_STAGES[matchedStageKey] || {
    icon: FiCpu,
    color: 'gray',
    description: currentStage
  };
  
  // Determine completed stages for stage indicator (FLEXIBLE MATCHING)
  const allStages = Object.keys(PROCESSING_STAGES);
  const currentStageIndex = allStages.findIndex(key => currentStage.startsWith(key));
  
  // Format file size
  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };
  
  // Debug logging
  console.log(`[ProcessingQueue] Job ${job.id}:`, {
    rawStage: currentStage,
    matchedStageKey,
    progress: job.progress,
    status: job.status
  });
  
  return (
    <Box
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="lg"
      p={5}
      bg={bgColor}
      boxShadow="sm"
      transition="all 0.2s ease"
      _hover={{ boxShadow: 'md' }}
    >
      <VStack align="stretch" spacing={4}>
        {/* Header */}
        <Flex align="center" justify="space-between">
          <HStack spacing={3} flex="1" minW="0">
            <Icon as={FiFile} color="blue.500" boxSize="20px" />
            <VStack align="start" spacing={0} flex="1" minW="0">
              <Text fontWeight="semibold" noOfLines={1} fontSize="md">
                {filename}
              </Text>
              <HStack spacing={2} fontSize="xs" color="gray.500">
                {document?.fileSize && (
                  <Text>{formatFileSize(document.fileSize)}</Text>
                )}
                {document?.extractionModel && (
                  <>
                    <Text>•</Text>
                    <Text>{document.extractionModel}</Text>
                  </>
                )}
              </HStack>
            </VStack>
          </HStack>
          <Badge 
            colorScheme={statusColor} 
            variant="subtle" 
            px={3} 
            py={1} 
            borderRadius="full"
            textTransform="capitalize"
          >
            {job.status}
          </Badge>
        </Flex>
        
        {/* Processing Status - Show whenever status is processing OR we have any meaningful stage */}
        {(job.status === 'processing' || 
          (job.progress?.stage && 
           !['Queued', 'Waiting to start...', 'Waiting...', ''].includes(job.progress.stage))) && (
          <>
            {/* Current Stage */}
            <Box>
              <HStack justify="space-between" mb={2}>
                <HStack spacing={2}>
                  {stageConfig && (
                    <Icon 
                      as={stageConfig.icon} 
                      color={`${stageConfig.color}.500`}
                      boxSize="16px" 
                    />
                  )}
                  <Text fontSize="sm" fontWeight="medium" color="gray.700">
                    {currentStage}
                  </Text>
                </HStack>
                <Text fontSize="sm" color="gray.500">
                  {job.progress?.current || 0} / {job.progress?.total || 0}
                </Text>
              </HStack>
              
              {/* Progress Bar */}
              <Progress
                value={progressPercent}
                size="md"
                colorScheme={stageConfig?.color || 'blue'}
                borderRadius="full"
                hasStripe
                isAnimated
                bg={progressBarBg}
              />
              
              <HStack justify="space-between" mt={2}>
                <Text fontSize="xs" color="gray.500">
                  {progressPercent}% complete
                </Text>
                <ProgressDetails progress={job.progress} startTime={job.startTime} />
              </HStack>
            </Box>
            
            {/* Stage Indicators */}
            <Box>
              <Text fontSize="xs" color="gray.500" mb={2}>Processing Pipeline:</Text>
              <HStack spacing={2} overflowX="auto" pb={1}>
                {allStages.map((stage, index) => (
                  <StageIndicator
                    key={stage}
                    stage={stage}
                    isActive={index === currentStageIndex}
                    isCompleted={index < currentStageIndex}
                  />
                ))}
              </HStack>
            </Box>
          </>
        )}
        
        {/* Queued Status - Only show if we don't have a processing stage */}
        {!(job.status === 'processing' || 
          (job.progress?.stage && 
           !['Queued', 'Waiting to start...', 'Waiting...', ''].includes(job.progress.stage))) && 
         job.status !== 'completed' && 
         job.status !== 'error' && (
          <Box textAlign="center" py={2}>
            <HStack justify="center" spacing={2} color="gray.500">
              <Icon as={FiClock} />
              <Text fontSize="sm">Waiting in queue...</Text>
            </HStack>
          </Box>
        )}
        
        {/* Error message */}
        {job.status === 'error' && job.error && (
          <Box
            p={3}
            bg={errorBg}
            borderRadius="md"
            borderLeftWidth="4px"
            borderLeftColor="red.500"
          >
            <HStack spacing={2}>
              <Icon as={FiAlertCircle} color="red.500" />
              <Text fontSize="sm" color="red.700">
                {job.error}
              </Text>
            </HStack>
          </Box>
        )}
        
        {/* Logs toggle */}
        {job.logs && job.logs.length > 0 && (
          <>
            <Button
              size="sm"
              variant="ghost"
              leftIcon={showLogs ? <FiChevronUp /> : <FiChevronDown />}
              onClick={() => setShowLogs(!showLogs)}
              justifyContent="flex-start"
            >
              {showLogs ? 'Hide' : 'Show'} Processing Logs ({job.logs.length})
            </Button>
            
            <Collapse in={showLogs}>
              <Box
                maxH="300px"
                overflowY="auto"
                p={4}
                bg={logsBgColor}
                borderRadius="md"
                border="1px solid"
                borderColor={borderColor}
              >
                <VStack align="stretch" spacing={1}>
                  {job.logs.map((log, idx) => {
                    const isRecent = idx >= job.logs.length - 5;
                    return (
                      <Text 
                        key={idx} 
                        fontSize="xs" 
                        fontFamily="mono"
                        color={isRecent ? "blue.600" : "gray.600"}
                        fontWeight={isRecent ? "medium" : "normal"}
                        transition="all 0.3s ease"
                      >
                        {log}
                      </Text>
                    );
                  })}
                </VStack>
              </Box>
            </Collapse>
          </>
        )}
      </VStack>
    </Box>
  );
}

export function ProcessingQueue() {
  const { processingJobs } = useDocuments();
  
  // Filter active jobs (queued or processing)
  const activeJobs = processingJobs.filter(
    job => job.status === 'queued' || job.status === 'processing'
  );
  
  // Sort jobs: processing first, then queued
  const sortedJobs = [...activeJobs].sort((a, b) => {
    if (a.status === 'processing' && b.status !== 'processing') return -1;
    if (a.status !== 'processing' && b.status === 'processing') return 1;
    return 0;
  });
  
  if (sortedJobs.length === 0) {
    return (
      <Box
        p={8}
        textAlign="center"
        borderWidth="2px"
        borderStyle="dashed"
        borderColor={useColorModeValue('gray.300', 'gray.600')}
        borderRadius="lg"
        bg={useColorModeValue('gray.50', 'gray.800')}
      >
        <VStack spacing={3}>
          <Icon 
            as={FiCheckCircle} 
            boxSize="40px" 
            color={useColorModeValue('gray.400', 'gray.500')} 
          />
          <Text color="gray.500" fontSize="md">
            No documents are currently being processed
          </Text>
          <Text color="gray.400" fontSize="sm">
            Upload a document to get started
          </Text>
        </VStack>
      </Box>
    );
  }
  
  return (
    <VStack spacing={4} align="stretch">
      {sortedJobs.map(job => (
        <JobItem key={job.id} job={job} />
      ))}
    </VStack>
  );
}

export default ProcessingQueue; 