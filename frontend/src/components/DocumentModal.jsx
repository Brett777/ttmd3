/**
 * Document Management Modal
 * Main modal for document upload, processing, and exploration
 */

import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  useColorModeValue,
  Box,
  Flex,
} from '@chakra-ui/react';
import { useDocuments } from '../contexts/DocumentContext';
import UploadTab from './UploadTab';
import ExplorerTab from './ExplorerTab';

export function DocumentModal({ isOpen, onClose }) {
  const { processingJobs } = useDocuments();
  const [selectedTab, setSelectedTab] = useState(0);
  
  // Count active jobs
  const activeJobsCount = processingJobs.filter(
    job => job.status === 'queued' || job.status === 'processing'
  ).length;
  
  // Modal styling
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  return (
    <Modal 
      isOpen={isOpen} 
      onClose={onClose} 
      size="7xl"
      scrollBehavior="inside"
    >
      <ModalOverlay />
      <ModalContent bg={bgColor} maxH="90vh" maxW="90vw">
        <ModalHeader>
          <Flex align="center" justify="space-between">
            <Box>
              Document Management
              {activeJobsCount > 0 && (
                <Badge ml={3} colorScheme="blue" variant="solid">
                  {activeJobsCount} processing
                </Badge>
              )}
            </Box>
          </Flex>
        </ModalHeader>
        <ModalCloseButton />
        
        <ModalBody p={0}>
          <Tabs 
            index={selectedTab} 
            onChange={setSelectedTab}
            variant="enclosed"
            colorScheme="blue"
          >
            <TabList borderBottom="1px" borderColor={borderColor} px={6}>
              <Tab>Upload & Process</Tab>
              <Tab>Document Explorer</Tab>
            </TabList>
            
            <TabPanels minH="70vh">
              <TabPanel p={6}>
                <UploadTab />
              </TabPanel>
              
              <TabPanel p={0}>
                <ExplorerTab />
              </TabPanel>
            </TabPanels>
          </Tabs>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
}

export default DocumentModal; 