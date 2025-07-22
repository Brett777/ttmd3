import React, { useState, useEffect } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Button,
  VStack,
  HStack,
  Text,
  Radio,
  RadioGroup,
  Divider,
  Box,
  Badge,
} from '@chakra-ui/react';
import { humanFileSize } from '../services/documentStorage'; // Assuming a utility function

const DuplicateFileModal = ({ isOpen, duplicateFiles, onConfirm, onCancel, isProcessing }) => {
  const [choices, setChoices] = useState({});

  useEffect(() => {
    // Initialize choices when the modal opens
    if (isOpen) {
      const initialChoices = duplicateFiles.reduce((acc, { file }) => {
        acc[file.name] = 'skip'; // Default to skipping
        return acc;
      }, {});
      setChoices(initialChoices);
    }
  }, [isOpen, duplicateFiles]);

  const handleChoiceChange = (fileName, value) => {
    setChoices(prev => ({ ...prev, [fileName]: value }));
  };

  const handleConfirm = () => {
    onConfirm(choices);
  };

  const handleSetAll = (action) => {
    const newChoices = Object.keys(choices).reduce((acc, fileName) => {
      acc[fileName] = action;
      return acc;
    }, {});
    setChoices(newChoices);
  };

  return (
    <Modal isOpen={isOpen} onClose={onCancel} size="2xl" isCentered>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Duplicate Files Found</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Text mb={4}>The following files already exist in this session. Please choose an action for each.</Text>
          <HStack spacing={4} mb={4}>
            <Button size="sm" onClick={() => handleSetAll('replace')}>Replace All</Button>
            <Button size="sm" onClick={() => handleSetAll('skip')}>Skip All</Button>
          </HStack>
          <Divider />
          <VStack spacing={4} mt={4} maxH="40vh" overflowY="auto" pr={2}>
            {duplicateFiles.map(({ file, existingDoc }) => (
              <Box key={file.name} p={4} borderWidth="1px" borderRadius="md" w="100%">
                <Text fontWeight="bold">{file.name}</Text>
                <Text fontSize="sm" color="gray.500">
                  New size: {humanFileSize(file.size)} | Existing size: {humanFileSize(existingDoc.fileSize)}
                </Text>
                <RadioGroup 
                  mt={2}
                  onChange={(value) => handleChoiceChange(file.name, value)} 
                  value={choices[file.name] || 'skip'}
                >
                  <HStack spacing={5}>
                    <Radio value="replace">
                      <Text>Replace <Badge colorScheme="orange">Will re-process</Badge></Text>
                    </Radio>
                    <Radio value="skip">
                      <Text>Skip <Badge>Keep original</Badge></Text>
                    </Radio>
                  </HStack>
                </RadioGroup>
              </Box>
            ))}
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onCancel} isDisabled={isProcessing}>
            Cancel Upload
          </Button>
          <Button colorScheme="blue" onClick={handleConfirm} isLoading={isProcessing}>
            Confirm
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default DuplicateFileModal; 