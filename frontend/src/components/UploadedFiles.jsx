import React from 'react';
import {
    Box,
    VStack,
    HStack,
    Text,
    IconButton,
    Flex,
    Tag,
    Tooltip,
} from '@chakra-ui/react';
import { FaFileCsv, FaTrash } from 'react-icons/fa';

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

const UploadedFiles = ({ uploadedFiles, selectedFile, onSelectFile, onDeleteFile }) => {
    return (
        <VStack spacing={2} align="stretch">
            {uploadedFiles.map((file) => (
                <Box
                    key={file.id}
                    p={3}
                    borderWidth="1px"
                    borderRadius="md"
                    onClick={() => onSelectFile(file)}
                    cursor="pointer"
                    bg={selectedFile?.id === file.id ? 'blue.50' : 'white'}
                    borderColor={selectedFile?.id === file.id ? 'blue.300' : 'gray.200'}
                    _hover={{ bg: 'gray.50' }}
                >
                    <Flex justify="space-between" align="center">
                        <HStack>
                            <FaFileCsv size="20px" color="green" />
                            <Box>
                                <Text fontWeight="bold">{file.name}</Text>
                                <Text fontSize="sm" color="gray.500">
                                    {file.rowCount} rows, {file.columnCount} columns, {formatBytes(file.size)}
                                </Text>
                            </Box>
                        </HStack>
                        <Tooltip label="Delete File" placement="top">
                            <IconButton
                                aria-label="Delete file"
                                icon={<FaTrash />}
                                size="sm"
                                variant="ghost"
                                colorScheme="red"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteFile(file.id);
                                }}
                            />
                        </Tooltip>
                    </Flex>
                </Box>
            ))}
        </VStack>
    );
};

export default UploadedFiles; 