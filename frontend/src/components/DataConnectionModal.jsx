import React, { useState, useCallback } from 'react';
import {
    Modal, ModalOverlay, ModalContent, ModalHeader, ModalFooter, ModalBody, ModalCloseButton,
    Button, Tabs, TabList, TabPanels, Tab, TabPanel, Box, VStack, Text, useToast, Spinner,
    FormControl, FormLabel, Input, SimpleGrid, HStack, Checkbox,
} from '@chakra-ui/react';
import { useDropzone } from 'react-dropzone';
import { useData } from '../contexts/DataContext';
import UploadedFiles from './UploadedFiles';
import DataViewer from './DataViewer';
import { testDBConnection, getDbSchema } from '../services/dataApi';

const SnowflakeForm = ({ onConnect, isLoading }) => {
    const [params, setParams] = useState({
        user: '', password: '', account: '', warehouse: '', database: '', schema: '',
    });

    const handleChange = (e) => {
        setParams({ ...params, [e.target.name]: e.target.value });
    };

    const handleConnect = () => {
        onConnect(params);
    };

    return (
        <VStack spacing={4} align="stretch">
            <SimpleGrid columns={2} spacing={4}>
                <FormControl isRequired><FormLabel>Account</FormLabel><Input name="account" value={params.account} onChange={handleChange} /></FormControl>
                <FormControl isRequired><FormLabel>Warehouse</FormLabel><Input name="warehouse" value={params.warehouse} onChange={handleChange} /></FormControl>
                <FormControl isRequired><FormLabel>User</FormLabel><Input name="user" value={params.user} onChange={handleChange} /></FormControl>
                <FormControl isRequired><FormLabel>Password</FormLabel><Input name="password" type="password" value={params.password} onChange={handleChange} /></FormControl>
                <FormControl><FormLabel>Database</FormLabel><Input name="database" value={params.database} onChange={handleChange} /></FormControl>
                <FormControl><FormLabel>Schema</FormLabel><Input name="schema" value={params.schema} onChange={handleChange} /></FormControl>
            </SimpleGrid>
            <Button onClick={handleConnect} isLoading={isLoading} colorScheme="blue">Test Connection</Button>
        </VStack>
    );
};

const DataConnectionModal = ({ isOpen, onClose }) => {
    const { registerFile, uploadedFiles, setUploadedFiles, isLoading: isUploading, setActiveConnection, deleteDataset } = useData();
    const [selectedFile, setSelectedFile] = useState(null);
    const toast = useToast();

    // DB connection state
    const [isConnecting, setIsConnecting] = useState(false);
    const [dbSchema, setDbSchema] = useState(null);
    const [connectionParams, setConnectionParams] = useState(null);
    const [selectedSchema, setSelectedSchema] = useState(null);
    const [selectedTables, setSelectedTables] = useState([]);

    const onDrop = useCallback(async (acceptedFiles) => {
        for (const file of acceptedFiles) {
            if (uploadedFiles.some(f => f.name === file.name)) {
                toast({
                    title: 'Duplicate file name.',
                    description: `A file named ${file.name} already exists. Please rename the file and try again.`,
                    status: 'warning',
                    duration: 5000,
                    isClosable: true,
                });
                continue;
            }
            const newFile = await registerFile(file);
            if (newFile) {
                toast({
                    title: 'File uploaded.',
                    description: `${file.name} has been processed.`,
                    status: 'success',
                    duration: 5000,
                    isClosable: true,
                });
                if (!selectedFile) {
                    setSelectedFile(newFile);
                }
            } else {
                toast({
                    title: 'Upload failed.',
                    description: `Could not process ${file.name}.`,
                    status: 'error',
                    duration: 5000,
                    isClosable: true,
                });
            }
        }
    }, [registerFile, toast, selectedFile, uploadedFiles]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'text/csv': ['.csv'] } });

    const handleDeleteFile = async (fileId) => {
        if (window.confirm('Are you sure you want to delete this dataset? This will remove it from DuckDB and free up resources.')) {
            deleteDataset(fileId);
            const fileMeta = uploadedFiles.find(f => f.id === fileId);
            if (fileMeta) {
                const sessionId = localStorage.getItem('sessionId');
                if (sessionId) {
                    try {
                        await fetch('/api/data/unregister-schema', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ session_id: sessionId, table_name: fileMeta.name })
                        });
                        console.log(`Unregistered schema for ${fileMeta.name}`);
                    } catch (e) {
                        console.error('Failed to unregister schema:', e);
                        toast({
                            title: 'Warning',
                            description: 'Dataset deleted locally, but backend schema may not have updated.',
                            status: 'warning',
                            duration: 5000,
                            isClosable: true,
                        });
                    }
                }
            }
            if (selectedFile?.id === fileId) setSelectedFile(null);
        }
    };

    const handleConnect = async (params) => {
        setIsConnecting(true);
        setConnectionParams(params);
        try {
            await testDBConnection('snowflake', params);
            const schema = await getDbSchema('snowflake', params);
            setDbSchema(schema);
            toast({ title: 'Connection successful', status: 'success' });
        } catch (error) {
            toast({ title: 'Connection failed', description: error.detail, status: 'error' });
            setDbSchema(null);
        } finally {
            setIsConnecting(false);
        }
    };
    
    const handleTableToggle = (table) => {
        setSelectedTables(prev => 
            prev.includes(table) ? prev.filter(t => t !== table) : [...prev, table]
        );
    }

    const handleDone = () => {
        if (selectedFile) {
            setActiveConnection({ type: 'file', file: selectedFile });
        } else if (connectionParams && selectedTables.length > 0) {
            setActiveConnection({
                type: 'database',
                connection_type: 'snowflake',
                params: connectionParams,
                tables: selectedTables,
            });
        }
        onClose();
    };

    return (
        <Modal isOpen={isOpen} onClose={onClose} size="6xl">
            <ModalOverlay />
            <ModalContent>
                <ModalHeader>Connect to Data Source</ModalHeader>
                <ModalCloseButton />
                <ModalBody>
                    <Tabs>
                        <TabList>
                            <Tab>Upload File</Tab>
                            <Tab>Connect Database</Tab>
                        </TabList>
                        <TabPanels>
                            <TabPanel>
                                <VStack spacing={4} align="stretch">
                                    <Box {...getRootProps()} p={10} border="2px dashed" borderColor={isDragActive ? 'blue.400' : 'gray.300'} borderRadius="md" textAlign="center" cursor="pointer">
                                        <input {...getInputProps()} />
                                        {isUploading ? <Spinner /> : <p>Drag 'n' drop files here, or click to select</p>}
                                    </Box>
                                    <UploadedFiles uploadedFiles={uploadedFiles} selectedFile={selectedFile} onSelectFile={setSelectedFile} onDeleteFile={handleDeleteFile} />
                                    <DataViewer selectedFile={selectedFile} />
                                </VStack>
                            </TabPanel>
                            <TabPanel>
                                <SnowflakeForm onConnect={handleConnect} isLoading={isConnecting} />
                                {dbSchema && (
                                    <HStack mt={4} spacing={4} align="start">
                                        {/* Schemas */}
                                        <Box flex={1} borderWidth="1px" borderRadius="md" p={2}>
                                            <Text fontWeight="bold">Schemas</Text>
                                            <VStack align="stretch">
                                                {Object.keys(dbSchema).map(s => (
                                                    <Button key={s} onClick={() => setSelectedSchema(s)} isActive={selectedSchema === s} variant="ghost">{s}</Button>
                                                ))}
                                            </VStack>
                                        </Box>
                                        {/* Tables */}
                                        <Box flex={1} borderWidth="1px" borderRadius="md" p={2}>
                                            <Text fontWeight="bold">Tables & Views</Text>
                                            {selectedSchema && (
                                                <VStack align="stretch">
                                                    {dbSchema[selectedSchema].map(t => (
                                                        <Checkbox key={t} isChecked={selectedTables.includes(t)} onChange={() => handleTableToggle(t)}>{t}</Checkbox>
                                                    ))}
                                                </VStack>
                                            )}
                                        </Box>
                                        {/* Selected */}
                                        <Box flex={1} borderWidth="1px" borderRadius="md" p={2}>
                                            <Text fontWeight="bold">Selected for Chat</Text>
                                            <VStack align="stretch">
                                                {selectedTables.map(t => <Text key={t}>{t}</Text>)}
                                            </VStack>
                                        </Box>
                                    </HStack>
                                )}
                            </TabPanel>
                        </TabPanels>
                    </Tabs>
                </ModalBody>
                <ModalFooter>
                    <Button variant="ghost" mr={3} onClick={onClose}>Cancel</Button>
                    <Button colorScheme="blue" onClick={handleDone}>Done</Button>
                </ModalFooter>
            </ModalContent>
        </Modal>
    );
};

export default DataConnectionModal; 