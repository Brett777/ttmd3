import React from 'react';
import {
    IconButton,
    Tooltip,
    Icon,
    HStack,
} from '@chakra-ui/react';
import { FaRegImage } from 'react-icons/fa';
import { FiFile, FiDatabase } from 'react-icons/fi';

const ActionButtons = ({
    visionCapable,
    selectedImages,
    loading,
    onDocModalOpen,
    onDataModalOpen,
    fileInputRef,
}) => {
    return (
        <HStack spacing={1}>
            <Tooltip label="Attach images (max 3)">
                <IconButton
                    icon={<Icon as={FaRegImage} boxSize={5} />}
                    aria-label="Add image"
                    onClick={() => fileInputRef.current?.click()}
                    isDisabled={!visionCapable || selectedImages.length >= 3 || loading}
                    variant="ghost"
                    colorScheme={visionCapable ? "gray" : "gray"}
                    opacity={visionCapable ? 1 : 0.4}
                    title={visionCapable ? "Add photos (max 3)" : "Selected model does not support images"}
                />
            </Tooltip>
            <Tooltip label="Upload documents for analysis">
                <IconButton
                    icon={<Icon as={FiFile} boxSize={5} />}
                    aria-label="Upload document"
                    onClick={onDocModalOpen}
                    variant="ghost"
                    title="Upload and process documents"
                />
            </Tooltip>
            <Tooltip label="Connect to data">
                <IconButton
                    icon={<Icon as={FiDatabase} boxSize={5} />}
                    aria-label="Connect to data"
                    onClick={onDataModalOpen}
                    variant="ghost"
                    title="Connect to data sources"
                />
            </Tooltip>
        </HStack>
    );
};

export default ActionButtons; 