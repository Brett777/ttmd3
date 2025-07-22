/**
 * Citation Block Component
 * Displays document citations in an expandable format
 */

import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Icon,
  Collapse,
  useColorModeValue,
  Badge,
  Flex,
} from '@chakra-ui/react';
import { FiChevronDown, FiChevronUp, FiFileText } from 'react-icons/fi';

export function CitationBlock({ citations, documentsUsed }) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const bgColor = useColorModeValue('blue.50', 'blue.900');
  const borderColor = useColorModeValue('blue.200', 'blue.700');
  const textColor = useColorModeValue('blue.800', 'blue.100');
  const citationBg = useColorModeValue('white', 'gray.800');
  
  if (!citations || citations.length === 0) return null;
  
  // Group citations by document
  const citationsByDoc = citations.reduce((acc, citation) => {
    const doc = citation.document || 'Unknown';
    if (!acc[doc]) acc[doc] = [];
    acc[doc].push(citation);
    return acc;
  }, {});
  
  return (
    <Box
      mt={2}
      p={3}
      bg={bgColor}
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="md"
      fontSize="sm"
    >
      <Flex
        justify="space-between"
        align="center"
        cursor="pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <HStack spacing={2}>
          <Icon as={FiFileText} color={textColor} />
          <Text color={textColor} fontWeight="medium">
            {citations.length} source{citations.length !== 1 ? 's' : ''} referenced
          </Text>
          {documentsUsed && documentsUsed.length > 0 && (
            <Badge colorScheme="blue" variant="subtle" size="sm">
              {documentsUsed.length} document{documentsUsed.length !== 1 ? 's' : ''}
            </Badge>
          )}
        </HStack>
        <Icon
          as={isExpanded ? FiChevronUp : FiChevronDown}
          color={textColor}
        />
      </Flex>
      
      <Collapse in={isExpanded} animateOpacity>
        <VStack align="stretch" mt={3} spacing={2}>
          {Object.entries(citationsByDoc).map(([docName, docCitations]) => (
            <Box key={docName}>
              <Text fontWeight="bold" color={textColor} mb={1}>
                {docName}
              </Text>
              <VStack align="stretch" spacing={1}>
                {docCitations.map((citation, idx) => (
                  <Box
                    key={idx}
                    p={2}
                    bg={citationBg}
                    borderRadius="md"
                    borderLeftWidth="3px"
                    borderLeftColor="blue.400"
                  >
                    <HStack justify="space-between" mb={1}>
                      <Text fontSize="xs" color="gray.500">
                        Page {citation.page}
                      </Text>
                    </HStack>
                    {citation.snippet && (
                      <Text fontSize="sm" fontStyle="italic">
                        "{citation.snippet}"
                      </Text>
                    )}
                  </Box>
                ))}
              </VStack>
            </Box>
          ))}
        </VStack>
      </Collapse>
    </Box>
  );
}

export default CitationBlock; 