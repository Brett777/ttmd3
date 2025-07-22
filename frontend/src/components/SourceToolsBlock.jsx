import React, { useState } from 'react';
import { Box, Flex, HStack, VStack, Text, Icon, Badge, Collapse, useColorModeValue, Tooltip } from '@chakra-ui/react';
import { FiChevronDown, FiChevronUp, FiFileText } from 'react-icons/fi';

const toolIconMap = {
  get_weather: '🌤️',
  get_stock_price: '💰',
  search_the_web: '🔍',
  analyze_documents: '📄',
};

function ToolChip({ tool }) {
  const bg = useColorModeValue('gray.100', 'gray.700');
  const txt = useColorModeValue('gray.800', 'gray.200');
  const iconEmoji = toolIconMap[tool.name] || '🛠️';
  
  // Safely access arguments, defaulting to an empty object
  const args = tool.arguments || {};
  
  // Pretty-print arguments for the tooltip
  const tooltipLabel = Object.keys(args).length > 0 
    ? JSON.stringify(args, null, 2)
    : "No arguments";

  return (
    <Tooltip label={tooltipLabel} hasArrow>
      <HStack
        spacing={1.5} // Adjusted spacing
        px={2.5}      // Adjusted padding
        py={1}
        bg={bg}
        borderRadius="full"
        fontSize="xs"
        cursor="pointer"
      >
        <span>{iconEmoji}</span>
        <Text color={txt} fontWeight="medium">{tool.name}</Text>
      </HStack>
    </Tooltip>
  );
}

export default function SourceToolsBlock({ 
  tools = [], 
  citations = [], 
  documentsUsed = [], 
  reasoning = "", 
  usedMetadataFields = [],
  rawContentUsed = false,
  imageContentUsed = false 
}) {
  // Group citations by document to determine if we should render
  const citationsByDoc = citations.reduce((acc, c) => {
    const doc = c.document || 'Unknown';
    if (!acc[doc]) acc[doc] = [];
    acc[doc].push(c);
    return acc;
  }, {});

  const totalCount = tools.length + Object.keys(citationsByDoc).length + (documentsUsed.length > 0 ? 1 : 0);
  
  // Return early if there's nothing to show, BEFORE any hooks are called.
  if (totalCount === 0 && !reasoning) {
      return null;
  }
  
  // All hooks are now below the early return, ensuring they are always called.
  const [expanded, setExpanded] = useState(false);
  const toggle = () => setExpanded(!expanded);

  const [docsExpanded, setDocsExpanded] = useState(false);
  const toggleDocs = () => setDocsExpanded(!docsExpanded);

  const linkColor = useColorModeValue('gray.600', 'gray.400');
  const boxBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const reasoningBg = useColorModeValue('yellow.50', 'yellow.900');
  const reasoningBorder = useColorModeValue('yellow.400', 'yellow.600');
  const reasoningText = useColorModeValue('yellow.800', 'yellow.200');

  return (
    <Box mt={3} borderWidth="1px" borderColor={borderColor} borderRadius="lg" bg={boxBg} fontSize="sm" overflow="hidden">
      <Flex
        px={4}
        py={2}
        align="center"
        justify="space-between"
        cursor="pointer"
        onClick={toggle}
        _hover={{ bg: useColorModeValue('gray.50', 'gray.700') }}
      >
        <HStack spacing={2.5} color={linkColor}>
          <Text fontSize="md">📑</Text>
          <Text fontWeight="medium">Sources & Tools ({totalCount})</Text>
        </HStack>
        <Icon as={expanded ? FiChevronUp : FiChevronDown} color={linkColor} />
      </Flex>
      <Collapse in={expanded} animateOpacity>
        <Box px={4} py={3} borderTopWidth="1px" borderColor={borderColor}>
          {/* Tools Section */}
          {tools.length > 0 && (
            <Box mb={4}>
              <Text fontWeight="bold" mb={2}>Tools Run</Text>
              <HStack spacing={2} wrap="wrap">
                {tools.map((t, idx) => (
                  <ToolChip key={idx} tool={t} />
                ))}
              </HStack>
            </Box>
          )}

          {/* Documents Consulted Section */}
          {documentsUsed.length > 0 && (
            <Box mb={4}>
              <Flex
                align="center"
                justify="space-between"
                cursor="pointer"
                onClick={toggleDocs}
                pb={docsExpanded ? 2 : 0}
              >
                <Text fontWeight="bold">Documents Consulted ({documentsUsed.length})</Text>
                <Icon as={docsExpanded ? FiChevronUp : FiChevronDown} color={linkColor} />
              </Flex>
              <Collapse in={docsExpanded} animateOpacity>
                <VStack align="stretch" spacing={3} pt={2}>
                  {documentsUsed.map((doc, idx) => (
                    <HStack key={idx} spacing={2}>
                      <Icon as={FiFileText} color="gray.500" />
                      <Text fontWeight="semibold" fontSize="sm">{doc}</Text>
                    </HStack>
                  ))}
                </VStack>
              </Collapse>
            </Box>
          )}

          {/* Citations Section */}
          {Object.keys(citationsByDoc).length > 0 && (
            <Box mb={4}>
              <Text fontWeight="bold" mb={2}>Document Citations</Text>
              <VStack align="stretch" spacing={3}>
                {Object.entries(citationsByDoc).map(([doc, cites]) => (
                  <Box key={doc}>
                    <HStack spacing={2} mb={1.5}>
                      <Icon as={FiFileText} color="gray.500" />
                      <Text fontWeight="semibold">{doc}</Text>
                    </HStack>
                    <HStack spacing={2} wrap="wrap">
                      {cites.map((c, i) => (
                        <Badge key={i} variant="subtle" colorScheme="green" fontSize="xs" px={2} py={0.5}>
                          Page {c.page} {c.raw_text_used && '📝'}{c.image_used && '🖼️'}
                        </Badge>
                      ))}
                    </HStack>
                  </Box>
                ))}
              </VStack>
            </Box>
          )}

          {/* Analysis Details Section */}
          {usedMetadataFields.length > 0 && (
            <Box mb={4}>
              <Text fontWeight="bold" mb={2}>Analysis Details</Text>
              <VStack align="stretch" spacing={2} fontSize="xs">
                <HStack>
                  <Text fontWeight="semibold" minW="140px">Metadata Fields Used:</Text>
                  <HStack wrap="wrap" spacing={1}>
                    {usedMetadataFields.map((field, i) => (
                      <Badge key={i} variant="outline" colorScheme="green">{field}</Badge>
                    ))}
                  </HStack>
                </HStack>
                <HStack>
                  <Text fontWeight="semibold" minW="140px">Raw Text Consulted:</Text>
                  {rawContentUsed ? (
                    <Badge colorScheme="green">Yes</Badge>
                  ) : (
                    <Text color="gray.500" fontSize="sm">No</Text>
                  )}
                </HStack>
                <HStack>
                  <Text fontWeight="semibold" minW="140px">Image Content Consulted:</Text>
                  {imageContentUsed ? (
                    <Badge colorScheme="green">Yes</Badge>
                  ) : (
                    <Text color="gray.500" fontSize="sm">No</Text>
                  )}
                </HStack>
              </VStack>
            </Box>
          )}

          {/* Reasoning Trace - Now always visible when expanded */}
          {reasoning && (
            <Box>
              <Text fontWeight="bold" mb={2}>Reasoning trace</Text>
              <Box
                fontSize="sm"
                whiteSpace="pre-wrap"
                bg={reasoningBg}
                p={3}
                borderRadius="md"
                borderLeftWidth="4px"
                borderColor={reasoningBorder}
                color={reasoningText}
              >
                {reasoning}
              </Box>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
} 