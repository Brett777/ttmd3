/**
 * Document Viewer Component
 * Displays document details, metadata, and page browser
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  Divider,
  SimpleGrid,
  Image,
  useColorModeValue,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Code,
  Button,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  IconButton,
  Tooltip,
  Flex,
  Icon,
} from '@chakra-ui/react';
import { FiFile, FiCalendar, FiUser, FiFileText, FiDownload, FiInfo } from 'react-icons/fi';

function PageThumbnail({ page, pageNumber, isSelected, onClick }) {
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const selectedBorderColor = useColorModeValue('blue.400', 'blue.300');
  const hoverBg = useColorModeValue('gray.100', 'gray.700');
  const selectedBg = useColorModeValue('blue.50', 'blue.900');
  
  // Helper to ensure proper base64 image format
  const getImageSrc = (imageData) => {
    if (!imageData) return null;
    if (imageData.startsWith('data:image')) return imageData;
    // Assume JPEG if not specified
    return `data:image/jpeg;base64,${imageData}`;
  };
  
  const imageData = page?.base64_image ?? page?.image ?? page?.preview_image;
  const imageSrc = getImageSrc(imageData);
  
  return (
    <Box
      borderWidth="1px"
      borderColor={isSelected ? selectedBorderColor : borderColor}
      borderRadius="md"
      p={2}
      cursor="pointer"
      onClick={onClick}
      bg={isSelected ? selectedBg : 'transparent'}
      _hover={{
        bg: isSelected ? selectedBg : hoverBg,
      }}
      transition="background-color 0.2s"
    >
      <VStack spacing={2}>
        {imageSrc ? (
          <Image
            src={imageSrc}
            alt={`Page ${pageNumber}`}
            w="120px"
            h="150px"
            objectFit="contain"
            bg={useColorModeValue('white', 'black')}
            fallback={
              <Box
                h="150px"
                w="120px"
                bg={useColorModeValue('gray.100', 'gray.700')}
                display="flex"
                alignItems="center"
                justifyContent="center"
                borderRadius="sm"
              >
                <Text color="gray.500" fontSize="sm">No preview</Text>
              </Box>
            }
          />
        ) : (
          <Box
            h="150px"
            w="120px"
            bg={useColorModeValue('gray.100', 'gray.700')}
            display="flex"
            alignItems="center"
            justifyContent="center"
            borderRadius="sm"
          >
            <Text color="gray.500" fontSize="sm">No preview</Text>
          </Box>
        )}
        <Text fontSize="xs" fontWeight="medium">
          Page {pageNumber}
        </Text>
      </VStack>
    </Box>
  );
}

function PageDetailView({ page, pageNumber }) {
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const [accordionIndex, setAccordionIndex] = useState([0]); // Default open 'Summary'

  if (!page) {
    return (
      <Flex
        justify="center"
        align="center"
        h="100%"
        bg={useColorModeValue('gray.50', 'gray.700')}
        borderRadius="md"
      >
        <VStack>
          <Icon as={FiInfo} boxSize={8} color="gray.400" />
          <Text color="gray.500">Select a page to view details</Text>
        </VStack>
      </Flex>
    );
  }

  // Helper to ensure proper base64 image format
  const getImageSrc = (imageData) => {
    if (!imageData) return null;
    if (imageData.startsWith('data:image')) return imageData;
    return `data:image/jpeg;base64,${imageData}`;
  };

  const imageData = page?.base64_image ?? page?.image ?? page?.preview_image;
  const imageSrc = getImageSrc(imageData);

  // Data fallbacks for older metadata versions
  const summaryText = page.one_sentence_summary || page.full_summary || page.summary || null;
  const topicsArray = page.main_topics || page.topics || [];
  const keywordsArray = page.keywords || [];
  const entitiesArray = page.entities || [];
  const keyInfoArray = page.key_information || [];
  const sentimentVal = page.sentiment || null;
  const acronymsObj = page.acronyms || null;
  const noteworthySentences = page.noteworthy_sentences || [];
  const visualElements = page.visual_elements || null;
  const chapterSection = page.chapter_or_section && page.chapter_or_section !== 'null' ? page.chapter_or_section : null;

  // Build a list of sections dynamically
  const extraSections = [];
  if (entitiesArray.length) extraSections.push({ label: 'Entities', type: 'list', data: entitiesArray, color: 'purple' });
  if (keyInfoArray.length) extraSections.push({ label: 'Key Information', type: 'list', data: keyInfoArray, color: 'teal' });
  if (sentimentVal) extraSections.push({ label: 'Sentiment', type: 'text', data: sentimentVal });
  if (acronymsObj && Object.keys(acronymsObj).length) extraSections.push({ label: 'Acronyms', type: 'dict', data: acronymsObj });
  if (noteworthySentences.length) extraSections.push({ label: 'Noteworthy Sentences', type: 'list', data: noteworthySentences });
  if (visualElements) extraSections.push({ label: 'Visual Elements', type: 'text', data: visualElements });
  if (chapterSection) extraSections.push({ label: 'Chapter / Section', type: 'text', data: chapterSection });

  const handleExpandAll = () => {
    const allIndexes = [0, 1, 2, 3 + extraSections.length];
    setAccordionIndex(allIndexes);
  };

  const handleCollapseAll = () => {
    setAccordionIndex([]);
  };

  return (
    <VStack align="stretch" spacing={4} h="100%" p={4}>
      {/* Page Image */}
      {imageSrc && (
        <Box>
          <Text fontWeight="bold" mb={2}>Page {pageNumber}</Text>
          <Image
            src={imageSrc}
            alt={`Page ${pageNumber}`}
            maxW="100%"
            borderRadius="md"
            borderWidth="1px"
            borderColor={borderColor}
          />
        </Box>
      )}

      {/* Accordion Controls */}
      <HStack justify="flex-end" spacing={2}>
        <Button size="xs" onClick={handleExpandAll}>Expand All</Button>
        <Button size="xs" onClick={handleCollapseAll}>Collapse All</Button>
      </HStack>

      <Accordion allowMultiple index={accordionIndex} onChange={setAccordionIndex}>
        {/* Summary */}
        {summaryText && (
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">Summary</Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <Text fontSize="sm">{summaryText}</Text>
            </AccordionPanel>
          </AccordionItem>
        )}

        {/* Topics */}
        {topicsArray && topicsArray.length > 0 && (
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">Topics</Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <HStack wrap="wrap" spacing={2}>
                {topicsArray.map((topic, idx) => (
                  <Badge key={idx} colorScheme="blue" variant="subtle">
                    {topic}
                  </Badge>
                ))}
              </HStack>
            </AccordionPanel>
          </AccordionItem>
        )}

        {/* Keywords */}
        {keywordsArray && keywordsArray.length > 0 && (
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">Keywords</Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <HStack wrap="wrap" spacing={2}>
                {keywordsArray.map((keyword, idx) => (
                  <Badge key={idx} colorScheme="green" variant="subtle">
                    {keyword}
                  </Badge>
                ))}
              </HStack>
            </AccordionPanel>
          </AccordionItem>
        )}

        {/* Raw Text */}
        {page.raw_text && (
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">Raw Text</Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <Box
                p={4}
                bg={useColorModeValue('gray.50', 'gray.800')}
                borderRadius="md"
                maxH="300px"
                overflowY="auto"
              >
                <Text whiteSpace="pre-wrap" fontSize="sm">
                  {page.raw_text}
                </Text>
              </Box>
            </AccordionPanel>
          </AccordionItem>
        )}

        {/* Extra dynamically detected sections */}
        {extraSections.map((section, idx) => (
          <AccordionItem key={`extra-${idx}`}>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">{section.label}</Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              {section.type === 'text' && (
                <Text fontSize="sm" whiteSpace="pre-wrap">{section.data}</Text>
              )}
              {section.type === 'list' && Array.isArray(section.data) && (
                <VStack align="start" spacing={2}>
                  {section.data.map((item, i) => (
                    section.color ? (
                      <Badge key={i} colorScheme={section.color} variant="subtle">{item}</Badge>
                    ) : (
                      <Text key={i} fontSize="sm">• {item}</Text>
                    )
                  ))}
                </VStack>
              )}
              {section.type === 'dict' && (
                <VStack align="start" spacing={1}>
                  {Object.entries(section.data).map(([abbr, meaning]) => (
                    <HStack key={abbr} spacing={2}>
                      <Badge colorScheme="blue" variant="solid">{abbr}</Badge>
                      <Text fontSize="sm">{meaning}</Text>
                    </HStack>
                  ))}
                </VStack>
              )}
            </AccordionPanel>
          </AccordionItem>
        ))}
      </Accordion>
    </VStack>
  );
}

export function DocumentViewer({ document }) {
  const [selectedTab, setSelectedTab] = useState(0);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedPageData, setSelectedPageData] = useState({ page: null, number: null });
  
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  const metadata = document.metadata || {};
  const details = metadata.document_details || {};
  const summary = metadata.document_summary || {};
  const pages = metadata.pages || {};
  
  // Set initial selected page when document changes or pages are loaded
  useEffect(() => {
    if (pages && Object.keys(pages).length > 0) {
      const firstPageNumber = Object.keys(pages).sort((a, b) => parseInt(a) - parseInt(b))[0];
      setSelectedPageData({
        page: pages[firstPageNumber],
        number: firstPageNumber,
      });
    } else {
      setSelectedPageData({ page: null, number: null });
    }
  }, [document]); // Dependency on document ensures this runs when the doc changes
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };
  
  // Format file size
  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };
  
  const handlePageClick = (page, pageNumber) => {
    setSelectedPageData({ page, number: pageNumber });
  };
  
  // Export metadata as JSON
  const handleExportMetadata = () => {
    const dataStr = JSON.stringify(metadata, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `${document.filename.replace(/\.[^/.]+$/, '')}_metadata.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };
  
  return (
    <Box p={6} bg={bgColor}>
      <VStack align="stretch" spacing={6}>
        {/* Header */}
        <Box>
          <Flex justify="space-between" align="start">
            <Box flex={1}>
              <HStack spacing={3} mb={2}>
                <FiFile size={24} />
                <Text fontSize="2xl" fontWeight="bold">
                  {details.title || document.filename}
                </Text>
              </HStack>
              
              <HStack spacing={4} fontSize="sm" color="gray.500">
                {details.author && (
                  <HStack>
                    <FiUser />
                    <Text>{details.author}</Text>
                  </HStack>
                )}
                <HStack>
                  <FiCalendar />
                  <Text>{formatDate(document.processedDate || document.uploadDate)}</Text>
                </HStack>
                <HStack>
                  <FiFileText />
                  <Text>{details.total_pages || 0} pages</Text>
                </HStack>
                <Text>•</Text>
                <Text>{formatFileSize(document.fileSize)}</Text>
              </HStack>
            </Box>
            
            <Tooltip label="Export metadata as JSON">
              <IconButton
                icon={<FiDownload />}
                size="sm"
                variant="outline"
                onClick={handleExportMetadata}
                aria-label="Export metadata"
              />
            </Tooltip>
          </Flex>
        </Box>
        
        <Divider />
        
        {/* Content Tabs */}
        <Tabs index={selectedTab} onChange={setSelectedTab}>
          <TabList>
            <Tab>Overview</Tab>
            <Tab>Pages</Tab>
            <Tab>Raw Metadata</Tab>
          </TabList>
          
          <TabPanels>
            {/* Overview Tab */}
            <TabPanel p={6} h="70vh" overflowY="auto">
              <VStack spacing={6} align="start">
                {/* Top-level summary */}
                <Box>
                  <Text fontSize="lg" fontWeight="bold" mb={2}>{details.title || document.filename}</Text>
                  <Text fontSize="sm" color={useColorModeValue('gray.600', 'gray.400')}>
                    {summary.short_summary}
                  </Text>
                </Box>
                <Divider />
                {/* Detailed Summary */}
                {summary.detailed_summary && (
                  <Box>
                    <Text fontWeight="bold" mb={2}>Detailed Summary</Text>
                    <Text fontSize="sm" whiteSpace="pre-wrap">{summary.detailed_summary}</Text>
                  </Box>
                )}
                {/* Main Topics */}
                {summary.main_topics && summary.main_topics.length > 0 && (
                  <Box>
                    <Text fontWeight="bold" mb={2}>Main Topics</Text>
                    <HStack wrap="wrap" spacing={2}>
                      {summary.main_topics.map((topic, idx) => (
                        <Badge key={idx} colorScheme="teal" variant="subtle">{topic}</Badge>
                      ))}
                    </HStack>
                  </Box>
                )}
                {/* Key Insights */}
                {summary.key_insights && summary.key_insights.length > 0 && (
                  <Box>
                    <Text fontWeight="bold" mb={2}>Key Insights</Text>
                    <VStack align="start" spacing={2}>
                      {summary.key_insights.map((insight, idx) => (
                        <HStack key={idx} align="start">
                          <Badge mt={1} colorScheme="blue" variant="solid" borderRadius="full" boxSize="1.2em" fontSize="0.6em" display="flex" alignItems="center" justifyContent="center">{idx + 1}</Badge>
                          <Text fontSize="sm">{insight}</Text>
                        </HStack>
                      ))}
                    </VStack>
                  </Box>
                )}
              </VStack>
            </TabPanel>
            
            {/* Pages Tab */}
            <TabPanel p={0} h="70vh">
              {Object.keys(pages).length > 0 ? (
                <Flex h="100%">
                  {/* Left Panel: Thumbnails */}
                  <Box
                    w="200px"
                    h="100%"
                    overflowY="auto"
                    p={2}
                    borderRightWidth="1px"
                    borderColor={borderColor}
                    flexShrink={0}
                  >
                    <VStack spacing={2} align="stretch">
                      {Object.entries(pages)
                        .sort(([numA], [numB]) => parseInt(numA) - parseInt(numB))
                        .map(([pageNumber, page]) => (
                          <PageThumbnail
                            key={pageNumber}
                            page={page}
                            pageNumber={pageNumber}
                            isSelected={selectedPageData.number === pageNumber}
                            onClick={() => handlePageClick(page, pageNumber)}
                          />
                      ))}
                    </VStack>
                  </Box>
                  
                  {/* Right Panel: Detail View */}
                  <Box flex="1" h="100%" overflowY="auto">
                    <PageDetailView
                      page={selectedPageData.page}
                      pageNumber={selectedPageData.number}
                    />
                  </Box>
                </Flex>
              ) : (
                <Flex justify="center" align="center" h="100%">
                  <Text color="gray.500">No pages with previews are available.</Text>
                </Flex>
              )}
            </TabPanel>
            
            {/* Raw Metadata Tab */}
            <TabPanel p={6} h="70vh" overflowY="auto">
              <Box
                p={4}
                bg={useColorModeValue('gray.50', 'gray.900')}
                borderRadius="md"
                overflowX="auto"
              >
                <Code
                  display="block"
                  whiteSpace="pre"
                  fontSize="xs"
                  bg="transparent"
                >
                  {JSON.stringify(metadata, null, 2)}
                </Code>
              </Box>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Box>
  );
}

export default DocumentViewer; 