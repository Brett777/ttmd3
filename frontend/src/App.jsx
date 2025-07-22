import React, { useState, useEffect, useRef } from "react";
import {
  Box,
  Flex,
  Input,
  IconButton,
  VStack,
  Text,
  HStack,
  Select,
  Spinner,
  useColorMode,
  useColorModeValue,
  Avatar,
  Spacer,
  Image,
  useDisclosure,
  ChakraProvider,
  Container,
  Heading,
  Badge,
  Tooltip,
    Icon,
  Button,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
} from "@chakra-ui/react";
import { ArrowUpIcon, SunIcon, MoonIcon, SmallAddIcon, AttachmentIcon, CloseIcon, HamburgerIcon, EditIcon, DeleteIcon } from "@chakra-ui/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import axios from "axios";
import { FaRegImage } from "react-icons/fa";
import { FiSun, FiMoon, FiSettings, FiSend, FiFile, FiUpload, FiBook, FiMessageCircle, FiDatabase } from 'react-icons/fi';
import { useDocuments } from "./contexts/DocumentContext";
import SourceToolsBlock from "./components/SourceToolsBlock";
import DocumentSelectionModal from "./components/DocumentSelectionModal";
import { useLiveQuery } from "dexie-react-hooks";
import { db } from "./db";
import DocumentModal from "./components/DocumentModal";
import DataConnectionModal from "./components/DataConnectionModal";
import ActionButtons from "./components/ActionButtons";
import { useData } from "./contexts/DataContext";
import { generateSql, generateVisualizations } from "./services/dataApi";
import InChatAnalysisPanel from "./components/InChatAnalysisPanel";
import { useDataChat } from "./hooks/useDataChat";

import "./App.css";

import logo from "./assets/dr-logo-for-light-bg.svg";
import smallLogo from "/favicon.png";

axios.defaults.baseURL =
  import.meta.env.MODE === "production"
    ? import.meta.env.BASE_URL
    : "http://localhost:8080";

const MessageBubble = ({ role, content, images, citations, documentsUsed, reasoning, toolsUsed, usedMetadataFields, rawContentUsed, imageContentUsed, data_analysis_result }) => {
  const isUser = role === "user";
  
  // Hooks must be called unconditionally at the top level.
  const userMessageBg = useColorModeValue("blue.500", "blue.600");
  const assistantMessageBg = useColorModeValue("gray.100", "gray.700");
  const reasoningBg = useColorModeValue("yellow.50", "yellow.900");
  const reasoningBorder = useColorModeValue("yellow.400", "yellow.600");
  const reasoningText = useColorModeValue("yellow.800", "yellow.200");
  const reasoningSubText = useColorModeValue("yellow.700", "yellow.300");
  const messageBg = isUser ? userMessageBg : assistantMessageBg;
  const messageColor = isUser ? "white" : "inherit";
  const boxShadow = useColorModeValue("sm", "lg");
  
  // Conditional rendering values can be derived from hooks.
  const userAvatar = <Avatar name="U" bg="purple.500" color="white" />;
  const assistantAvatar = <Avatar name="AI" bg="teal.500" color="white" />;

  return (
    <Flex w="100%" my={3}>
      <HStack
        spacing={4}
        alignItems="flex-start"
        w="100%"
        flexDir={isUser ? "row-reverse" : "row"}
      >
        {isUser ? userAvatar : assistantAvatar}
        <Box maxW="80%">
          <Box
            position="relative"
            bg={messageBg}
            color={messageColor}
            px={4}
            py={2}
            borderRadius="lg"
            className="markdown-content"
            boxShadow={boxShadow}
            sx={{
              "&:after": {
                content: '""',
                position: "absolute",
                top: "12px",
                borderTop: "8px solid transparent",
                borderBottom: "8px solid transparent",
                ...(isUser
                  ? {
                      right: "-8px",
                      borderLeft: "8px solid",
                      borderLeftColor: messageBg,
                    }
                  : {
                      left: "-8px",
                      borderRight: "8px solid",
                      borderRightColor: messageBg,
                    }),
              },
            }}
          >
            {images && images.length > 0 && (
              <HStack mb={2}>
                {images.map((src, idx) => (
                  <Image key={idx} src={src} boxSize="120px" objectFit="cover" borderRadius="md" />
                ))}
              </HStack>
            )}
            {role === 'assistant' && !content ? (
              <Spinner size="sm" />
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {(() => {
                  const rawText = typeof content === 'string' ? content : content.text;
                  return data_analysis_result
                    ? rawText.replace(/```sql[\s\S]*?```/g, '').trim()
                    : rawText;
                })()}
              </ReactMarkdown>
            )}
            
            {role === 'assistant' && data_analysis_result && (
              <InChatAnalysisPanel {...data_analysis_result} />
            )}
            
            {role === 'assistant' && toolsUsed && (
              <SourceToolsBlock tools={toolsUsed} citations={citations} documentsUsed={documentsUsed} reasoning={reasoning} usedMetadataFields={usedMetadataFields} rawContentUsed={rawContentUsed} imageContentUsed={imageContentUsed} />
            )}
          </Box>
        </Box>
        <Spacer />
      </HStack>
    </Flex>
  );
};

const ChatApp = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const { isOpen: isSidebarOpen, onOpen: onSidebarOpen, onClose: onSidebarClose } = useDisclosure();
  const [renamingId, setRenamingId] = useState(null);
  const [newTitle, setNewTitle] = useState("");
  const [chatToDelete, setChatToDelete] = useState(null);
  const { isOpen: isAlertOpen, onOpen: onAlertOpen, onClose: onAlertClose } = useDisclosure();
  const cancelRef = useRef();
  const bottomRef = useRef(null);
  const { colorMode, toggleColorMode } = useColorMode();
  const bgColor = useColorModeValue("white", "gray.800");
  const containerBgColor = useColorModeValue("gray.50", "gray.900");
  const borderColor = useColorModeValue("gray.200", "gray.700");
  const hoverBg = useColorModeValue("gray.100", "gray.700");
  const [selectedImages, setSelectedImages] = useState([]);
  const fileInputRef = useRef(null);
  const textInputRef = useRef(null);
  
  // Document modal state
  const { isOpen: isDocModalOpen, onOpen: onDocModalOpen, onClose: onDocModalClose } = useDisclosure();

  // Data connection modal state
  const { isOpen: isDataModalOpen, onOpen: onDataModalOpen, onClose: onDataModalClose } = useDisclosure();
  
  // Document selection modal state
  const { isOpen: isDocSelectOpen, onOpen: onDocSelectOpen, onClose: onDocSelectClose } = useDisclosure();
  
  // Document context
  const { documents, selectedDocumentIds, syncStatus } = useDocuments();
  const { activeConnection, query } = useData();
  const { analyzeData, isLoading: isAnalyzing, error: analysisError } = useDataChat();
  
  // Combine loading states
  const isProcessing = loading || isAnalyzing;

  // Get completed documents count
  const completedDocsCount = documents.filter(doc => doc.status === 'completed').length;
  const selectedDocsCount = selectedDocumentIds.length;
  
  // Document context indicator colors - must be called unconditionally
  const docIndicatorBg = useColorModeValue('blue.50', 'blue.900');
  const docIndicatorBorder = useColorModeValue('blue.200', 'blue.700');
  const docIndicatorText = useColorModeValue('blue.800', 'blue.100');

  const conversations = useLiveQuery(() => db.conversations.orderBy('lastModified').reverse().toArray());

  // Ensure chat and document processing share the SAME browser session id
  const getSessionId = () => {
    let id = localStorage.getItem('sessionId');
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem('sessionId', id);
    }
    return id;
  };

  // Use useRef to guarantee sessionId is stable for the lifetime of the app
  const sessionIdRef = useRef(getSessionId());

  useEffect(() => {
    const initChat = async () => {
      if (activeConversationId) {
        const conversation = await db.conversations.get(activeConversationId);
        if (conversation) {
          setMessages(conversation.messages);
        }
      } else {
        // Initial greeting for a new chat
        setMessages([
          {
            role: "assistant",
            content:
              "Hello! I'm an agentic AI assistant. I can help you with weather, stock prices, and general questions. How can I assist you today?",
          },
        ]);
      }
    };
    initChat();
  }, [activeConversationId]);

  useEffect(() => {
    fetch('api/models')
      .then((res) => res.json())
      .then((data) => {
        const fetchedModels = data.models || [];
        
        // Define the desired order for OpenAI models by name
        const openAIOrder = [
          'gpt-4.1',
          'gpt-4o',
          'gpt-4.1-mini',
          'gpt-4o-mini',
          'gpt-4.1-nano'
        ];

        // Sort the models
        const sortedModels = fetchedModels.sort((a, b) => {
          const isA_OpenAI = a.provider === 'OpenAI';
          const isB_OpenAI = b.provider === 'OpenAI';

          // Prioritize OpenAI models to be at the top
          if (isA_OpenAI && !isB_OpenAI) return -1;
          if (!isA_OpenAI && isB_OpenAI) return 1;

          // If both are OpenAI, use the custom order
          if (isA_OpenAI && isB_OpenAI) {
            const indexA = openAIOrder.indexOf(a.name);
            const indexB = openAIOrder.indexOf(b.name);

            if (indexA !== -1 && indexB !== -1) {
              return indexA - indexB; // Both are in the custom order list
            }
            if (indexA !== -1) return -1; // Only A is in the list
            if (indexB !== -1) return 1;  // Only B is in the list
            return a.name.localeCompare(b.name); // Fallback for other OpenAI models
          }
          
          // For non-OpenAI models, sort by provider then by name
          if (a.provider < b.provider) return -1;
          if (a.provider > b.provider) return 1;
          return a.name.localeCompare(b.name);
        });
        
        setModels(sortedModels);
        if (sortedModels.length > 0) {
          setSelectedModel(sortedModels[0].id);
        }
      })
      .catch((err) => console.error("Failed to fetch models:", err));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    // When a message is done sending, focus the input
    if (!loading) {
      textInputRef.current?.focus();
    }
  }, [loading]);

  const addImages = (files) => {
    const newItems = [];
    const remaining = 3 - selectedImages.length;
    Array.from(files)
      .slice(0, remaining)
      .forEach((file) => {
        if (!file.type.startsWith("image/")) return;
        if (file.size > 5 * 1024 * 1024) return; // skip >5MB
        const preview = URL.createObjectURL(file);
        newItems.push({ file, preview });
      });
    if (newItems.length) {
      setSelectedImages((prev) => [...prev, ...newItems]);
    }
  };

  const handleFileChange = (e) => {
    addImages(e.target.files);
    e.target.value = ""; // reset
  };

  const removeImage = (idx) => {
    setSelectedImages((prev) => {
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
  };

  const selectedModelInfo = models.find((m) => m.id === selectedModel);
  const visionCapable = selectedModelInfo?.vision;

  const handleNewChat = () => {
    setActiveConversationId(null);
    setMessages([
      {
        role: "assistant",
        content: "Hello! How can I help you today?",
      },
    ]);
    onSidebarClose();
  };

  const handleRename = (convo) => {
    setRenamingId(convo.id);
    setNewTitle(convo.title);
  };

  const handleSaveRename = async (id) => {
    if (newTitle.trim()) {
      await db.conversations.update(id, { title: newTitle.trim() });
    }
    setRenamingId(null);
    setNewTitle("");
  };

  const handleDelete = async () => {
    if (chatToDelete) {
      await db.conversations.delete(chatToDelete.id);
      if (activeConversationId === chatToDelete.id) {
        handleNewChat();
      }
      setChatToDelete(null);
      onAlertClose();
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const sendMessage = async () => {
    if (!input.trim() && selectedImages.length === 0) return;

    const newConversationId = activeConversationId || crypto.randomUUID();
    const isNewConversation = !activeConversationId;

    const userMessage = {
      role: "user",
      content: input,
      images: selectedImages.map(f => f.preview),
    };
    
    // Add a placeholder for the assistant's response
    const assistantMessagePlaceholder = {
          role: "assistant",
      content: "",
      citations: [],
      documentsUsed: [],
      reasoning: "",
      toolsUsed: null,
      usedMetadataFields: [],
      rawContentUsed: false,
      imageContentUsed: false,
      data_analysis_result: null,
    };

    const updatedMessages = [...messages, userMessage, assistantMessagePlaceholder];
    setMessages(updatedMessages);

    // Persist immediately for responsiveness
    await db.conversations.put({
      id: newConversationId,
      title: isNewConversation ? (input.substring(0, 40) || "New Chat") : (conversations.find(c => c.id === newConversationId)?.title || "Chat"),
      messages: updatedMessages,
              lastModified: new Date(),
            });

    if (isNewConversation) {
      setActiveConversationId(newConversationId);
    }
    
    const inputForApi = input;
    setInput("");
    setSelectedImages([]);
    fileInputRef.current.value = ""; // Reset file input
    setLoading(true);

    const formData = new FormData();
    formData.append("message", inputForApi);
    formData.append("model", selectedModel);
    formData.append("session_id", sessionIdRef.current);

    // Filter out placeholder and non-API compliant fields from history
    const historyForApi = messages.map(({ role, content }) => ({ role, content }));
    
    // Check if this conversation has previously used data analysis tools
    const hasDataAnalysisHistory = messages.some(msg => 
      msg.toolsUsed && msg.toolsUsed.some(tool => tool.name === 'talk_to_data')
    );
    
    // Add context about previous data analysis if it exists
    if (hasDataAnalysisHistory) {
      formData.append("has_data_analysis_history", "true");
    }
    
    formData.append("conversation_history", JSON.stringify(historyForApi));

    selectedImages.forEach((file) => {
      formData.append("images", file);
    });

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      let currentAssistantMessage = { ...assistantMessagePlaceholder };

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunk = decoder.decode(value, { stream: true });
        
        // Process SSE chunks
        const lines = chunk.split('\n\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const json = JSON.parse(line.substring(6));
              
              if (json.response) {
                currentAssistantMessage.content += json.response;
              }
              // The final metadata chunk will not have 'response' but other keys
              if (json.citations) currentAssistantMessage.citations = json.citations;
              if (json.documentsUsed) currentAssistantMessage.documentsUsed = json.documentsUsed;
              if (json.reasoning) currentAssistantMessage.reasoning = json.reasoning;
              if (json.toolsUsed) currentAssistantMessage.toolsUsed = json.toolsUsed;
              if (json.usedMetadataFields) currentAssistantMessage.usedMetadataFields = json.usedMetadataFields;
              if (json.rawContentUsed) currentAssistantMessage.rawContentUsed = json.rawContentUsed;
              if (json.imageContentUsed) currentAssistantMessage.imageContentUsed = json.imageContentUsed;
              if (json.data_analysis_result) currentAssistantMessage.data_analysis_result = json.data_analysis_result;

            } catch (e) {
              console.error("Error parsing stream JSON:", e);
              console.error("Problematic line:", line);
            }
          }
        }

        // Update state with the progressively built message
        setMessages(prev => [...prev.slice(0, -1), currentAssistantMessage]);
      }
      
      // After streaming is complete, check if we need to execute SQL for data analysis
      if (currentAssistantMessage.data_analysis_result?.requiresExecution && activeConnection) {
        try {
            console.log('Executing SQL for data analysis:', currentAssistantMessage.data_analysis_result.sql);
            
            // Execute SQL using DuckDB
            const dataResult = await query(currentAssistantMessage.data_analysis_result.sql);
            
            // New: Check for null or empty results
            if (!dataResult) {
                const errorDetails = `Null result from query - possible execution error: ${currentAssistantMessage.data_analysis_result.sql}`;
                console.error(errorDetails);
                try {
                    await axios.post('/api/log-error', { error: 'Null query result', sql: currentAssistantMessage.data_analysis_result.sql, context: 'Chat data analysis' });
                } catch (logError) { console.warn('Failed to log:', logError); }
                // Proceed to error handling below
                throw new Error('Null query result');
            }
            
            const queryData = dataResult.toArray().map(row => row.toJSON());
            console.log('Query results:', queryData);
            
            if (queryData.length === 0) {
                const warnDetails = `Empty result set from query - possible invalid filters or data: ${currentAssistantMessage.data_analysis_result.sql}`;
                console.warn(warnDetails);
                try {
                    await axios.post('/api/log-error', { error: 'Empty result set', sql: currentAssistantMessage.data_analysis_result.sql, context: 'Chat data analysis (soft error)' });
                } catch (logError) { console.warn('Failed to log:', logError); }
            }
            
            // New: Check for 'fake error' in results
            if (queryData.length === 1) {
                const firstRow = queryData[0];
                const keys = Object.keys(firstRow);
                const values = Object.values(firstRow).join(' ').toLowerCase();
                if (keys.some(k => k.toLowerCase().includes('error')) ||
                    values.includes('error') || values.includes('does not exist') || values.includes('not found')) {
                    const errorDetails = `Detected error in query result: ${JSON.stringify(firstRow)} - SQL: ${currentAssistantMessage.data_analysis_result.sql}`;
                    console.error(errorDetails);
                    try {
                        await axios.post('/api/log-error', {
                            error: 'Detected error in result data',
                            sql: currentAssistantMessage.data_analysis_result.sql,
                            context: 'Chat data analysis (fake error)',
                            result: firstRow
                        });
                    } catch (logError) {
                        console.warn('Failed to log:', logError);
                    }
                    throw new Error('Detected error in query result');
                }
            }
            
            // Generate visualization
                let chartConfig = null;
                try {
                    const vizResponse = await generateVisualizations(inputForApi, currentAssistantMessage.data_analysis_result.sql, queryData);
                    chartConfig = vizResponse ? vizResponse.chart_config : null;
                } catch (vizError) {
                    const vizErrorDetails = `Error generating visualization: ${vizError.message} - SQL: ${currentAssistantMessage.data_analysis_result.sql} - Stack: ${vizError.stack}`;
                    console.error(vizErrorDetails);
                    
                    // Send to backend
                    try {
                        await axios.post('/api/log-error', {
                            error: vizError.message,
                            sql: currentAssistantMessage.data_analysis_result.sql,
                            context: 'Visualization generation'
                        });
                    } catch (logError) {
                        console.warn('Failed to log viz error to backend:', logError);
                    }
                }
                
                // Update the message with complete results
                const updatedAnalysisResult = {
                  ...currentAssistantMessage.data_analysis_result,
                  data: queryData,
                  chartConfig: chartConfig,
                  requiresExecution: false  // Mark as executed
                };
                
                const finalMessage = {
                  ...currentAssistantMessage,
                  data_analysis_result: updatedAnalysisResult
                };
                
                // Update the message in state
                setMessages(prev => [...prev.slice(0, -1), finalMessage]);
                
                // Update in database
                await db.conversations.update(newConversationId, { 
                  messages: [...messages.slice(0, -1), finalMessage] 
                });
                
                console.log('Data analysis completed with', queryData.length, 'rows and chart config:', !!chartConfig);
            } catch (execError) {
                const errorDetails = `Error executing data analysis SQL: ${execError.message} - SQL: ${currentAssistantMessage.data_analysis_result.sql} - Stack: ${execError.stack}`;
                console.error(errorDetails);
                
                // Send to backend for terminal logging
                try {
                    await axios.post('/api/log-error', {
                        error: execError.message,
                        sql: currentAssistantMessage.data_analysis_result.sql,
                        context: 'Chat data analysis execution'
                    });
                } catch (logError) {
                    console.warn('Failed to log error to backend:', logError);
                }
                
                // Update with error message
                const errorAnalysisResult = {
                  ...currentAssistantMessage.data_analysis_result,
                  data: [],
                  chartConfig: null,
                  requiresExecution: false,
                  error: `Failed to execute query: ${execError.message}`
                };
                
                const errorMessage = {
                  ...currentAssistantMessage,
                  data_analysis_result: errorAnalysisResult
                };
                
                setMessages(prev => [...prev.slice(0, -1), errorMessage]);
                await db.conversations.update(newConversationId, { 
                  messages: [...messages.slice(0, -1), errorMessage] 
                });
            }
      } else {
        // Normal persistence for non-data-analysis messages
        await db.conversations.update(newConversationId, { messages: [...messages.slice(0, -1), currentAssistantMessage] });
      }

    } catch (error) {
      console.error("Error sending message:", error);
       const errorResponseMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again.",
        citations: [],
        documentsUsed: [],
        reasoning: "",
        toolsUsed: null,
        usedMetadataFields: [],
        rawContentUsed: false,
        imageContentUsed: false,
        data_analysis_result: null,
      };
      setMessages(prev => [...prev.slice(0, -1), errorResponseMessage]);
      await db.conversations.update(newConversationId, { messages: [...messages.slice(0, -1), errorResponseMessage] });
    } finally {
      setLoading(false);
    }
  };

  return (
    <ChakraProvider>
      <Flex h="100vh" bg={containerBgColor}>
        <DataConnectionModal isOpen={isDataModalOpen} onClose={onDataModalClose} />
        <DocumentModal isOpen={isDocModalOpen} onClose={onDocModalClose} />
        <DocumentSelectionModal isOpen={isDocSelectOpen} onClose={onDocSelectClose} />
        <Drawer placement="left" onClose={onSidebarClose} isOpen={isSidebarOpen}>
          <DrawerOverlay />
          <DrawerContent bg={bgColor} maxW="280px">
            <DrawerCloseButton />
            <DrawerHeader borderBottomWidth="1px" borderColor={borderColor}>
              Chat History
            </DrawerHeader>
            <DrawerBody p={2}>
              <VStack align="stretch" spacing={2}>
                <Button
                  leftIcon={<FiMessageCircle />}
                  onClick={handleNewChat}
                  variant="outline"
                  mb={2}
                  size="sm"
                >
                  Create new chat
                </Button>
                {conversations?.map((convo) => (
                  <Flex
                    key={convo.id}
                    className="chat-item"
                    p={2}
                    borderRadius="md"
                    bg={activeConversationId === convo.id ? "blue.600" : "transparent"}
                    color={activeConversationId === convo.id ? "white" : "inherit"}
                    cursor="pointer"
                    onClick={() => renamingId !== convo.id && setActiveConversationId(convo.id)}
                    _hover={{ bg: activeConversationId !== convo.id && hoverBg }}
                    align="center"
                    justify="space-between"
                  >
                    {renamingId === convo.id ? (
                      <Input
                        size="sm"
                        value={newTitle}
                        onChange={(e) => setNewTitle(e.target.value)}
                        onBlur={() => handleSaveRename(convo.id)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSaveRename(convo.id)}
                        autoFocus
                      />
                    ) : (
                      <Text
                        whiteSpace="nowrap"
                        overflow="hidden"
                        textOverflow="ellipsis"
                        flex="1"
                      >
                        {convo.title}
                      </Text>
                    )}
                    <HStack spacing={1} className="action-buttons">
                      <IconButton
                        icon={<EditIcon />}
                        size="xs"
                        variant="ghost"
                        aria-label="Rename chat"
                        onClick={(e) => { e.stopPropagation(); handleRename(convo); }}
                      />
                      <IconButton
                        icon={<DeleteIcon />}
                        size="xs"
                        variant="ghost"
                        aria-label="Delete chat"
                        onClick={(e) => { e.stopPropagation(); setChatToDelete(convo); onAlertOpen(); }}
                      />
                    </HStack>
                  </Flex>
                ))}
              </VStack>
            </DrawerBody>
          </DrawerContent>
        </Drawer>

        <AlertDialog
          isOpen={isAlertOpen}
          leastDestructiveRef={cancelRef}
          onClose={onAlertClose}
        >
          <AlertDialogOverlay>
            <AlertDialogContent>
              <AlertDialogHeader fontSize="lg" fontWeight="bold">
                Delete Chat
              </AlertDialogHeader>
              <AlertDialogBody>
                Are you sure you want to delete the chat "{chatToDelete?.title}"? This action cannot be undone.
              </AlertDialogBody>
              <AlertDialogFooter>
                <Button ref={cancelRef} onClick={onAlertClose}>
                  Cancel
                </Button>
                <Button colorScheme="red" onClick={handleDelete} ml={3}>
                  Delete
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>

        <Flex className="chat-layout" flex="1">
          {/* Header */}
          <Box>
            <HStack
              className="chat-header"
              p={4}
              justify="space-between"
              borderBottomWidth="1px"
              bg={bgColor}
              borderColor={borderColor}
            >
              <HStack gap={4}>
                <IconButton
                  icon={<HamburgerIcon />}
                  aria-label="Open chat history"
                  onClick={onSidebarOpen}
                  size="sm"
                />
                <Tooltip label="Create new chat" placement="bottom">
                  <IconButton
                    icon={<FiMessageCircle />}
                    aria-label="New chat"
                    onClick={handleNewChat}
                    size="sm"
                    variant="outline"
                  />
                </Tooltip>
                <Text fontWeight="bold">Model</Text>
                <Select
                  size="sm"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  borderRadius="md"
                >
                  {Object.entries(
                    models.reduce((acc, m) => {
                      const prov = m.provider || "Other";
                      acc[prov] = acc[prov] ? [...acc[prov], m] : [m];
                      return acc;
                    }, {})
                  ).map(([provider, list]) => (
                    <optgroup key={provider} label={provider} style={{ fontWeight: "bold" }}>
                      {list.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </Select>
              </HStack>
              
              <HStack spacing={3}>
                {/* Document Analysis Indicator */}
                {completedDocsCount > 0 && (
                  <Tooltip 
                    label={`${selectedDocsCount} document${selectedDocsCount !== 1 ? 's' : ''} available for analysis`}
                    placement="bottom-end"
                    maxW="220px"
                    whiteSpace="normal"
                  >
                    <HStack spacing={2}>
                      <Icon as={FiBook} color="blue.500" />
                      <Badge 
                        colorScheme="blue" 
                        variant="subtle"
                        cursor="pointer"
                        onClick={onDocSelectOpen}
                        _hover={{ opacity: 0.8 }}
                      >
                        {selectedDocsCount} doc{selectedDocsCount !== 1 ? 's' : ''}
                      </Badge>
                      {syncStatus === 'syncing' && <Spinner size="xs" color="blue.500" />}
                    </HStack>
                  </Tooltip>
                )}
                
                <IconButton
                  aria-label="toggle theme"
                  icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
                  onClick={toggleColorMode}
                  size="sm"
                />
              </HStack>
            </HStack>
          </Box>

          {/* Messages */}
          <Flex className="chat-messages" flex="1" p={[0, 4]} overflowY="auto" direction="column">
            <VStack align="stretch" spacing={0} w="100%" maxW="800px" mx="auto">
              {/* Document Analysis Indicator */}
              {selectedDocsCount > 0 && (
                <Box
                  mb={4}
                  p={3}
                  bg={docIndicatorBg}
                  borderRadius="md"
                  borderWidth="1px"
                  borderColor={docIndicatorBorder}
                >
                  <HStack>
                    <Icon as={FiBook} color="blue.500" />
                    <Text fontSize="sm" color={docIndicatorText}>
                      {selectedDocsCount} document{selectedDocsCount !== 1 ? 's' : ''} available for analysis. 
                      Responses will include relevant information from your documents.
                    </Text>
                    <Spacer />
                    <Button
                      size="xs"
                      variant="ghost"
                      colorScheme="blue"
                      onClick={onDocSelectOpen}
                    >
                      Manage
                    </Button>
                    {syncStatus === 'syncing' && (
                      <HStack spacing={1}>
                        <Spinner size="xs" color="blue.500" />
                        <Text fontSize="xs" color="blue.500">Syncing...</Text>
                      </HStack>
                    )}
                  </HStack>
                </Box>
              )}
              
              {messages.map((m, idx) => (
                <MessageBubble 
                  key={idx} 
                  role={m.role} 
                  content={m.content} 
                  images={m.images}
                  citations={m.citations}
                  documentsUsed={m.documentsUsed}
                  reasoning={m.reasoning}
                  toolsUsed={m.toolsUsed}
                  usedMetadataFields={m.usedMetadataFields}
                  rawContentUsed={m.rawContentUsed}
                  imageContentUsed={m.imageContentUsed}
                  data_analysis_result={m.data_analysis_result}
                />
              ))}
              <div ref={bottomRef} />
            </VStack>
          </Flex>

          {/* Input */}
          <Box p={4} bg={bgColor} borderTopWidth="1px" borderColor={borderColor}>
            <HStack maxW="800px" mx="auto" alignItems="flex-end" gap={2}>
              <VStack align="stretch" spacing={2} flex="1">
                <Box pl={1} display="flex" alignItems="center">
                  {selectedImages.length > 0 && (
                    <HStack spacing={2} pr={2} py={2}>
                      {selectedImages.map((img, idx) => (
                        <Box key={idx} position="relative">
                          <Image src={img.preview} boxSize="48px" objectFit="cover" borderRadius="md" />
                          <IconButton
                            size="xs"
                            icon={<CloseIcon boxSize={2} />}
                            position="absolute"
                            top="-4px"
                            right="-4px"
                            onClick={() => removeImage(idx)}
                            borderRadius="full"
                            colorScheme="red"
                            variant="solid"
                          />
                        </Box>
                      ))}
                    </HStack>
                  )}
                </Box>

                <Flex align="flex-end" gap={2}>
                  <ActionButtons
                    visionCapable={visionCapable}
                    selectedImages={selectedImages}
                    loading={isProcessing}
                    onDocModalOpen={onDocModalOpen}
                    onDataModalOpen={onDataModalOpen}
                    fileInputRef={fileInputRef}
                  />
                  <Input
                    ref={textInputRef}
                    placeholder="Type a message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKey}
                    borderRadius="lg"
                    size="lg"
                    isDisabled={isProcessing}
                    flex="1"
                  />
                  <IconButton
                    colorScheme="blue"
                    icon={<ArrowUpIcon />}
                    onClick={sendMessage}
                    isDisabled={!input.trim() || isProcessing}
                    isLoading={isProcessing}
                    size="lg"
                    borderRadius="lg"
                  />
                  <input type="file" accept="image/*" multiple style={{ display: "none" }} ref={fileInputRef} onChange={handleFileChange} />
                </Flex>
              </VStack>
            </HStack>
          </Box>
        </Flex>

      </Flex>
    </ChakraProvider>
  );
};

export default ChatApp;
