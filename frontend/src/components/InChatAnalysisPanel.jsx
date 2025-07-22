import React from 'react';
import {
    Box,
    Text,
    Collapse,
    useDisclosure,
    Button,
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
    TableContainer,
    Code,
    VStack,
} from '@chakra-ui/react';
import { Chart as ChartJS, registerables } from 'chart.js';
import { Chart } from 'react-chartjs-2';

ChartJS.register(...registerables);

const InChatAnalysisPanel = ({ explanation, sql, data, chartConfig, error }) => {
    const { isOpen, onToggle } = useDisclosure();

    const tableHeaders = data && data.length > 0 ? Object.keys(data[0]) : [];

    return (
        <Box
            borderWidth="1px"
            borderRadius="lg"
            p={4}
            mt={2}
            bg="gray.50"
            _dark={{ bg: 'gray.700' }}
        >
            <VStack align="stretch" spacing={4}>
                <Text>{explanation}</Text>

                <Button onClick={onToggle} size="sm">
                    View Generated SQL
                </Button>
                <Collapse in={isOpen} animateOpacity>
                    <Code display="block" p={2} borderRadius="md" whiteSpace="pre-wrap">
                        {sql}
                    </Code>
                </Collapse>

                {error && (
                    <Box bg="red.50" _dark={{ bg: 'red.900' }} p={3} borderRadius="md" borderLeft="4px" borderLeftColor="red.500">
                        <Text color="red.600" _dark={{ color: 'red.200' }} fontWeight="medium">
                            Error executing query:
                        </Text>
                        <Text color="red.600" _dark={{ color: 'red.200' }} fontSize="sm">
                            {error}
                        </Text>
                    </Box>
                )}

                {chartConfig && !error && (
                    <Box>
                        <Text fontWeight="bold" mb={2}>Visualization</Text>
                        <Chart type={chartConfig.type} data={chartConfig.data} options={chartConfig.options} />
                    </Box>
                )}
                
                {data && data.length > 0 && !error && (
                <Box>
                     <Text fontWeight="bold" mb={2}>Results Preview</Text>
                    <TableContainer>
                        <Table variant="simple" size="sm">
                            <Thead>
                                <Tr>
                                    {tableHeaders.map((header) => (
                                        <Th key={header}>{header}</Th>
                                    ))}
                                </Tr>
                            </Thead>
                            <Tbody>
                                {data.slice(0, 5).map((row, rowIndex) => (
                                    <Tr key={rowIndex}>
                                        {tableHeaders.map((header) => (
                                            <Td key={header}>{String(row[header])}</Td>
                                        ))}
                                    </Tr>
                                ))}
                            </Tbody>
                        </Table>
                    </TableContainer>
                        {data.length > 5 && (
                            <Text fontSize="sm" color="gray.600" _dark={{ color: 'gray.400' }} mt={2}>
                                Showing first 5 of {data.length} rows
                            </Text>
                        )}
                    </Box>
                )}
                
                {!error && (!data || data.length === 0) && (
                    <Box bg="yellow.50" _dark={{ bg: 'yellow.900' }} p={3} borderRadius="md">
                        <Text color="yellow.700" _dark={{ color: 'yellow.200' }} fontSize="sm">
                            Query executed successfully but returned no results.
                        </Text>
                </Box>
                )}
            </VStack>
        </Box>
    );
};

export default InChatAnalysisPanel; 