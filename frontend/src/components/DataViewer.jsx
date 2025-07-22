import React, { useEffect, useState, useMemo } from 'react';
import {
    useReactTable,
    getCoreRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    flexRender,
} from '@tanstack/react-table';
import {
    Table,
    Thead,
    Tbody,
    Tr,
    Th,
    Td,
    IconButton,
    Text,
    Flex,
    Select,
    Box,
    Icon,
} from '@chakra-ui/react';
import { FaSort, FaSortUp, FaSortDown } from 'react-icons/fa';
import { useData } from '../contexts/DataContext';

const DataViewer = ({ selectedFile }) => {
    const { query } = useData();
    const [data, setData] = useState([]);
    const [columns, setColumns] = useState([]);
    const [sorting, setSorting] = useState([]);

    useEffect(() => {
        if (selectedFile) {
            const fetchData = async () => {
                const result = await query(`SELECT * FROM "${selectedFile.name}" LIMIT 100`);
                if (result) {
                    const dataArray = result.toArray().map(row => row.toJSON());
                    if (dataArray.length > 0) {
                        const columnDefs = Object.keys(dataArray[0]).map(key => ({
                            accessorKey: key,
                            header: key,
                        }));
                        setColumns(columnDefs);
                        setData(dataArray);
                    }
                }
            };
            fetchData();
        }
    }, [selectedFile, query]);

    const table = useReactTable({
        columns,
        data,
        state: {
            sorting,
        },
        onSortingChange: setSorting,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        initialState: {
            pagination: {
                pageSize: 10,
            }
        }
    });

    if (!selectedFile) {
        return (
            <Box p={5} borderWidth="1px" borderRadius="md" textAlign="center">
                <Text>Select a file to view its data.</Text>
            </Box>
        );
    }
    
    return (
        <Box>
            <Box overflowX="auto">
                <Table variant="simple">
                    <Thead>
                        {table.getHeaderGroups().map(headerGroup => (
                            <Tr key={headerGroup.id}>
                                {headerGroup.headers.map(header => (
                                    <Th key={header.id} onClick={header.column.getToggleSortingHandler()}>
                                        <Flex>
                                            {flexRender(header.column.columnDef.header, header.getContext())}
                                            <Box ml={2}>
                                                {header.column.getIsSorted() === 'desc' ? <FaSortDown /> : header.column.getIsSorted() === 'asc' ? <FaSortUp /> : <FaSort />}
                                            </Box>
                                        </Flex>
                                    </Th>
                                ))}
                            </Tr>
                        ))}
                    </Thead>
                    <Tbody>
                        {table.getRowModel().rows.map(row => (
                            <Tr key={row.id}>
                                {row.getVisibleCells().map(cell => (
                                    <Td key={cell.id}>
                                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                    </Td>
                                ))}
                            </Tr>
                        ))}
                    </Tbody>
                </Table>
            </Box>
            <Flex justify="space-between" align="center" mt={4}>
                <Flex>
                    <Text>
                        Page{' '}
                        <strong>
                            {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
                        </strong>
                    </Text>
                </Flex>
                <Flex>
                     <IconButton
                        onClick={() => table.previousPage()}
                        isDisabled={!table.getCanPreviousPage()}
                        aria-label="Previous page"
                        icon={<Icon as={FaSortUp} transform="rotate(-90deg)" />}
                        mr={2}
                    />
                    <IconButton
                        onClick={() => table.nextPage()}
                        isDisabled={!table.getCanNextPage()}
                        aria-label="Next page"
                        icon={<Icon as={FaSortDown} transform="rotate(-90deg)" />}
                    />
                </Flex>
                 <Select
                    width="auto"
                    value={table.getState().pagination.pageSize}
                    onChange={e => {
                        table.setPageSize(Number(e.target.value))
                    }}
                >
                    {[10, 20, 30, 40, 50].map(pageSize => (
                        <option key={pageSize} value={pageSize}>
                            Show {pageSize}
                        </option>
                    ))}
                </Select>
            </Flex>
        </Box>
    );
};

export default DataViewer; 