import { useState, useCallback } from 'react';
import { useData } from '../contexts/DataContext';
import { generateSql, generateVisualizations } from '../services/dataApi';

export const useDataChat = () => {
    const { query } = useData();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const analyzeData = useCallback(async (prompt, connection) => {
        if (!connection) {
            setError("No active data connection.");
            return null;
        }

        setIsLoading(true);
        setError(null);

        try {
            let schema, sql, explanation, data, chart_config;

            if (connection.type === 'file') {
                // Use DESCRIBE to get schema for files, which is the correct DuckDB approach
                const schemaResult = await query(`DESCRIBE "${connection.file.name}"`);
                if (!schemaResult) throw new Error('Failed to retrieve table schema.');
                
                const fields = schemaResult.toArray().map(row => {
                    const rowJson = row.toJSON();
                    return { name: rowJson.column_name, type: rowJson.column_type };
                });
                schema = { [connection.file.name]: fields };
                
                const sqlResponse = await generateSql(prompt, schema);
                if (!sqlResponse || !sqlResponse.sql) throw new Error('Failed to generate SQL query.');
                sql = sqlResponse.sql;
                explanation = sqlResponse.explanation;

                const dataResult = await query(sql);
                if (!dataResult) throw new Error('Failed to execute the generated SQL query.');
                data = dataResult.toArray().map(row => row.toJSON());

            } else if (connection.type === 'database') {
                // This logic remains for database connections
                const dbSchema = await getDbSchema(connection.connection_type, connection.params);
                const relevantSchema = Object.keys(dbSchema).reduce((acc, schemaName) => {
                    const relevantTables = dbSchema[schemaName].filter(table => connection.tables.includes(table));
                    if (relevantTables.length > 0) {
                        acc[schemaName] = relevantTables;
                    }
                    return acc;
                }, {});

                ({ sql, explanation } = await generateSql(prompt, relevantSchema));
                if (!sql) throw new Error('Failed to generate SQL query.');
                
                data = await executeRemoteSql(connection.connection_type, connection.params, sql);

            } else {
                throw new Error('Unsupported connection type for data analysis.');
            }

            const vizResponse = await generateVisualizations(prompt, sql, data);
            chart_config = vizResponse ? vizResponse.chart_config : null;

            return { explanation, sql, data, chartConfig: chart_config };

        } catch (err) {
            console.error('Data analysis failed:', err);
            setError(err.message || 'An unexpected error occurred during analysis.');
            return null;
        } finally {
            setIsLoading(false);
        }
    }, [query]);

    return { analyzeData, isLoading, error };
}; 