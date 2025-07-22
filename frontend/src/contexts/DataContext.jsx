import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';
import * as duckdb from '@duckdb/duckdb-wasm';
import { v4 as uuidv4 } from 'uuid';
import { db as dexieDb } from '../db';
import { registerSchema } from '../services/dataApi';
import axios from 'axios'; // Added axios import

const DataContext = createContext(null);

export const useData = () => {
    const context = useContext(DataContext);
    if (!context) {
        throw new Error('useData must be used within a DataProvider');
    }
    return context;
};

// DuckDB singleton instance
let dbInstance = null;
let dbPromise = null;

async function getDB() {
    if (dbInstance) {
        return dbInstance;
    }
    if (!dbPromise) {
        dbPromise = (async () => {
            const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
            const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

            const worker_url = URL.createObjectURL(
                new Blob([`importScripts("${bundle.mainWorker}");`], {
                    type: 'text/javascript',
                })
            );

            const worker = new Worker(worker_url);
            const logger = new duckdb.ConsoleLogger();
            const db = new duckdb.AsyncDuckDB(logger, worker);
            await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
            URL.revokeObjectURL(worker_url);
            
            // Restore files from IndexedDB on startup
            console.log("Restoring data files from IndexedDB...");
            const savedFilesMeta = JSON.parse(localStorage.getItem('uploadedDataFiles') || '[]');
            if (savedFilesMeta.length > 0) {
                for (const fileMeta of savedFilesMeta) {
                    const fileData = await dexieDb.dataFiles.get(fileMeta.id);
                    if (fileData) {
                        try {
                            await db.registerFileBuffer(fileMeta.name, new Uint8Array(fileData.data));
                             console.log(`Restored and re-registered: ${fileMeta.name}`);
                        } catch (e) {
                            console.error(`Failed to re-register file ${fileMeta.name}:`, e);
                        }
                    }
                }
            }


            dbInstance = db;
            return db;
        })();
    }
    return dbPromise;
}


export const DataProvider = ({ children }) => {
    const [db, setDb] = useState(null);
    const [isInitialized, setIsInitialized] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [activeConnection, _setActiveConnection] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    
    useEffect(() => {
        const initDB = async () => {
            try {
                const db = await getDB();
                setDb(db);
                const loadedMeta = await loadTables(db);
                setUploadedFiles(loadedMeta);
                // Only set initialized if at least one table loaded successfully
                setIsInitialized(loadedMeta.length > 0);
                if (loadedMeta.length === 0) {
                    console.warn('No tables loaded - clearing potentially bad state');
                    localStorage.removeItem('uploadedDataFiles');
                    localStorage.removeItem('activeDataConnection');
                } else {
                    console.log(`Successfully loaded ${loadedMeta.length} table(s)`);
                }
            } catch (e) {
                console.error('Error initializing DuckDB:', e);
                setIsInitialized(false); // Mark as failed
            }
        };
        initDB();
    }, []);

    useEffect(() => {
        // Restore activeConnection from localStorage
        const savedConnection = localStorage.getItem('activeDataConnection');
        if (savedConnection) {
            try {
                _setActiveConnection(JSON.parse(savedConnection));
                console.log('Restored activeConnection from localStorage');
            } catch (e) {
                console.error('Failed to restore activeConnection:', e);
                localStorage.removeItem('activeDataConnection');
            }
        }
    }, []);
    
    const setActiveConnection = useCallback((connection) => {
        _setActiveConnection(connection);
        if (connection) {
            localStorage.setItem('activeDataConnection', JSON.stringify(connection));
            console.log('Saved activeConnection to localStorage');
        } else {
            localStorage.removeItem('activeDataConnection');
            console.log('Cleared activeConnection from localStorage');
        }
    }, []);

    const loadTables = async (db) => {
        const savedFilesMeta = JSON.parse(localStorage.getItem('uploadedDataFiles') || '[]');
        const loadedMeta = [];
        for (const fileMeta of savedFilesMeta) {
            let retries = 3;
            let success = false;
            while (retries > 0 && !success) {
                try {
                    console.log(`Loading table "${fileMeta.name}" (attempt ${4 - retries})`);
                    const root = await navigator.storage.getDirectory();
                    const parquetHandle = await root.getFileHandle(`${fileMeta.id}.parquet`, { create: false });
                    const file = await parquetHandle.getFile();
                    const buffer = await file.arrayBuffer();
                    await db.registerFileBuffer(`${fileMeta.id}.parquet`, new Uint8Array(buffer));
                    
                    const conn = await db.connect();
                    
                    // Drop if exists to avoid conflicts
                    await conn.query(`DROP TABLE IF EXISTS "${fileMeta.name}"`);
                    console.log(`Dropped existing table "${fileMeta.name}" if it existed`);
                    
                    // Create from Parquet
                    await conn.query(`CREATE TABLE "${fileMeta.name}" AS SELECT * FROM read_parquet('${fileMeta.id}.parquet')`);
                    console.log(`Created table "${fileMeta.name}" from Parquet`);
                    
                    await conn.close();
                    
                    loadedMeta.push(fileMeta);
                    success = true;
                } catch (e) {
                    console.error(`Failed to load table "${fileMeta.name}" (attempt ${4 - retries}):`, e);
                    retries--;
                    if (retries === 0) {
                        console.warn(`Giving up on "${fileMeta.name}" after 3 attempts`);
                    } else {
                        await new Promise(resolve => setTimeout(resolve, 500)); // Short delay before retry
                    }
                }
            }
        }
        console.log(`Loaded ${loadedMeta.length} table(s) from Parquet.`);
        return loadedMeta;
    };

    const registerFile = async (file) => {
        if (!db) return null;

        setIsLoading(true);
        try {
            const buffer = await file.arrayBuffer();
            await db.registerFileBuffer(file.name, new Uint8Array(buffer));

            const newFile = {
                id: uuidv4(),
                name: file.name,
                size: file.size,
                type: file.type,
                rowCount: 0,
                columnCount: 0,
            };

            const conn = await db.connect();
            await conn.query(`CREATE TABLE "${file.name}" AS SELECT * FROM read_csv_auto('${file.name}')`);

            const rowCountResult = await conn.query(`SELECT COUNT(*) FROM "${file.name}"`);
            const rowCountArray = rowCountResult.toArray();
            if (rowCountArray.length > 0) {
                const firstRow = rowCountArray[0].toJSON();
                const count = Object.values(firstRow)[0];
                newFile.rowCount = Number(count || 0);
            } else {
                newFile.rowCount = 0;
            }

            const tableInfo = await conn.query(`DESCRIBE "${file.name}"`);
            newFile.columnCount = tableInfo.numRows;

            const schemaArray = tableInfo.toArray();
            const columns = schemaArray.map(row => {
                const rowData = row.toJSON();
                return {
                    name: rowData.column_name,
                    type: rowData.column_type
                };
            });

            const sampleResult = await conn.query(`SELECT * FROM "${file.name}" LIMIT 5;`);
            const sampleRows = sampleResult.toArray().map(row => row.toJSON());

            const sessionId = localStorage.getItem('sessionId');
            if (sessionId) {
                try {
                    const schemaData = {
                        [file.name]: {
                            name: file.name,
                            rowCount: newFile.rowCount,
                            columns: columns,
                            sampleRows: sampleRows
                        }
                    };
                    await registerSchema(sessionId, schemaData);
                    console.log(`Schema registered with backend for ${file.name}`);
                } catch (error) {
                    console.error('Failed to register schema with backend:', error);
                }
            }

            // Export to Parquet
            const parquetName = `${newFile.id}.parquet`;
            await db.registerEmptyFileBuffer(parquetName);
            await conn.query(`COPY (SELECT * FROM "${file.name}") TO '${parquetName}' (FORMAT PARQUET)`);
            const parquetBuffer = await db.copyFileToBuffer(parquetName);

            // Save to OPFS
            const root = await navigator.storage.getDirectory();
            const parquetHandle = await root.getFileHandle(parquetName, { create: true });
            const writable = await parquetHandle.createWritable();
            await writable.write(parquetBuffer);
            await writable.close();

            await conn.close();

            setUploadedFiles(prev => {
                const updatedFiles = [...prev, newFile];
                localStorage.setItem('uploadedDataFiles', JSON.stringify(updatedFiles));
                console.log(`Updated localStorage with new file metadata.`);
                return updatedFiles;
            });

            return newFile;
        } catch (e) {
            console.error('Error registering file with DuckDB:', e);
            return null;
        } finally {
            setIsLoading(false);
        }
    };
    
    const query = async (sql) => {
        if (!db) {
            console.error("DB not initialized");
            return null;
        }
        console.log(`Executing SQL: ${sql}`);
        let c = null;
        try {
            c = await db.connect();
            const result = await c.query(sql);
            // New: Inspect result for errors or empty
            if (!result) {
                const errorDetails = `Query returned null result - possible error for SQL: ${sql}`;
                console.error(errorDetails);
                try {
                    await axios.post('/api/log-error', { error: 'Null result', sql, context: 'DuckDB query execution' });
                } catch (logError) { console.warn('Failed to log:', logError); }
                return null;
            }
            if (result.numRows === 0) {
                const warnDetails = `Query executed successfully but returned 0 rows - possible invalid query: ${sql}`;
                console.warn(warnDetails);
                try {
                    await axios.post('/api/log-error', { error: 'Empty result set', sql, context: 'DuckDB query execution (soft error)' });
                } catch (logError) { console.warn('Failed to log:', logError); }
            } else {
                console.log(`Query successful: ${result.numRows} rows returned for ${sql}`);
                // New: Check for 'fake error' results (e.g., LLM-generated error messages)
                if (result.numRows === 1) {
                    const firstRow = result.get(0).toJSON();
                    const keys = Object.keys(firstRow);
                    const values = Object.values(firstRow).join(' ').toLowerCase();
                    if (keys.some(k => k.toLowerCase().includes('error')) ||
                        values.includes('error') || values.includes('does not exist') || values.includes('not found')) {
                        const errorDetails = `Detected error in query result: ${JSON.stringify(firstRow)} - SQL: ${sql}`;
                        console.error(errorDetails);
                        try {
                            await axios.post('/api/log-error', {
                                error: 'Detected error in result data',
                                sql: sql,
                                context: 'DuckDB query execution (fake error)',
                                result: firstRow
                            });
                        } catch (logError) {
                            console.warn('Failed to log:', logError);
                        }
                        return null;  // Treat as failure
                    }
                }
            }
            return result;
        } catch(e) {
            const errorDetails = `Failed to execute SQL: ${sql} - Error: ${e.message} - Stack: ${e.stack}`;
            console.error(errorDetails);
            // Send to backend for terminal logging
            try {
                await axios.post('/api/log-error', {
                    error: e.message,
                    sql: sql,
                    context: 'DuckDB query execution'
                });
            } catch (logError) {
                console.warn('Failed to log error to backend:', logError);
            }
            return null;
        } finally {
            if (c) {
                await c.close();
            }
        }
    }

    const deleteDataset = async (id) => {
      const fileMeta = uploadedFiles.find(f => f.id === id);
      if (!fileMeta) return;

      try {
        const conn = await db.connect();
        await conn.query(`DROP TABLE IF EXISTS "${fileMeta.name}"`);
        await conn.close();

        const root = await navigator.storage.getDirectory();
        await root.removeEntry(`${id}.parquet`).catch(e => console.warn(`Failed to remove Parquet file: ${e}`));

        setUploadedFiles(prev => {
          const updated = prev.filter(f => f.id !== id);
          localStorage.setItem('uploadedDataFiles', JSON.stringify(updated));
          return updated;
        });
      } catch (e) {
        console.error('Error deleting dataset:', e);
      }
    };


    const value = {
        db,
        isInitialized,
        isLoading,
        uploadedFiles,
        setUploadedFiles,
        activeConnection,
        setActiveConnection,
        registerFile,
        query,
        deleteDataset,
    };

    // Temporary global exposure for debugging - REMOVE AFTER TESTING
    window.queryDuckDB = query;

    return (
        <DataContext.Provider value={value}>
            {children}
        </DataContext.Provider>
    );
}; 