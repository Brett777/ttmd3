import axios from 'axios';

const API_URL = '/api/data';

export const registerSchema = async (sessionId, schemaData) => {
    try {
        const response = await axios.post(`${API_URL}/register-schema`, {
            session_id: sessionId,
            schema_data: schemaData,
        });
        return response.data;
    } catch (error) {
        console.error('Error registering schema:', error);
        throw error;
    }
};

export const generateSql = async (prompt, schema) => {
    try {
        const response = await axios.post(`${API_URL}/generate-sql`, {
            prompt,
            schema,
        });
        return response.data;
    } catch (error) {
        console.error('Error generating SQL:', error);
        throw error;
    }
};

export const generateVisualizations = async (prompt, sql, dataSample) => {
    try {
        const response = await axios.post(`${API_URL}/generate-visualizations`, {
            prompt,
            sql,
            data_sample: dataSample,
        });
        return response.data;
    } catch (error) {
        console.error('Error generating visualization:', error);
        throw error;
    }
};

export const testDBConnection = async (connectionType, params) => {
    try {
        const response = await axios.post(`${API_URL}/test-connection`, {
            connection_type: connectionType,
            params: params,
        });
        return response.data;
    } catch (error) {
        console.error('Error testing DB connection:', error);
        throw error.response.data;
    }
}

export const getDbSchema = async (connectionType, params) => {
    try {
        const response = await axios.post(`${API_URL}/db-schema`, {
            connection_type: connectionType,
            params: params,
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching DB schema:', error);
        throw error;
    }
}

export const executeRemoteSql = async (connectionType, params, sql) => {
    try {
        const response = await axios.post(`${API_URL}/execute-remote-sql`, {
            connection_type: connectionType,
            params: params,
            sql: sql,
        });
        return response.data;
    } catch (error) {
        console.error('Error executing remote SQL:', error);
        throw error;
    }
} 