/**
 * API client – Axios wrapper for backend communication.
 *
 * Centralises base URL configuration and error handling so that
 * individual components never construct HTTP requests directly.
 */

import axios from 'axios';

const api = axios.create({
  baseURL: '/',
  timeout: 30000,
  headers: {
    Accept: 'application/json',
  },
});

/**
 * Upload a CSV or SQLite file to the backend.
 * @param {File} file – The file object from a file input.
 * @returns {Promise<Object>} UploadResponse with session_id, columns, row_count, preview.
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Send a natural-language query against an uploaded dataset.
 * @param {string} sessionId – Session ID from the upload response.
 * @param {string} question  – The plain-English question.
 * @returns {Promise<Object>} QueryResponse with answer, chart_type, chart_data, etc.
 */
export async function queryData(sessionId, question) {
  const response = await api.post('/query', {
    session_id: sessionId,
    question,
  });
  return response.data;
}

export default api;
