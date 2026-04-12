/**
 * useQuery hook – manages the query lifecycle (loading, result, error).
 *
 * Encapsulates the API call, loading state, and error handling so
 * that components remain declarative and free of async logic.
 */

import { useState, useCallback } from 'react';
import { queryData } from '../api/client';

export default function useQuery() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const ask = useCallback(async (sessionId, question) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await queryData(sessionId, question);
      setResult(data);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Something went wrong. Please try again.';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setLoading(false);
  }, []);

  return { result, loading, error, ask, reset };
}
