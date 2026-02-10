/**
 * useApi Hook
 * 
 * Manages API calls with loading states, errors, and retries.
 */

import { useState, useCallback, useRef } from 'react';

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: any;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
}

interface UseApiReturn<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
  execute: (url: string, options?: RequestOptions) => Promise<T>;
  reset: () => void;
}

export function useApi<T = any>(baseUrl?: string): UseApiReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const execute = useCallback(
    async (url: string, options: RequestOptions = {}): Promise<T> => {
      const {
        timeout = 30000,
        retries = 0,
        retryDelay = 1000,
        body,
        ...fetchOptions
      } = options;

      // Cancel previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();
      const { signal } = abortControllerRef.current;

      setIsLoading(true);
      setError(null);

      const fullUrl = baseUrl ? `${baseUrl}${url}` : url;

      const performFetch = async (attemptsLeft: number): Promise<T> => {
        try {
          // Timeout wrapper
          const timeoutId = setTimeout(() => {
            abortControllerRef.current?.abort();
          }, timeout);

          const response = await fetch(fullUrl, {
            ...fetchOptions,
            body: body ? JSON.stringify(body) : undefined,
            headers: {
              'Content-Type': 'application/json',
              ...fetchOptions.headers,
            },
            signal,
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(
              `HTTP ${response.status}: ${errorText || response.statusText}`
            );
          }

          const result = await response.json();
          setData(result);
          return result;

        } catch (err) {
          if (attemptsLeft > 0 && !(err as Error).message?.includes('aborted')) {
            // Wait and retry
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            return performFetch(attemptsLeft - 1);
          }
          throw err;
        }
      };

      try {
        const result = await performFetch(retries);
        return result;
      } catch (err) {
        const error = err as Error;
        setError(error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [baseUrl]
  );

  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setData(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    data,
    error,
    isLoading,
    execute,
    reset,
  };
}

// Specialized hooks for common API patterns

export function useChat(serverUrl: string) {
  const api = useApi<{ choices: Array<{ message: { content: string } }> }>(serverUrl);

  const sendMessage = useCallback(
    async (
      messages: Array<{ role: string; content: string }>,
      options?: { temperature?: number; maxTokens?: number }
    ) => {
      const response = await api.execute('/v1/chat/completions', {
        method: 'POST',
        body: {
          messages,
          temperature: options?.temperature ?? 0.7,
          max_tokens: options?.maxTokens ?? 256,
        },
      });
      return response.choices?.[0]?.message?.content || '';
    },
    [api]
  );

  return {
    ...api,
    sendMessage,
  };
}

export function useCompletion(serverUrl: string) {
  const api = useApi<{ choices: Array<{ text: string }> }>(serverUrl);

  const complete = useCallback(
    async (prompt: string, options?: { temperature?: number; maxTokens?: number }) => {
      const response = await api.execute('/v1/completions', {
        method: 'POST',
        body: {
          prompt,
          temperature: options?.temperature ?? 0.7,
          max_tokens: options?.maxTokens ?? 256,
        },
      });
      return response.choices?.[0]?.text || '';
    },
    [api]
  );

  return {
    ...api,
    complete,
  };
}

export default useApi;
