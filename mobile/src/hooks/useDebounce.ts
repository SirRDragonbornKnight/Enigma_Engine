/**
 * useDebounce Hook
 * 
 * Debounces a value for search inputs, API calls, etc.
 */

import { useState, useEffect } from 'react';

export function useDebounce<T>(value: T, delay: number = 500): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * useDebounceCallback Hook
 * 
 * Debounces a callback function.
 */
export function useDebounceCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number = 500
): T {
  const [timeout, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

  return ((...args: Parameters<T>) => {
    if (timeout) {
      clearTimeout(timeout);
    }

    setTimeoutId(
      setTimeout(() => {
        callback(...args);
      }, delay)
    );
  }) as T;
}

export default useDebounce;
