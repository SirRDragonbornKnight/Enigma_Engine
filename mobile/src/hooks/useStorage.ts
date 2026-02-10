/**
 * useStorage Hook
 * 
 * Persistent storage with automatic JSON serialization.
 */

import { useState, useEffect, useCallback } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface UseStorageOptions<T> {
  defaultValue: T;
}

interface UseStorageReturn<T> {
  value: T;
  setValue: (newValue: T | ((prev: T) => T)) => Promise<void>;
  isLoading: boolean;
  error: Error | null;
  remove: () => Promise<void>;
}

export function useStorage<T>(
  key: string,
  options: UseStorageOptions<T>
): UseStorageReturn<T> {
  const [value, setValueState] = useState<T>(options.defaultValue);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Load value on mount
  useEffect(() => {
    loadValue();
  }, [key]);

  const loadValue = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const stored = await AsyncStorage.getItem(key);
      if (stored !== null) {
        setValueState(JSON.parse(stored));
      }
    } catch (e) {
      setError(e as Error);
      console.error(`Failed to load ${key}:`, e);
    } finally {
      setIsLoading(false);
    }
  };

  const setValue = useCallback(
    async (newValue: T | ((prev: T) => T)) => {
      try {
        const valueToStore =
          typeof newValue === 'function'
            ? (newValue as (prev: T) => T)(value)
            : newValue;
        
        setValueState(valueToStore);
        await AsyncStorage.setItem(key, JSON.stringify(valueToStore));
      } catch (e) {
        setError(e as Error);
        console.error(`Failed to save ${key}:`, e);
        throw e;
      }
    },
    [key, value]
  );

  const remove = useCallback(async () => {
    try {
      await AsyncStorage.removeItem(key);
      setValueState(options.defaultValue);
    } catch (e) {
      setError(e as Error);
      throw e;
    }
  }, [key, options.defaultValue]);

  return {
    value,
    setValue,
    isLoading,
    error,
    remove,
  };
}

export default useStorage;
