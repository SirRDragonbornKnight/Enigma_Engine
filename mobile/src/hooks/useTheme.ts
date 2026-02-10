/**
 * useTheme Hook
 * 
 * Manages app theme (light/dark mode) with system detection.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

type ThemeMode = 'light' | 'dark' | 'system';

interface Theme {
  isDark: boolean;
  colors: {
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    primary: string;
    border: string;
    error: string;
    success: string;
    warning: string;
  };
}

interface UseThemeReturn {
  theme: Theme;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  toggleDark: () => void;
}

const THEME_STORAGE_KEY = '@enigma_theme_mode';

const lightTheme: Theme = {
  isDark: false,
  colors: {
    background: '#f5f5f5',
    surface: '#ffffff',
    text: '#000000',
    textSecondary: '#666666',
    primary: '#007AFF',
    border: '#e0e0e0',
    error: '#FF3B30',
    success: '#34C759',
    warning: '#FF9500',
  },
};

const darkTheme: Theme = {
  isDark: true,
  colors: {
    background: '#1a1a1a',
    surface: '#2a2a2a',
    text: '#ffffff',
    textSecondary: '#aaaaaa',
    primary: '#0A84FF',
    border: '#3a3a3a',
    error: '#FF453A',
    success: '#30D158',
    warning: '#FF9F0A',
  },
};

export function useTheme(): UseThemeReturn {
  const systemColorScheme = useColorScheme();
  const [mode, setModeState] = useState<ThemeMode>('system');
  const [isLoading, setIsLoading] = useState(true);

  // Load saved preference
  useEffect(() => {
    loadThemePreference();
  }, []);

  const loadThemePreference = async () => {
    try {
      const stored = await AsyncStorage.getItem(THEME_STORAGE_KEY);
      if (stored && ['light', 'dark', 'system'].includes(stored)) {
        setModeState(stored as ThemeMode);
      }
    } catch (error) {
      console.error('Failed to load theme preference:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const setMode = useCallback(async (newMode: ThemeMode) => {
    setModeState(newMode);
    try {
      await AsyncStorage.setItem(THEME_STORAGE_KEY, newMode);
    } catch (error) {
      console.error('Failed to save theme preference:', error);
    }
  }, []);

  const toggleDark = useCallback(() => {
    if (mode === 'system') {
      // Override system with opposite
      setMode(systemColorScheme === 'dark' ? 'light' : 'dark');
    } else {
      setMode(mode === 'dark' ? 'light' : 'dark');
    }
  }, [mode, systemColorScheme, setMode]);

  const theme = useMemo(() => {
    const isDark = 
      mode === 'dark' || 
      (mode === 'system' && systemColorScheme === 'dark');
    
    return isDark ? darkTheme : lightTheme;
  }, [mode, systemColorScheme]);

  return {
    theme,
    mode,
    setMode,
    toggleDark,
  };
}

export default useTheme;
