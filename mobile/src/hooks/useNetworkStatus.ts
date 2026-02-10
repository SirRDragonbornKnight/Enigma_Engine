/**
 * useNetworkStatus Hook
 * 
 * Monitors network connectivity for offline mode support.
 */

import { useState, useEffect, useCallback } from 'react';
import NetInfo, { NetInfoState, NetInfoStateType } from '@react-native-community/netinfo';

interface NetworkStatus {
  isConnected: boolean;
  isInternetReachable: boolean | null;
  type: NetInfoStateType;
  details: any;
}

interface UseNetworkStatusReturn extends NetworkStatus {
  refresh: () => Promise<void>;
}

export function useNetworkStatus(): UseNetworkStatusReturn {
  const [status, setStatus] = useState<NetworkStatus>({
    isConnected: true,
    isInternetReachable: null,
    type: NetInfoStateType.unknown,
    details: null,
  });

  useEffect(() => {
    // Initial fetch
    fetchNetworkStatus();

    // Subscribe to changes
    const unsubscribe = NetInfo.addEventListener((state) => {
      updateStatus(state);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  const updateStatus = (state: NetInfoState) => {
    setStatus({
      isConnected: state.isConnected ?? false,
      isInternetReachable: state.isInternetReachable,
      type: state.type,
      details: state.details,
    });
  };

  const fetchNetworkStatus = async () => {
    try {
      const state = await NetInfo.fetch();
      updateStatus(state);
    } catch (error) {
      console.error('Failed to fetch network status:', error);
    }
  };

  const refresh = useCallback(async () => {
    await fetchNetworkStatus();
  }, []);

  return {
    ...status,
    refresh,
  };
}

export default useNetworkStatus;
