/**
 * Model Management Screen
 * 
 * View, download, and manage AI models on mobile device.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import * as Haptics from 'expo-haptics';

// Types
interface ModelInfo {
  id: string;
  name: string;
  description: string;
  size: number; // bytes
  parameters: string;
  quantization?: string;
  isLocal: boolean;
  downloadProgress?: number;
  isDownloading?: boolean;
  path?: string;
  lastUsed?: number;
}

interface ModelConfig {
  activeModelId: string | null;
  downloadedModels: string[];
}

interface ModelScreenProps {
  darkMode?: boolean;
  serverUrl: string;
  onModelChange?: (modelId: string) => void;
}

const MODELS_DIR = `${FileSystem.documentDirectory}models/`;
const CONFIG_KEY = '@enigma_model_config';

// Available models
const AVAILABLE_MODELS: ModelInfo[] = [
  {
    id: 'enigma-nano',
    name: 'Enigma Nano',
    description: 'Ultra-lightweight model for basic tasks. Runs on any device.',
    size: 50 * 1024 * 1024, // 50MB
    parameters: '~1M',
    isLocal: false,
  },
  {
    id: 'enigma-micro',
    name: 'Enigma Micro',
    description: 'Small model optimized for mobile. Good for simple conversations.',
    size: 100 * 1024 * 1024, // 100MB
    parameters: '~2M',
    isLocal: false,
  },
  {
    id: 'enigma-tiny',
    name: 'Enigma Tiny',
    description: 'Compact model with better reasoning. Ideal for phones.',
    size: 200 * 1024 * 1024, // 200MB
    parameters: '~5M',
    isLocal: false,
  },
  {
    id: 'enigma-small',
    name: 'Enigma Small',
    description: 'Balanced model for general use. Great quality-size ratio.',
    size: 500 * 1024 * 1024, // 500MB
    parameters: '~27M',
    isLocal: false,
  },
  {
    id: 'enigma-small-q4',
    name: 'Enigma Small Q4',
    description: 'Quantized version of Small. 4-bit for smaller download.',
    size: 150 * 1024 * 1024,
    parameters: '~27M',
    quantization: 'Q4',
    isLocal: false,
  },
  {
    id: 'enigma-medium',
    name: 'Enigma Medium',
    description: 'Higher quality responses. Requires more storage.',
    size: 1 * 1024 * 1024 * 1024, // 1GB
    parameters: '~85M',
    isLocal: false,
  },
  {
    id: 'cloud-api',
    name: 'Cloud API',
    description: 'Use server-hosted model. Requires internet connection.',
    size: 0,
    parameters: 'Various',
    isLocal: false,
  },
];

const ModelScreen: React.FC<ModelScreenProps> = ({
  darkMode = false,
  serverUrl,
  onModelChange,
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [config, setConfig] = useState<ModelConfig>({
    activeModelId: null,
    downloadedModels: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState<string | null>(null);

  // Initialize
  useEffect(() => {
    initialize();
  }, []);

  const initialize = async () => {
    setIsLoading(true);
    try {
      // Ensure models directory exists
      const dirInfo = await FileSystem.getInfoAsync(MODELS_DIR);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(MODELS_DIR, { intermediates: true });
      }

      // Load config
      const storedConfig = await AsyncStorage.getItem(CONFIG_KEY);
      if (storedConfig) {
        setConfig(JSON.parse(storedConfig));
      }

      // Check which models are downloaded
      await refreshModels();
    } catch (error) {
      console.error('Initialize error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshModels = async () => {
    try {
      const files = await FileSystem.readDirectoryAsync(MODELS_DIR);
      const downloadedIds = files
        .filter(f => f.endsWith('.bin') || f.endsWith('.gguf'))
        .map(f => f.replace(/\.(bin|gguf)$/, ''));

      const updatedModels = AVAILABLE_MODELS.map(model => ({
        ...model,
        isLocal: downloadedIds.includes(model.id),
        path: downloadedIds.includes(model.id)
          ? `${MODELS_DIR}${model.id}.bin`
          : undefined,
      }));

      setModels(updatedModels);
      setConfig(prev => ({
        ...prev,
        downloadedModels: downloadedIds,
      }));
    } catch (error) {
      console.error('Refresh models error:', error);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refreshModels();
    setIsRefreshing(false);
  };

  const saveConfig = async (newConfig: ModelConfig) => {
    try {
      await AsyncStorage.setItem(CONFIG_KEY, JSON.stringify(newConfig));
      setConfig(newConfig);
    } catch (error) {
      console.error('Save config error:', error);
    }
  };

  const downloadModel = async (model: ModelInfo) => {
    if (model.id === 'cloud-api') {
      // Cloud API doesn't need download
      selectModel(model);
      return;
    }

    Alert.alert(
      'Download Model',
      `Download ${model.name}? (${formatBytes(model.size)})`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Download',
          onPress: async () => {
            setDownloadingModel(model.id);
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

            try {
              const downloadUrl = `${serverUrl}/models/${model.id}.bin`;
              const localPath = `${MODELS_DIR}${model.id}.bin`;

              const downloadResumable = FileSystem.createDownloadResumable(
                downloadUrl,
                localPath,
                {},
                (progress) => {
                  const percent = progress.totalBytesWritten / progress.totalBytesExpectedToWrite;
                  setModels(prev =>
                    prev.map(m =>
                      m.id === model.id
                        ? { ...m, downloadProgress: percent, isDownloading: true }
                        : m
                    )
                  );
                }
              );

              const result = await downloadResumable.downloadAsync();
              
              if (result?.uri) {
                await refreshModels();
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                Alert.alert('Success', `${model.name} downloaded successfully`);
              }
            } catch (error: any) {
              console.error('Download error:', error);
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
              Alert.alert('Download Failed', error.message || 'Could not download model');
            } finally {
              setDownloadingModel(null);
              setModels(prev =>
                prev.map(m =>
                  m.id === model.id
                    ? { ...m, downloadProgress: undefined, isDownloading: false }
                    : m
                )
              );
            }
          },
        },
      ]
    );
  };

  const deleteModel = async (model: ModelInfo) => {
    Alert.alert(
      'Delete Model',
      `Delete ${model.name}? This will free ${formatBytes(model.size)}.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const localPath = `${MODELS_DIR}${model.id}.bin`;
              await FileSystem.deleteAsync(localPath, { idempotent: true });
              
              // If deleted model was active, clear selection
              if (config.activeModelId === model.id) {
                await saveConfig({
                  ...config,
                  activeModelId: null,
                });
              }

              await refreshModels();
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            } catch (error) {
              console.error('Delete error:', error);
              Alert.alert('Error', 'Failed to delete model');
            }
          },
        },
      ]
    );
  };

  const selectModel = async (model: ModelInfo) => {
    const newConfig = {
      ...config,
      activeModelId: model.id,
    };
    await saveConfig(newConfig);
    onModelChange?.(model.id);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return 'Cloud';
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    let size = bytes;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const getStorageUsed = (): number => {
    return models
      .filter(m => m.isLocal)
      .reduce((sum, m) => sum + m.size, 0);
  };

  const styles = createStyles(darkMode);

  if (isLoading) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator size="large" color={darkMode ? '#fff' : '#007AFF'} />
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={isRefreshing} onRefresh={handleRefresh} />
      }
    >
      {/* Header */}
      <View style={styles.header}>
        <Text style={[styles.title, darkMode && styles.darkText]}>
          AI Models
        </Text>
        <View style={[styles.storageBadge, darkMode && styles.darkCard]}>
          <Text style={[styles.storageText, darkMode && styles.darkSubtext]}>
            {formatBytes(getStorageUsed())} used
          </Text>
        </View>
      </View>

      {/* Active Model */}
      {config.activeModelId && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
            ACTIVE MODEL
          </Text>
          <View style={[styles.activeModelCard, darkMode && styles.darkCard]}>
            <Text style={[styles.activeModelName, darkMode && styles.darkText]}>
              {models.find(m => m.id === config.activeModelId)?.name || 'Unknown'}
            </Text>
            <View style={styles.activeBadge}>
              <Text style={styles.activeBadgeText}>Active</Text>
            </View>
          </View>
        </View>
      )}

      {/* Available Models */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          AVAILABLE MODELS
        </Text>

        {models.map(model => (
          <View
            key={model.id}
            style={[
              styles.modelCard,
              darkMode && styles.darkCard,
              config.activeModelId === model.id && styles.modelCardActive,
            ]}
          >
            <View style={styles.modelHeader}>
              <View style={styles.modelInfo}>
                <Text style={[styles.modelName, darkMode && styles.darkText]}>
                  {model.name}
                </Text>
                <Text style={[styles.modelParams, darkMode && styles.darkSubtext]}>
                  {model.parameters} params
                  {model.quantization && ` (${model.quantization})`}
                </Text>
              </View>
              <Text style={[styles.modelSize, darkMode && styles.darkSubtext]}>
                {formatBytes(model.size)}
              </Text>
            </View>

            <Text style={[styles.modelDescription, darkMode && styles.darkSubtext]}>
              {model.description}
            </Text>

            {/* Download Progress */}
            {model.isDownloading && model.downloadProgress !== undefined && (
              <View style={styles.progressContainer}>
                <View
                  style={[
                    styles.progressBar,
                    { width: `${model.downloadProgress * 100}%` },
                  ]}
                />
                <Text style={styles.progressText}>
                  {Math.round(model.downloadProgress * 100)}%
                </Text>
              </View>
            )}

            {/* Actions */}
            <View style={styles.modelActions}>
              {model.isLocal || model.id === 'cloud-api' ? (
                <>
                  <TouchableOpacity
                    style={[
                      styles.selectButton,
                      config.activeModelId === model.id && styles.selectButtonActive,
                    ]}
                    onPress={() => selectModel(model)}
                  >
                    <Text
                      style={[
                        styles.selectButtonText,
                        config.activeModelId === model.id &&
                          styles.selectButtonTextActive,
                      ]}
                    >
                      {config.activeModelId === model.id ? 'Selected' : 'Select'}
                    </Text>
                  </TouchableOpacity>

                  {model.isLocal && model.id !== 'cloud-api' && (
                    <TouchableOpacity
                      style={styles.deleteButton}
                      onPress={() => deleteModel(model)}
                    >
                      <Text style={styles.deleteButtonText}>Delete</Text>
                    </TouchableOpacity>
                  )}
                </>
              ) : (
                <TouchableOpacity
                  style={[
                    styles.downloadButton,
                    downloadingModel === model.id && styles.downloadButtonDisabled,
                  ]}
                  onPress={() => downloadModel(model)}
                  disabled={downloadingModel === model.id}
                >
                  {downloadingModel === model.id ? (
                    <ActivityIndicator color="#fff" size="small" />
                  ) : (
                    <Text style={styles.downloadButtonText}>Download</Text>
                  )}
                </TouchableOpacity>
              )}
            </View>

            {/* Local badge */}
            {model.isLocal && (
              <View style={styles.localBadge}>
                <Text style={styles.localBadgeText}>On Device</Text>
              </View>
            )}
          </View>
        ))}
      </View>

      {/* Info Section */}
      <View style={styles.infoSection}>
        <Text style={[styles.infoTitle, darkMode && styles.darkText]}>
          About Local Models
        </Text>
        <Text style={[styles.infoText, darkMode && styles.darkSubtext]}>
          Local models run directly on your device for privacy and offline use.
          Larger models provide better quality but require more storage and
          processing power.
        </Text>
        <Text style={[styles.infoText, darkMode && styles.darkSubtext]}>
          Cloud API uses server-hosted models for the best quality but requires
          an internet connection.
        </Text>
      </View>

      {/* Spacer */}
      <View style={{ height: 40 }} />
    </ScrollView>
  );
};

const createStyles = (darkMode: boolean) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: darkMode ? '#1a1a1a' : '#f5f5f5',
    },
    center: {
      justifyContent: 'center',
      alignItems: 'center',
    },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 16,
    },
    title: {
      fontSize: 28,
      fontWeight: 'bold',
    },
    darkText: {
      color: '#fff',
    },
    darkSubtext: {
      color: '#aaa',
    },
    storageBadge: {
      backgroundColor: '#fff',
      paddingHorizontal: 12,
      paddingVertical: 6,
      borderRadius: 16,
    },
    darkCard: {
      backgroundColor: '#2a2a2a',
    },
    storageText: {
      fontSize: 12,
      color: '#666',
    },
    section: {
      marginBottom: 24,
      paddingHorizontal: 16,
    },
    sectionTitle: {
      fontSize: 12,
      fontWeight: '600',
      color: '#888',
      marginBottom: 12,
    },
    activeModelCard: {
      backgroundColor: '#fff',
      borderRadius: 12,
      padding: 16,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    activeModelName: {
      fontSize: 18,
      fontWeight: '600',
    },
    activeBadge: {
      backgroundColor: '#34C759',
      paddingHorizontal: 10,
      paddingVertical: 4,
      borderRadius: 12,
    },
    activeBadgeText: {
      color: '#fff',
      fontSize: 12,
      fontWeight: '600',
    },
    modelCard: {
      backgroundColor: '#fff',
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
      position: 'relative',
    },
    modelCardActive: {
      borderWidth: 2,
      borderColor: '#007AFF',
    },
    modelHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: 8,
    },
    modelInfo: {
      flex: 1,
    },
    modelName: {
      fontSize: 18,
      fontWeight: '600',
    },
    modelParams: {
      fontSize: 12,
      color: '#888',
      marginTop: 2,
    },
    modelSize: {
      fontSize: 14,
      fontWeight: '600',
      color: '#007AFF',
    },
    modelDescription: {
      fontSize: 14,
      color: '#666',
      lineHeight: 20,
      marginBottom: 12,
    },
    progressContainer: {
      height: 24,
      backgroundColor: '#e0e0e0',
      borderRadius: 12,
      marginBottom: 12,
      overflow: 'hidden',
      justifyContent: 'center',
    },
    progressBar: {
      position: 'absolute',
      left: 0,
      top: 0,
      bottom: 0,
      backgroundColor: '#007AFF',
      borderRadius: 12,
    },
    progressText: {
      textAlign: 'center',
      fontSize: 12,
      fontWeight: '600',
      color: '#333',
    },
    modelActions: {
      flexDirection: 'row',
      gap: 8,
    },
    selectButton: {
      flex: 1,
      backgroundColor: '#e0e0e0',
      paddingVertical: 10,
      borderRadius: 8,
      alignItems: 'center',
    },
    selectButtonActive: {
      backgroundColor: '#007AFF',
    },
    selectButtonText: {
      fontWeight: '600',
      color: '#333',
    },
    selectButtonTextActive: {
      color: '#fff',
    },
    deleteButton: {
      paddingHorizontal: 16,
      paddingVertical: 10,
      borderRadius: 8,
      borderWidth: 1,
      borderColor: '#FF3B30',
    },
    deleteButtonText: {
      color: '#FF3B30',
      fontWeight: '600',
    },
    downloadButton: {
      flex: 1,
      backgroundColor: '#007AFF',
      paddingVertical: 10,
      borderRadius: 8,
      alignItems: 'center',
    },
    downloadButtonDisabled: {
      opacity: 0.6,
    },
    downloadButtonText: {
      color: '#fff',
      fontWeight: '600',
    },
    localBadge: {
      position: 'absolute',
      top: 8,
      right: 8,
      backgroundColor: '#34C759',
      paddingHorizontal: 8,
      paddingVertical: 2,
      borderRadius: 8,
    },
    localBadgeText: {
      color: '#fff',
      fontSize: 10,
      fontWeight: '600',
    },
    infoSection: {
      marginHorizontal: 16,
      padding: 16,
      backgroundColor: darkMode ? '#2a2a2a' : '#fff',
      borderRadius: 12,
    },
    infoTitle: {
      fontSize: 16,
      fontWeight: '600',
      marginBottom: 8,
    },
    infoText: {
      fontSize: 14,
      color: '#666',
      lineHeight: 20,
      marginBottom: 8,
    },
  });

export default ModelScreen;
