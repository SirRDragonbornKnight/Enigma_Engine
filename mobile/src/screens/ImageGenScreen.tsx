/**
 * Image Generation Screen
 * 
 * Generate AI images on mobile with various styles and options.
 */

import React, { useState, useCallback, useRef } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Image,
  Alert,
  ActivityIndicator,
  Share,
  Dimensions,
  Platform,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as MediaLibrary from 'expo-media-library';
import * as FileSystem from 'expo-file-system';
import * as Haptics from 'expo-haptics';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Types
interface GeneratedImage {
  id: string;
  prompt: string;
  imageUri: string;
  timestamp: number;
  style: string;
  size: string;
}

interface ImageStyle {
  id: string;
  name: string;
  description: string;
  modifier: string;
}

interface ImageScreenProps {
  darkMode?: boolean;
  serverUrl: string;
  hapticFeedback?: boolean;
}

const IMAGE_STYLES: ImageStyle[] = [
  { id: 'none', name: 'None', description: 'No style modifier', modifier: '' },
  { id: 'photo', name: 'Photorealistic', description: 'High-quality photo', modifier: 'photorealistic, 8k, detailed' },
  { id: 'anime', name: 'Anime', description: 'Japanese animation style', modifier: 'anime style, vibrant colors' },
  { id: 'oil', name: 'Oil Painting', description: 'Classic oil painting', modifier: 'oil painting, textured brushstrokes' },
  { id: 'watercolor', name: 'Watercolor', description: 'Soft watercolor art', modifier: 'watercolor painting, soft edges' },
  { id: 'digital', name: 'Digital Art', description: 'Modern digital illustration', modifier: 'digital art, trending on artstation' },
  { id: '3d', name: '3D Render', description: 'Realistic 3D rendering', modifier: '3d render, octane render, volumetric lighting' },
  { id: 'pixel', name: 'Pixel Art', description: 'Retro pixel graphics', modifier: 'pixel art, 8-bit, retro game style' },
  { id: 'sketch', name: 'Sketch', description: 'Pencil sketch drawing', modifier: 'pencil sketch, detailed linework' },
];

const IMAGE_SIZES = [
  { id: 'small', name: 'Small', width: 256, height: 256 },
  { id: 'medium', name: 'Medium', width: 512, height: 512 },
  { id: 'large', name: 'Large', width: 768, height: 768 },
  { id: 'wide', name: 'Wide', width: 768, height: 512 },
  { id: 'tall', name: 'Tall', width: 512, height: 768 },
];

const HISTORY_KEY = '@enigma_image_history';

const ImageGenScreen: React.FC<ImageScreenProps> = ({
  darkMode = false,
  serverUrl,
  hapticFeedback = true,
}) => {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [selectedStyle, setSelectedStyle] = useState<string>('none');
  const [selectedSize, setSelectedSize] = useState<string>('medium');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImage, setGeneratedImage] = useState<GeneratedImage | null>(null);
  const [history, setHistory] = useState<GeneratedImage[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Advanced settings
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [seed, setSeed] = useState<number | null>(null);

  const scrollRef = useRef<ScrollView>(null);

  // Load history on mount
  React.useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const stored = await AsyncStorage.getItem(HISTORY_KEY);
      if (stored) {
        setHistory(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const saveToHistory = async (image: GeneratedImage) => {
    try {
      const newHistory = [image, ...history].slice(0, 50); // Keep last 50
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(newHistory));
      setHistory(newHistory);
    } catch (error) {
      console.error('Failed to save to history:', error);
    }
  };

  const generateImage = async () => {
    if (!prompt.trim()) {
      Alert.alert('Error', 'Please enter a prompt');
      return;
    }

    if (hapticFeedback) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    }

    setIsGenerating(true);
    setGeneratedImage(null);

    try {
      // Build full prompt with style
      const style = IMAGE_STYLES.find(s => s.id === selectedStyle);
      const size = IMAGE_SIZES.find(s => s.id === selectedSize);
      
      let fullPrompt = prompt;
      if (style && style.modifier) {
        fullPrompt = `${prompt}, ${style.modifier}`;
      }

      const response = await fetch(`${serverUrl}/v1/images/generations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: fullPrompt,
          negative_prompt: negativePrompt || undefined,
          width: size?.width || 512,
          height: size?.height || 512,
          num_inference_steps: steps,
          guidance_scale: guidanceScale,
          seed: seed ?? undefined,
        }),
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.status}`);
      }

      const data = await response.json();
      
      // Handle different response formats
      let imageUri: string;
      if (data.data?.[0]?.url) {
        imageUri = data.data[0].url;
      } else if (data.data?.[0]?.b64_json) {
        imageUri = `data:image/png;base64,${data.data[0].b64_json}`;
      } else if (data.image_url) {
        imageUri = data.image_url;
      } else if (data.image_base64) {
        imageUri = `data:image/png;base64,${data.image_base64}`;
      } else {
        throw new Error('No image in response');
      }

      const newImage: GeneratedImage = {
        id: Date.now().toString(),
        prompt,
        imageUri,
        timestamp: Date.now(),
        style: selectedStyle,
        size: selectedSize,
      };

      setGeneratedImage(newImage);
      await saveToHistory(newImage);

      if (hapticFeedback) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }

    } catch (error: any) {
      console.error('Image generation error:', error);
      Alert.alert('Generation Failed', error.message || 'Could not generate image');
      
      if (hapticFeedback) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const saveImage = async () => {
    if (!generatedImage) return;

    try {
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Required', 'Please allow access to save images');
        return;
      }

      // If base64, need to save to file first
      if (generatedImage.imageUri.startsWith('data:')) {
        const base64 = generatedImage.imageUri.split(',')[1];
        const filename = `${FileSystem.documentDirectory}enigma_${Date.now()}.png`;
        await FileSystem.writeAsStringAsync(filename, base64, {
          encoding: FileSystem.EncodingType.Base64,
        });
        await MediaLibrary.createAssetAsync(filename);
      } else {
        // Download and save URL
        const filename = `${FileSystem.documentDirectory}enigma_${Date.now()}.png`;
        await FileSystem.downloadAsync(generatedImage.imageUri, filename);
        await MediaLibrary.createAssetAsync(filename);
      }

      Alert.alert('Saved', 'Image saved to your gallery');
      
      if (hapticFeedback) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }
    } catch (error) {
      console.error('Save error:', error);
      Alert.alert('Error', 'Failed to save image');
    }
  };

  const shareImage = async () => {
    if (!generatedImage) return;

    try {
      await Share.share({
        message: `Generated with Enigma AI: "${generatedImage.prompt}"`,
        url: generatedImage.imageUri,
      });
    } catch (error) {
      console.error('Share error:', error);
    }
  };

  const useFromHistory = (image: GeneratedImage) => {
    setPrompt(image.prompt);
    setSelectedStyle(image.style);
    setSelectedSize(image.size);
    setShowHistory(false);
  };

  const styles = createStyles(darkMode);

  return (
    <ScrollView ref={scrollRef} style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={[styles.title, darkMode && styles.darkText]}>
          Image Generation
        </Text>
        <TouchableOpacity
          style={styles.historyButton}
          onPress={() => setShowHistory(!showHistory)}
        >
          <Text style={styles.historyButtonText}>
            {showHistory ? 'Hide History' : 'History'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* History Panel */}
      {showHistory && (
        <View style={styles.historyPanel}>
          <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
            RECENT GENERATIONS
          </Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {history.length === 0 ? (
              <Text style={[styles.emptyText, darkMode && styles.darkSubtext]}>
                No images generated yet
              </Text>
            ) : (
              history.map(item => (
                <TouchableOpacity
                  key={item.id}
                  style={styles.historyItem}
                  onPress={() => useFromHistory(item)}
                >
                  <Image
                    source={{ uri: item.imageUri }}
                    style={styles.historyThumbnail}
                  />
                  <Text
                    style={[styles.historyPrompt, darkMode && styles.darkSubtext]}
                    numberOfLines={2}
                  >
                    {item.prompt}
                  </Text>
                </TouchableOpacity>
              ))
            )}
          </ScrollView>
        </View>
      )}

      {/* Prompt Input */}
      <View style={styles.inputSection}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          PROMPT
        </Text>
        <TextInput
          style={[styles.promptInput, darkMode && styles.darkInput]}
          value={prompt}
          onChangeText={setPrompt}
          placeholder="Describe your image..."
          placeholderTextColor="#888"
          multiline
          numberOfLines={3}
        />
      </View>

      {/* Style Selection */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          STYLE
        </Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {IMAGE_STYLES.map(style => (
            <TouchableOpacity
              key={style.id}
              style={[
                styles.styleButton,
                darkMode && styles.darkStyleButton,
                selectedStyle === style.id && styles.styleButtonActive,
              ]}
              onPress={() => setSelectedStyle(style.id)}
            >
              <Text
                style={[
                  styles.styleButtonText,
                  selectedStyle === style.id && styles.styleButtonTextActive,
                ]}
              >
                {style.name}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Size Selection */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          SIZE
        </Text>
        <View style={styles.sizeGrid}>
          {IMAGE_SIZES.map(size => (
            <TouchableOpacity
              key={size.id}
              style={[
                styles.sizeButton,
                darkMode && styles.darkSizeButton,
                selectedSize === size.id && styles.sizeButtonActive,
              ]}
              onPress={() => setSelectedSize(size.id)}
            >
              <Text
                style={[
                  styles.sizeButtonText,
                  selectedSize === size.id && styles.sizeButtonTextActive,
                ]}
              >
                {size.name}
              </Text>
              <Text style={styles.sizeDimensions}>
                {size.width}x{size.height}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Advanced Settings */}
      <TouchableOpacity
        style={styles.advancedToggle}
        onPress={() => setShowAdvanced(!showAdvanced)}
      >
        <Text style={[styles.advancedToggleText, darkMode && styles.darkText]}>
          {showAdvanced ? 'Hide Advanced' : 'Show Advanced'}
        </Text>
      </TouchableOpacity>

      {showAdvanced && (
        <View style={[styles.advancedSection, darkMode && styles.darkCard]}>
          <View style={styles.advancedRow}>
            <Text style={[styles.advancedLabel, darkMode && styles.darkText]}>
              Negative Prompt
            </Text>
            <TextInput
              style={[styles.advancedInput, darkMode && styles.darkInput]}
              value={negativePrompt}
              onChangeText={setNegativePrompt}
              placeholder="What to avoid..."
              placeholderTextColor="#888"
            />
          </View>

          <View style={styles.advancedRow}>
            <Text style={[styles.advancedLabel, darkMode && styles.darkText]}>
              Steps: {steps}
            </Text>
            <View style={styles.stepperContainer}>
              <TouchableOpacity
                style={styles.stepperButton}
                onPress={() => setSteps(Math.max(10, steps - 5))}
              >
                <Text style={styles.stepperText}>-</Text>
              </TouchableOpacity>
              <Text style={[styles.stepperValue, darkMode && styles.darkText]}>
                {steps}
              </Text>
              <TouchableOpacity
                style={styles.stepperButton}
                onPress={() => setSteps(Math.min(100, steps + 5))}
              >
                <Text style={styles.stepperText}>+</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.advancedRow}>
            <Text style={[styles.advancedLabel, darkMode && styles.darkText]}>
              Guidance: {guidanceScale.toFixed(1)}
            </Text>
            <View style={styles.stepperContainer}>
              <TouchableOpacity
                style={styles.stepperButton}
                onPress={() => setGuidanceScale(Math.max(1, guidanceScale - 0.5))}
              >
                <Text style={styles.stepperText}>-</Text>
              </TouchableOpacity>
              <Text style={[styles.stepperValue, darkMode && styles.darkText]}>
                {guidanceScale.toFixed(1)}
              </Text>
              <TouchableOpacity
                style={styles.stepperButton}
                onPress={() => setGuidanceScale(Math.min(20, guidanceScale + 0.5))}
              >
                <Text style={styles.stepperText}>+</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.advancedRow}>
            <Text style={[styles.advancedLabel, darkMode && styles.darkText]}>
              Seed
            </Text>
            <TextInput
              style={[styles.seedInput, darkMode && styles.darkInput]}
              value={seed?.toString() || ''}
              onChangeText={text => setSeed(text ? parseInt(text) : null)}
              placeholder="Random"
              placeholderTextColor="#888"
              keyboardType="number-pad"
            />
          </View>
        </View>
      )}

      {/* Generate Button */}
      <TouchableOpacity
        style={[
          styles.generateButton,
          isGenerating && styles.generateButtonDisabled,
        ]}
        onPress={generateImage}
        disabled={isGenerating}
      >
        {isGenerating ? (
          <View style={styles.generatingContent}>
            <ActivityIndicator color="#fff" />
            <Text style={styles.generateButtonText}>Generating...</Text>
          </View>
        ) : (
          <Text style={styles.generateButtonText}>Generate Image</Text>
        )}
      </TouchableOpacity>

      {/* Generated Image Display */}
      {generatedImage && (
        <View style={styles.resultSection}>
          <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
            RESULT
          </Text>
          <View style={[styles.imageContainer, darkMode && styles.darkCard]}>
            <Image
              source={{ uri: generatedImage.imageUri }}
              style={styles.generatedImage}
              resizeMode="contain"
            />
          </View>

          <View style={styles.imageActions}>
            <TouchableOpacity style={styles.actionButton} onPress={saveImage}>
              <Text style={styles.actionButtonText}>Save</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionButton} onPress={shareImage}>
              <Text style={styles.actionButtonText}>Share</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionButton, styles.regenerateButton]}
              onPress={generateImage}
            >
              <Text style={[styles.actionButtonText, styles.regenerateText]}>
                Regenerate
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

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
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 16,
    },
    title: {
      fontSize: 28,
      fontWeight: 'bold',
      color: '#000',
    },
    darkText: {
      color: '#fff',
    },
    darkSubtext: {
      color: '#aaa',
    },
    historyButton: {
      padding: 8,
    },
    historyButtonText: {
      color: '#007AFF',
      fontWeight: '600',
    },
    historyPanel: {
      paddingHorizontal: 16,
      marginBottom: 16,
    },
    historyItem: {
      width: 100,
      marginRight: 12,
    },
    historyThumbnail: {
      width: 100,
      height: 100,
      borderRadius: 8,
    },
    historyPrompt: {
      fontSize: 11,
      marginTop: 4,
      color: '#666',
    },
    emptyText: {
      color: '#888',
      fontStyle: 'italic',
    },
    inputSection: {
      paddingHorizontal: 16,
      marginBottom: 16,
    },
    sectionTitle: {
      fontSize: 12,
      fontWeight: '600',
      color: '#888',
      marginBottom: 8,
    },
    promptInput: {
      backgroundColor: '#fff',
      borderRadius: 12,
      padding: 16,
      fontSize: 16,
      minHeight: 80,
      textAlignVertical: 'top',
    },
    darkInput: {
      backgroundColor: '#2a2a2a',
      color: '#fff',
    },
    section: {
      marginBottom: 16,
      paddingLeft: 16,
    },
    styleButton: {
      paddingHorizontal: 16,
      paddingVertical: 10,
      borderRadius: 20,
      backgroundColor: '#fff',
      marginRight: 8,
    },
    darkStyleButton: {
      backgroundColor: '#2a2a2a',
    },
    styleButtonActive: {
      backgroundColor: '#007AFF',
    },
    styleButtonText: {
      fontSize: 14,
      color: '#333',
    },
    styleButtonTextActive: {
      color: '#fff',
      fontWeight: '600',
    },
    sizeGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      paddingRight: 16,
    },
    sizeButton: {
      paddingHorizontal: 16,
      paddingVertical: 10,
      borderRadius: 12,
      backgroundColor: '#fff',
      marginRight: 8,
      marginBottom: 8,
      alignItems: 'center',
    },
    darkSizeButton: {
      backgroundColor: '#2a2a2a',
    },
    sizeButtonActive: {
      backgroundColor: '#007AFF',
    },
    sizeButtonText: {
      fontSize: 14,
      fontWeight: '600',
      color: '#333',
    },
    sizeButtonTextActive: {
      color: '#fff',
    },
    sizeDimensions: {
      fontSize: 10,
      color: '#888',
      marginTop: 2,
    },
    advancedToggle: {
      alignItems: 'center',
      padding: 12,
    },
    advancedToggleText: {
      color: '#007AFF',
      fontWeight: '600',
    },
    advancedSection: {
      backgroundColor: '#fff',
      marginHorizontal: 16,
      borderRadius: 12,
      padding: 16,
      marginBottom: 16,
    },
    darkCard: {
      backgroundColor: '#2a2a2a',
    },
    advancedRow: {
      marginBottom: 16,
    },
    advancedLabel: {
      fontSize: 14,
      fontWeight: '600',
      marginBottom: 8,
      color: '#333',
    },
    advancedInput: {
      backgroundColor: '#f0f0f0',
      borderRadius: 8,
      padding: 12,
      fontSize: 14,
    },
    stepperContainer: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    stepperButton: {
      width: 40,
      height: 40,
      borderRadius: 20,
      backgroundColor: '#007AFF',
      justifyContent: 'center',
      alignItems: 'center',
    },
    stepperText: {
      color: '#fff',
      fontSize: 24,
      fontWeight: 'bold',
    },
    stepperValue: {
      fontSize: 18,
      fontWeight: '600',
      marginHorizontal: 16,
      minWidth: 40,
      textAlign: 'center',
    },
    seedInput: {
      backgroundColor: '#f0f0f0',
      borderRadius: 8,
      padding: 12,
      fontSize: 14,
      width: 120,
    },
    generateButton: {
      backgroundColor: '#007AFF',
      marginHorizontal: 16,
      padding: 18,
      borderRadius: 12,
      alignItems: 'center',
      marginBottom: 16,
    },
    generateButtonDisabled: {
      opacity: 0.6,
    },
    generateButtonText: {
      color: '#fff',
      fontSize: 18,
      fontWeight: 'bold',
    },
    generatingContent: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
    },
    resultSection: {
      paddingHorizontal: 16,
    },
    imageContainer: {
      backgroundColor: '#fff',
      borderRadius: 12,
      padding: 8,
      alignItems: 'center',
    },
    generatedImage: {
      width: SCREEN_WIDTH - 48,
      height: SCREEN_WIDTH - 48,
      borderRadius: 8,
    },
    imageActions: {
      flexDirection: 'row',
      justifyContent: 'center',
      marginTop: 16,
      gap: 12,
    },
    actionButton: {
      paddingHorizontal: 24,
      paddingVertical: 12,
      borderRadius: 8,
      backgroundColor: '#fff',
      borderWidth: 1,
      borderColor: '#007AFF',
    },
    actionButtonText: {
      color: '#007AFF',
      fontWeight: '600',
    },
    regenerateButton: {
      backgroundColor: '#007AFF',
    },
    regenerateText: {
      color: '#fff',
    },
  });

export default ImageGenScreen;
