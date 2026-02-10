/**
 * Voice Assistant Screen
 * 
 * Dedicated voice-first interface for hands-free AI interaction.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Animated,
  Vibration,
  AppState,
  AppStateStatus,
  Platform,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';

// Types
interface VoiceMessage {
  id: string;
  text: string;
  role: 'user' | 'assistant';
  timestamp: number;
  audioUri?: string;
}

interface VoiceScreenProps {
  darkMode?: boolean;
  serverUrl: string;
  onMessage?: (message: VoiceMessage) => void;
}

type ListeningState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error';

const VoiceScreen: React.FC<VoiceScreenProps> = ({
  darkMode = false,
  serverUrl,
  onMessage,
}) => {
  const [state, setState] = useState<ListeningState>('idle');
  const [transcript, setTranscript] = useState('');
  const [messages, setMessages] = useState<VoiceMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isContinuousMode, setIsContinuousMode] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  // Animation values
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const waveAnim = useRef(new Animated.Value(0)).current;
  const levelAnim = useRef(new Animated.Value(0)).current;
  
  // Audio recording
  const recordingRef = useRef<Audio.Recording | null>(null);
  const soundRef = useRef<Audio.Sound | null>(null);
  
  const scrollRef = useRef<ScrollView>(null);

  // Setup audio
  useEffect(() => {
    setupAudio();
    return () => {
      cleanup();
    };
  }, []);

  // Handle app state changes
  useEffect(() => {
    const subscription = AppState.addEventListener('change', handleAppStateChange);
    return () => subscription.remove();
  }, []);

  // Animate based on state
  useEffect(() => {
    if (state === 'listening') {
      startPulseAnimation();
    } else if (state === 'processing') {
      startWaveAnimation();
    } else {
      stopAnimations();
    }
  }, [state]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  const setupAudio = async () => {
    try {
      await Audio.requestPermissionsAsync();
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: true,
      });
    } catch (error) {
      console.error('Audio setup error:', error);
    }
  };

  const cleanup = async () => {
    if (recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
      } catch {}
    }
    if (soundRef.current) {
      try {
        await soundRef.current.unloadAsync();
      } catch {}
    }
  };

  const handleAppStateChange = (nextState: AppStateStatus) => {
    if (nextState !== 'active') {
      stopRecording();
    }
  };

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 500,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 500,
          useNativeDriver: true,
        }),
      ])
    ).start();
  };

  const startWaveAnimation = () => {
    Animated.loop(
      Animated.timing(waveAnim, {
        toValue: 1,
        duration: 1500,
        useNativeDriver: true,
      })
    ).start();
  };

  const stopAnimations = () => {
    pulseAnim.setValue(1);
    waveAnim.setValue(0);
  };

  const startRecording = async () => {
    try {
      setState('listening');
      setError(null);
      setTranscript('');

      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      const recording = new Audio.Recording();
      await recording.prepareToRecordAsync({
        android: {
          extension: '.wav',
          outputFormat: Audio.AndroidOutputFormat.DEFAULT,
          audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
          sampleRate: 16000,
          numberOfChannels: 1,
          bitRate: 128000,
        },
        ios: {
          extension: '.wav',
          outputFormat: Audio.IOSOutputFormat.LINEARPCM,
          audioQuality: Audio.IOSAudioQuality.HIGH,
          sampleRate: 16000,
          numberOfChannels: 1,
          bitRate: 128000,
          linearPCMBitDepth: 16,
          linearPCMIsBigEndian: false,
          linearPCMIsFloat: false,
        },
        web: {
          mimeType: 'audio/wav',
          bitsPerSecond: 128000,
        },
      });

      // Monitor audio levels
      recording.setOnRecordingStatusUpdate(status => {
        if (status.isRecording) {
          const level = status.metering ? Math.max(0, (status.metering + 160) / 160) : 0;
          setAudioLevel(level);
          Animated.timing(levelAnim, {
            toValue: level,
            duration: 100,
            useNativeDriver: true,
          }).start();
        }
      });

      await recording.startAsync();
      recordingRef.current = recording;

    } catch (error: any) {
      console.error('Recording error:', error);
      setState('error');
      setError('Failed to start recording');
    }
  };

  const stopRecording = async () => {
    if (!recordingRef.current) return;

    try {
      setState('processing');
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      if (uri) {
        await processAudio(uri);
      } else {
        setState('idle');
      }

    } catch (error: any) {
      console.error('Stop recording error:', error);
      setState('error');
      setError('Failed to process recording');
    }
  };

  const processAudio = async (audioUri: string) => {
    try {
      // Transcribe audio
      const transcriptText = await transcribeAudio(audioUri);
      
      if (!transcriptText.trim()) {
        setState('idle');
        return;
      }

      setTranscript(transcriptText);

      // Add user message
      const userMessage: VoiceMessage = {
        id: Date.now().toString(),
        text: transcriptText,
        role: 'user',
        timestamp: Date.now(),
        audioUri,
      };
      setMessages(prev => [...prev, userMessage]);
      onMessage?.(userMessage);

      // Get AI response
      const response = await getAIResponse(transcriptText);

      // Add assistant message
      const assistantMessage: VoiceMessage = {
        id: (Date.now() + 1).toString(),
        text: response,
        role: 'assistant',
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, assistantMessage]);
      onMessage?.(assistantMessage);

      // Speak response
      await speakResponse(response);

    } catch (error: any) {
      console.error('Process audio error:', error);
      setState('error');
      setError(error.message || 'Failed to process audio');
    }
  };

  const transcribeAudio = async (audioUri: string): Promise<string> => {
    // Create form data with audio file
    const formData = new FormData();
    formData.append('file', {
      uri: audioUri,
      type: 'audio/wav',
      name: 'recording.wav',
    } as any);
    formData.append('model', 'whisper-1');

    const response = await fetch(`${serverUrl}/v1/audio/transcriptions`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      // Fallback: use device speech recognition
      throw new Error('Transcription failed');
    }

    const data = await response.json();
    return data.text || '';
  };

  const getAIResponse = async (message: string): Promise<string> => {
    const response = await fetch(`${serverUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
          { role: 'system', content: 'You are a helpful voice assistant. Keep responses concise and conversational.' },
          ...messages.map(m => ({ role: m.role, content: m.text })),
          { role: 'user', content: message },
        ],
        max_tokens: 200,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to get response');
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || 'I could not generate a response.';
  };

  const speakResponse = async (text: string) => {
    setState('speaking');
    
    return new Promise<void>((resolve) => {
      Speech.speak(text, {
        language: 'en-US',
        pitch: 1.0,
        rate: Platform.OS === 'ios' ? 0.5 : 0.9,
        onDone: () => {
          setState(isContinuousMode ? 'listening' : 'idle');
          if (isContinuousMode) {
            startRecording();
          }
          resolve();
        },
        onError: () => {
          setState('idle');
          resolve();
        },
      });
    });
  };

  const toggleContinuousMode = () => {
    setIsContinuousMode(!isContinuousMode);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const clearMessages = () => {
    setMessages([]);
    setTranscript('');
  };

  const stopSpeaking = () => {
    Speech.stop();
    setState('idle');
  };

  const handleMicPress = () => {
    if (state === 'idle') {
      startRecording();
    } else if (state === 'listening') {
      stopRecording();
    } else if (state === 'speaking') {
      stopSpeaking();
    }
  };

  const getStateText = () => {
    switch (state) {
      case 'idle': return 'Tap to speak';
      case 'listening': return 'Listening...';
      case 'processing': return 'Processing...';
      case 'speaking': return 'Speaking...';
      case 'error': return error || 'Error occurred';
      default: return '';
    }
  };

  const getStateColor = () => {
    switch (state) {
      case 'idle': return '#007AFF';
      case 'listening': return '#FF3B30';
      case 'processing': return '#FF9500';
      case 'speaking': return '#34C759';
      case 'error': return '#FF3B30';
      default: return '#007AFF';
    }
  };

  const styles = createStyles(darkMode);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={[styles.title, darkMode && styles.darkText]}>
          Voice Assistant
        </Text>
        <View style={styles.headerActions}>
          <TouchableOpacity
            style={[
              styles.modeButton,
              isContinuousMode && styles.modeButtonActive,
            ]}
            onPress={toggleContinuousMode}
          >
            <Text
              style={[
                styles.modeButtonText,
                isContinuousMode && styles.modeButtonTextActive,
              ]}
            >
              Continuous
            </Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.clearButton} onPress={clearMessages}>
            <Text style={styles.clearButtonText}>Clear</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Messages */}
      <ScrollView
        ref={scrollRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
      >
        {messages.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={[styles.emptyTitle, darkMode && styles.darkText]}>
              Start a conversation
            </Text>
            <Text style={[styles.emptySubtitle, darkMode && styles.darkSubtext]}>
              Tap the microphone and speak
            </Text>
          </View>
        ) : (
          messages.map(message => (
            <View
              key={message.id}
              style={[
                styles.messageBubble,
                message.role === 'user'
                  ? styles.userBubble
                  : [styles.assistantBubble, darkMode && styles.darkAssistantBubble],
              ]}
            >
              <Text
                style={[
                  styles.messageText,
                  message.role === 'user'
                    ? styles.userText
                    : [styles.assistantText, darkMode && styles.darkText],
                ]}
              >
                {message.text}
              </Text>
            </View>
          ))
        )}
      </ScrollView>

      {/* Voice UI */}
      <View style={styles.voiceContainer}>
        {/* Current transcript */}
        {transcript && (
          <View style={[styles.transcriptBox, darkMode && styles.darkCard]}>
            <Text style={[styles.transcriptText, darkMode && styles.darkText]}>
              {transcript}
            </Text>
          </View>
        )}

        {/* Status text */}
        <Text style={[styles.statusText, { color: getStateColor() }]}>
          {getStateText()}
        </Text>

        {/* Audio level indicator */}
        {state === 'listening' && (
          <View style={styles.levelContainer}>
            {[...Array(5)].map((_, i) => (
              <Animated.View
                key={i}
                style={[
                  styles.levelBar,
                  {
                    height: Animated.multiply(levelAnim, 30 + i * 5),
                    backgroundColor: getStateColor(),
                  },
                ]}
              />
            ))}
          </View>
        )}

        {/* Main microphone button */}
        <TouchableOpacity
          style={styles.micButtonContainer}
          onPress={handleMicPress}
          activeOpacity={0.8}
        >
          <Animated.View
            style={[
              styles.micButton,
              { 
                backgroundColor: getStateColor(),
                transform: [{ scale: state === 'listening' ? pulseAnim : 1 }],
              },
            ]}
          >
            <Text style={styles.micIcon}>
              {state === 'listening' ? 'Stop' : state === 'speaking' ? 'Stop' : 'Mic'}
            </Text>
          </Animated.View>
        </TouchableOpacity>

        {/* Instructions */}
        <Text style={[styles.instructions, darkMode && styles.darkSubtext]}>
          {state === 'listening'
            ? 'Tap to stop recording'
            : state === 'speaking'
            ? 'Tap to stop speaking'
            : 'Hold and release to send'}
        </Text>
      </View>
    </View>
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
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: darkMode ? '#333' : '#e0e0e0',
    },
    title: {
      fontSize: 24,
      fontWeight: 'bold',
      color: '#000',
    },
    darkText: {
      color: '#fff',
    },
    darkSubtext: {
      color: '#aaa',
    },
    headerActions: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
    },
    modeButton: {
      paddingHorizontal: 12,
      paddingVertical: 6,
      borderRadius: 16,
      backgroundColor: '#e0e0e0',
    },
    modeButtonActive: {
      backgroundColor: '#007AFF',
    },
    modeButtonText: {
      fontSize: 12,
      fontWeight: '600',
      color: '#333',
    },
    modeButtonTextActive: {
      color: '#fff',
    },
    clearButton: {
      padding: 8,
    },
    clearButtonText: {
      color: '#007AFF',
      fontWeight: '600',
    },
    messagesContainer: {
      flex: 1,
    },
    messagesContent: {
      padding: 16,
    },
    emptyState: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      paddingTop: 100,
    },
    emptyTitle: {
      fontSize: 20,
      fontWeight: '600',
      marginBottom: 8,
    },
    emptySubtitle: {
      fontSize: 14,
      color: '#888',
    },
    messageBubble: {
      maxWidth: '80%',
      padding: 12,
      borderRadius: 16,
      marginBottom: 12,
    },
    userBubble: {
      alignSelf: 'flex-end',
      backgroundColor: '#007AFF',
    },
    assistantBubble: {
      alignSelf: 'flex-start',
      backgroundColor: '#fff',
    },
    darkAssistantBubble: {
      backgroundColor: '#2a2a2a',
    },
    messageText: {
      fontSize: 16,
      lineHeight: 22,
    },
    userText: {
      color: '#fff',
    },
    assistantText: {
      color: '#000',
    },
    voiceContainer: {
      padding: 24,
      alignItems: 'center',
      borderTopWidth: StyleSheet.hairlineWidth,
      borderTopColor: darkMode ? '#333' : '#e0e0e0',
    },
    transcriptBox: {
      backgroundColor: '#fff',
      padding: 12,
      borderRadius: 12,
      marginBottom: 16,
      width: '100%',
    },
    darkCard: {
      backgroundColor: '#2a2a2a',
    },
    transcriptText: {
      fontSize: 14,
      textAlign: 'center',
    },
    statusText: {
      fontSize: 16,
      fontWeight: '600',
      marginBottom: 16,
    },
    levelContainer: {
      flexDirection: 'row',
      height: 40,
      marginBottom: 16,
      alignItems: 'flex-end',
    },
    levelBar: {
      width: 8,
      marginHorizontal: 3,
      borderRadius: 4,
    },
    micButtonContainer: {
      marginBottom: 16,
    },
    micButton: {
      width: 80,
      height: 80,
      borderRadius: 40,
      justifyContent: 'center',
      alignItems: 'center',
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.3,
      shadowRadius: 8,
      elevation: 8,
    },
    micIcon: {
      color: '#fff',
      fontSize: 16,
      fontWeight: 'bold',
    },
    instructions: {
      fontSize: 12,
      color: '#888',
    },
  });

export default VoiceScreen;
