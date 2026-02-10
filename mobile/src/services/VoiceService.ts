/**
 * Voice-First Interface Service for Enigma AI Mobile
 * 
 * Provides:
 * - Hold-to-talk recording
 * - Continuous conversation mode
 * - Wake word detection ("Hey Enigma")
 * - Speech-to-text transcription
 * - Text-to-speech responses
 * - Audio streaming
 */

import { Platform, AppState, AppStateStatus } from 'react-native';

// Audio recording/playback types
interface AudioRecording {
  uri: string;
  duration: number;
  mimeType: string;
}

interface TranscriptionResult {
  text: string;
  confidence: number;
  language?: string;
  segments?: TranscriptionSegment[];
}

interface TranscriptionSegment {
  text: string;
  start: number;
  end: number;
  confidence: number;
}

interface VoiceConfig {
  /** Server URL for API calls */
  serverUrl: string;
  /** Enable wake word detection */
  enableWakeWord: boolean;
  /** Wake word phrase (default: "hey enigma") */
  wakeWordPhrase: string;
  /** Auto-play TTS responses */
  autoPlayResponses: boolean;
  /** Preferred TTS voice */
  preferredVoice?: string;
  /** Speech recognition language */
  speechLanguage: string;
  /** Enable continuous listening after response */
  continuousMode: boolean;
  /** Silence threshold for auto-stop (ms) */
  silenceThreshold: number;
  /** Max recording duration (ms) */
  maxRecordingDuration: number;
  /** Enable haptic feedback */
  hapticFeedback: boolean;
  /** Audio sample rate */
  sampleRate: number;
}

interface VoiceState {
  isRecording: boolean;
  isProcessing: boolean;
  isPlaying: boolean;
  isListeningForWakeWord: boolean;
  currentTranscript: string;
  error: string | null;
  audioLevel: number;
  recordingDuration: number;
}

type VoiceEventType = 
  | 'recording_started'
  | 'recording_stopped'
  | 'transcript_partial'
  | 'transcript_final'
  | 'response_started'
  | 'response_finished'
  | 'wake_word_detected'
  | 'error'
  | 'audio_level';

type VoiceEventCallback = (event: VoiceEventType, data?: any) => void;

const DEFAULT_CONFIG: VoiceConfig = {
  serverUrl: '',
  enableWakeWord: false,
  wakeWordPhrase: 'hey enigma',
  autoPlayResponses: true,
  preferredVoice: undefined,
  speechLanguage: 'en-US',
  continuousMode: false,
  silenceThreshold: 1500,
  maxRecordingDuration: 60000,
  hapticFeedback: true,
  sampleRate: 16000,
};

/**
 * Voice-First Interface Service
 * 
 * Example usage:
 * ```typescript
 * const voice = VoiceService.getInstance();
 * voice.configure({ serverUrl: 'http://192.168.1.100:8080' });
 * 
 * // Start hold-to-talk
 * voice.startRecording();
 * // ... user speaks ...
 * const result = await voice.stopRecording();
 * 
 * // Or use continuous mode
 * voice.startContinuousConversation();
 * ```
 */
class VoiceService {
  private static instance: VoiceService;
  
  private config: VoiceConfig = { ...DEFAULT_CONFIG };
  private state: VoiceState = {
    isRecording: false,
    isProcessing: false,
    isPlaying: false,
    isListeningForWakeWord: false,
    currentTranscript: '',
    error: null,
    audioLevel: 0,
    recordingDuration: 0,
  };
  
  private eventListeners: Map<string, Set<VoiceEventCallback>> = new Map();
  private recordingTimer: ReturnType<typeof setInterval> | null = null;
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  private appStateSubscription: any = null;
  
  // Audio components (will be initialized based on platform)
  private audioRecorder: any = null;
  private audioPlayer: any = null;
  private wakeWordDetector: any = null;
  
  private constructor() {
    // Handle app state changes
    this.appStateSubscription = AppState.addEventListener(
      'change',
      this.handleAppStateChange.bind(this)
    );
  }
  
  public static getInstance(): VoiceService {
    if (!VoiceService.instance) {
      VoiceService.instance = new VoiceService();
    }
    return VoiceService.instance;
  }
  
  // ===========================================================================
  // Configuration
  // ===========================================================================
  
  public configure(config: Partial<VoiceConfig>): void {
    this.config = { ...this.config, ...config };
  }
  
  public getConfig(): VoiceConfig {
    return { ...this.config };
  }
  
  public getState(): VoiceState {
    return { ...this.state };
  }
  
  // ===========================================================================
  // Event Handling
  // ===========================================================================
  
  public addEventListener(
    event: VoiceEventType,
    callback: VoiceEventCallback
  ): () => void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.eventListeners.get(event)?.delete(callback);
    };
  }
  
  private emit(event: VoiceEventType, data?: any): void {
    this.eventListeners.get(event)?.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error(`Voice event callback error: ${error}`);
      }
    });
  }
  
  // ===========================================================================
  // Recording - Hold-to-Talk
  // ===========================================================================
  
  /**
   * Start recording audio (hold-to-talk).
   * Call stopRecording() when user releases button.
   */
  public async startRecording(): Promise<boolean> {
    if (this.state.isRecording) {
      return true;
    }
    
    try {
      // Request microphone permission if needed
      const hasPermission = await this.requestMicrophonePermission();
      if (!hasPermission) {
        this.setError('Microphone permission denied');
        return false;
      }
      
      // Initialize recorder if needed
      await this.initializeRecorder();
      
      // Start recording
      await this.audioRecorder?.startAsync?.();
      
      this.state.isRecording = true;
      this.state.recordingDuration = 0;
      this.state.error = null;
      
      // Start duration timer
      this.recordingTimer = setInterval(() => {
        this.state.recordingDuration += 100;
        
        // Auto-stop if max duration reached
        if (this.state.recordingDuration >= this.config.maxRecordingDuration) {
          this.stopRecording();
        }
      }, 100);
      
      // Haptic feedback
      if (this.config.hapticFeedback) {
        this.triggerHaptic('impactLight');
      }
      
      this.emit('recording_started');
      return true;
    } catch (error) {
      this.setError(`Failed to start recording: ${error}`);
      return false;
    }
  }
  
  /**
   * Stop recording and transcribe audio.
   * Returns transcription result.
   */
  public async stopRecording(): Promise<TranscriptionResult | null> {
    if (!this.state.isRecording) {
      return null;
    }
    
    // Clear timers
    if (this.recordingTimer) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
    
    try {
      // Stop recording
      const recording = await this.audioRecorder?.stopAndUnloadAsync?.();
      this.state.isRecording = false;
      
      // Haptic feedback
      if (this.config.hapticFeedback) {
        this.triggerHaptic('impactMedium');
      }
      
      this.emit('recording_stopped', { duration: this.state.recordingDuration });
      
      // Transcribe
      if (recording?.uri || this.audioRecorder?.getURI?.()) {
        const uri = recording?.uri || this.audioRecorder.getURI();
        return await this.transcribeAudio(uri);
      }
      
      return null;
    } catch (error) {
      this.state.isRecording = false;
      this.setError(`Failed to stop recording: ${error}`);
      return null;
    }
  }
  
  /**
   * Cancel current recording without processing.
   */
  public async cancelRecording(): Promise<void> {
    if (!this.state.isRecording) {
      return;
    }
    
    if (this.recordingTimer) {
      clearInterval(this.recordingTimer);
      this.recordingTimer = null;
    }
    
    try {
      await this.audioRecorder?.stopAndUnloadAsync?.();
    } catch (error) {
      // Ignore errors on cancel
    }
    
    this.state.isRecording = false;
    this.state.recordingDuration = 0;
  }
  
  // ===========================================================================
  // Transcription
  // ===========================================================================
  
  /**
   * Transcribe audio file to text.
   */
  public async transcribeAudio(audioUri: string): Promise<TranscriptionResult | null> {
    if (!this.config.serverUrl) {
      this.setError('Server URL not configured');
      return null;
    }
    
    this.state.isProcessing = true;
    
    try {
      // Create form data with audio file
      const formData = new FormData();
      formData.append('audio', {
        uri: audioUri,
        type: 'audio/wav',
        name: 'recording.wav',
      } as any);
      formData.append('language', this.config.speechLanguage);
      
      // Send to server for transcription
      const response = await fetch(
        `${this.config.serverUrl}/api/voice/transcribe`,
        {
          method: 'POST',
          body: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      if (!response.ok) {
        throw new Error(`Transcription failed: ${response.status}`);
      }
      
      const result = await response.json();
      
      this.state.currentTranscript = result.text || '';
      this.emit('transcript_final', result);
      
      return {
        text: result.text || '',
        confidence: result.confidence || 1.0,
        language: result.language,
        segments: result.segments,
      };
    } catch (error) {
      this.setError(`Transcription error: ${error}`);
      return null;
    } finally {
      this.state.isProcessing = false;
    }
  }
  
  // ===========================================================================
  // Text-to-Speech
  // ===========================================================================
  
  /**
   * Speak text using TTS.
   */
  public async speak(text: string): Promise<void> {
    if (!text || this.state.isPlaying) {
      return;
    }
    
    this.state.isPlaying = true;
    this.emit('response_started', { text });
    
    try {
      if (this.config.serverUrl) {
        // Use server TTS
        const response = await fetch(
          `${this.config.serverUrl}/api/voice/synthesize`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text,
              voice: this.config.preferredVoice,
            }),
          }
        );
        
        if (!response.ok) {
          throw new Error(`TTS failed: ${response.status}`);
        }
        
        // Get audio data and play
        const audioBlob = await response.blob();
        await this.playAudioBlob(audioBlob);
      } else {
        // Use device TTS as fallback
        await this.speakWithDeviceTTS(text);
      }
    } catch (error) {
      this.setError(`TTS error: ${error}`);
      // Fall back to device TTS
      await this.speakWithDeviceTTS(text);
    } finally {
      this.state.isPlaying = false;
      this.emit('response_finished');
    }
  }
  
  /**
   * Stop current TTS playback.
   */
  public async stopSpeaking(): Promise<void> {
    if (this.audioPlayer) {
      await this.audioPlayer.stopAsync?.();
    }
    this.state.isPlaying = false;
  }
  
  // ===========================================================================
  // Continuous Conversation Mode
  // ===========================================================================
  
  /**
   * Start continuous conversation mode.
   * Automatically listens after each response.
   */
  public async startContinuousConversation(): Promise<void> {
    this.config.continuousMode = true;
    
    // Start wake word detection or recording
    if (this.config.enableWakeWord) {
      await this.startWakeWordDetection();
    } else {
      await this.startRecording();
    }
  }
  
  /**
   * Stop continuous conversation mode.
   */
  public async stopContinuousConversation(): Promise<void> {
    this.config.continuousMode = false;
    await this.stopWakeWordDetection();
    await this.cancelRecording();
    await this.stopSpeaking();
  }
  
  // ===========================================================================
  // Wake Word Detection
  // ===========================================================================
  
  /**
   * Start listening for wake word.
   */
  public async startWakeWordDetection(): Promise<boolean> {
    if (this.state.isListeningForWakeWord) {
      return true;
    }
    
    try {
      const hasPermission = await this.requestMicrophonePermission();
      if (!hasPermission) {
        this.setError('Microphone permission denied');
        return false;
      }
      
      // Initialize wake word detector
      // This would typically use a library like Porcupine or Snowboy
      // For now, we'll use a simplified approach
      this.state.isListeningForWakeWord = true;
      
      console.log('Wake word detection started');
      return true;
    } catch (error) {
      this.setError(`Wake word detection error: ${error}`);
      return false;
    }
  }
  
  /**
   * Stop wake word detection.
   */
  public async stopWakeWordDetection(): Promise<void> {
    this.state.isListeningForWakeWord = false;
    this.wakeWordDetector?.stop?.();
  }
  
  // ===========================================================================
  // Voice Conversation (Full Flow)
  // ===========================================================================
  
  /**
   * Process a voice message end-to-end.
   * Records -> Transcribes -> Gets AI response -> Speaks response
   */
  public async processVoiceMessage(
    onTranscript?: (text: string) => void,
    onResponse?: (text: string) => void
  ): Promise<{ transcript: string; response: string } | null> {
    // Record
    await this.startRecording();
    
    // Wait for user to finish speaking (or max duration)
    await this.waitForSilence();
    
    // Stop and transcribe
    const transcription = await this.stopRecording();
    if (!transcription?.text) {
      return null;
    }
    
    onTranscript?.(transcription.text);
    
    // Get AI response
    const response = await this.getAIResponse(transcription.text);
    if (!response) {
      return null;
    }
    
    onResponse?.(response);
    
    // Speak response
    if (this.config.autoPlayResponses) {
      await this.speak(response);
    }
    
    // Continue if in continuous mode
    if (this.config.continuousMode) {
      // Brief delay then start listening again
      setTimeout(() => {
        if (this.config.continuousMode) {
          this.startRecording();
        }
      }, 500);
    }
    
    return {
      transcript: transcription.text,
      response,
    };
  }
  
  /**
   * Get AI response from server.
   */
  private async getAIResponse(text: string): Promise<string | null> {
    if (!this.config.serverUrl) {
      return null;
    }
    
    try {
      const response = await fetch(`${this.config.serverUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      
      if (!response.ok) {
        throw new Error(`Chat API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data.response || data.message || null;
    } catch (error) {
      this.setError(`Chat error: ${error}`);
      return null;
    }
  }
  
  // ===========================================================================
  // Audio Level Monitoring
  // ===========================================================================
  
  /**
   * Get current audio input level (0-1).
   */
  public getAudioLevel(): number {
    return this.state.audioLevel;
  }
  
  /**
   * Wait for silence (user stopped speaking).
   */
  private async waitForSilence(): Promise<void> {
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        // Check if audio level is below threshold
        if (this.state.audioLevel < 0.1) {
          // Start silence timer
          if (!this.silenceTimer) {
            this.silenceTimer = setTimeout(() => {
              clearInterval(checkInterval);
              resolve();
            }, this.config.silenceThreshold);
          }
        } else {
          // Reset silence timer if audio detected
          if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
          }
        }
        
        // Also resolve if max duration reached
        if (this.state.recordingDuration >= this.config.maxRecordingDuration) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
    });
  }
  
  // ===========================================================================
  // Platform-Specific Helpers
  // ===========================================================================
  
  private async requestMicrophonePermission(): Promise<boolean> {
    // Platform-specific permission request
    // In a real implementation, this would use react-native-permissions
    // or expo-av for permission handling
    try {
      // Simulated permission check
      console.log('Requesting microphone permission...');
      return true;
    } catch (error) {
      console.error('Permission error:', error);
      return false;
    }
  }
  
  private async initializeRecorder(): Promise<void> {
    // Initialize audio recorder
    // In a real implementation, this would use expo-av or react-native-audio-recorder
    console.log('Initializing audio recorder...');
  }
  
  private async playAudioBlob(blob: Blob): Promise<void> {
    // Play audio blob
    // In a real implementation, this would use expo-av or react-native-sound
    console.log('Playing audio...');
  }
  
  private async speakWithDeviceTTS(text: string): Promise<void> {
    // Use device TTS
    // In a real implementation, this would use expo-speech or react-native-tts
    console.log('Speaking:', text);
  }
  
  private triggerHaptic(type: string): void {
    // Trigger haptic feedback
    // In a real implementation, this would use react-native-haptic-feedback
    console.log('Haptic:', type);
  }
  
  private handleAppStateChange(state: AppStateStatus): void {
    if (state !== 'active') {
      // App went to background - stop recording
      if (this.state.isRecording) {
        this.cancelRecording();
      }
      if (this.state.isPlaying) {
        this.stopSpeaking();
      }
    }
  }
  
  private setError(error: string): void {
    this.state.error = error;
    this.emit('error', { error });
    console.error('VoiceService:', error);
  }
  
  // ===========================================================================
  // Cleanup
  // ===========================================================================
  
  public async cleanup(): Promise<void> {
    await this.cancelRecording();
    await this.stopSpeaking();
    await this.stopWakeWordDetection();
    
    if (this.appStateSubscription) {
      this.appStateSubscription.remove();
    }
  }
}

// Export singleton instance getter
export const getVoiceService = (): VoiceService => VoiceService.getInstance();

export {
  VoiceService,
  VoiceConfig,
  VoiceState,
  VoiceEventType,
  TranscriptionResult,
  TranscriptionSegment,
  AudioRecording,
};

export default VoiceService;
