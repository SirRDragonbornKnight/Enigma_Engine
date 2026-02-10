/**
 * Camera Integration Service for Enigma AI Mobile
 * 
 * Provides:
 * - Photo capture for vision analysis
 * - Camera preview with overlay
 * - Image upload to PC for processing
 * - Screenshot capture
 * - QR/barcode scanning
 * - OCR text extraction
 */

import { Platform, Dimensions } from 'react-native';

// Types
interface CameraConfig {
  /** Server URL for API calls */
  serverUrl: string;
  /** Default camera (front/back) */
  defaultCamera: 'front' | 'back';
  /** Image quality (0-1) */
  imageQuality: number;
  /** Max image dimension for upload */
  maxDimension: number;
  /** Enable flash */
  flashEnabled: boolean;
  /** Auto-analyze photos after capture */
  autoAnalyze: boolean;
  /** Store photos locally */
  storeLocally: boolean;
  /** Local storage path */
  storagePath: string;
}

interface CapturedImage {
  uri: string;
  width: number;
  height: number;
  base64?: string;
  timestamp: number;
  metadata?: ImageMetadata;
}

interface ImageMetadata {
  location?: {
    latitude: number;
    longitude: number;
  };
  deviceOrientation?: string;
  cameraType?: 'front' | 'back';
}

interface VisionAnalysisResult {
  description: string;
  objects: DetectedObject[];
  text?: OCRResult;
  faces?: DetectedFace[];
  colors?: ColorInfo[];
  tags?: string[];
  confidence: number;
  processingTime: number;
}

interface DetectedObject {
  label: string;
  confidence: number;
  boundingBox?: BoundingBox;
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DetectedFace {
  boundingBox: BoundingBox;
  landmarks?: FaceLandmarks;
  emotion?: string;
  ageRange?: [number, number];
}

interface FaceLandmarks {
  leftEye?: { x: number; y: number };
  rightEye?: { x: number; y: number };
  nose?: { x: number; y: number };
  leftMouth?: { x: number; y: number };
  rightMouth?: { x: number; y: number };
}

interface OCRResult {
  fullText: string;
  blocks: TextBlock[];
  language?: string;
}

interface TextBlock {
  text: string;
  boundingBox: BoundingBox;
  confidence: number;
}

interface ColorInfo {
  color: string;
  hex: string;
  percentage: number;
}

interface ScanResult {
  type: 'qr' | 'barcode';
  data: string;
  format?: string;
  timestamp: number;
}

type CameraEventType =
  | 'capture_started'
  | 'capture_complete'
  | 'analysis_started'
  | 'analysis_complete'
  | 'scan_detected'
  | 'error';

type CameraEventCallback = (event: CameraEventType, data?: any) => void;

const DEFAULT_CONFIG: CameraConfig = {
  serverUrl: '',
  defaultCamera: 'back',
  imageQuality: 0.8,
  maxDimension: 1920,
  flashEnabled: false,
  autoAnalyze: true,
  storeLocally: false,
  storagePath: '',
};

/**
 * Camera Integration Service
 * 
 * Example usage:
 * ```typescript
 * const camera = CameraService.getInstance();
 * camera.configure({ serverUrl: 'http://192.168.1.100:8080' });
 * 
 * // Capture and analyze
 * const result = await camera.captureAndAnalyze();
 * console.log(result.description);
 * 
 * // Or just capture
 * const image = await camera.capturePhoto();
 * 
 * // Analyze existing image
 * const analysis = await camera.analyzeImage(image.uri);
 * ```
 */
class CameraService {
  private static instance: CameraService;
  
  private config: CameraConfig = { ...DEFAULT_CONFIG };
  private isInitialized: boolean = false;
  private eventListeners: Map<string, Set<CameraEventCallback>> = new Map();
  
  // Camera state
  private isCameraOpen: boolean = false;
  private isCapturing: boolean = false;
  private isAnalyzing: boolean = false;
  private currentCamera: 'front' | 'back' = 'back';
  
  // Recent captures for history
  private captureHistory: CapturedImage[] = [];
  private maxHistorySize: number = 50;
  
  private constructor() {}
  
  public static getInstance(): CameraService {
    if (!CameraService.instance) {
      CameraService.instance = new CameraService();
    }
    return CameraService.instance;
  }
  
  // ===========================================================================
  // Configuration
  // ===========================================================================
  
  public configure(config: Partial<CameraConfig>): void {
    this.config = { ...this.config, ...config };
    this.currentCamera = this.config.defaultCamera;
  }
  
  public getConfig(): CameraConfig {
    return { ...this.config };
  }
  
  // ===========================================================================
  // Event Handling
  // ===========================================================================
  
  public addEventListener(
    event: CameraEventType,
    callback: CameraEventCallback
  ): () => void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
    
    return () => {
      this.eventListeners.get(event)?.delete(callback);
    };
  }
  
  private emit(event: CameraEventType, data?: any): void {
    this.eventListeners.get(event)?.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error(`Camera event callback error: ${error}`);
      }
    });
  }
  
  // ===========================================================================
  // Camera Operations
  // ===========================================================================
  
  /**
   * Initialize camera (request permissions, etc.)
   */
  public async initialize(): Promise<boolean> {
    if (this.isInitialized) {
      return true;
    }
    
    try {
      const hasPermission = await this.requestCameraPermission();
      if (!hasPermission) {
        this.emit('error', { error: 'Camera permission denied' });
        return false;
      }
      
      this.isInitialized = true;
      return true;
    } catch (error) {
      this.emit('error', { error: `Camera init failed: ${error}` });
      return false;
    }
  }
  
  /**
   * Switch between front and back camera.
   */
  public switchCamera(): 'front' | 'back' {
    this.currentCamera = this.currentCamera === 'front' ? 'back' : 'front';
    return this.currentCamera;
  }
  
  /**
   * Toggle flash on/off.
   */
  public toggleFlash(): boolean {
    this.config.flashEnabled = !this.config.flashEnabled;
    return this.config.flashEnabled;
  }
  
  /**
   * Capture a photo.
   */
  public async capturePhoto(): Promise<CapturedImage | null> {
    if (this.isCapturing) {
      return null;
    }
    
    if (!this.isInitialized) {
      await this.initialize();
    }
    
    this.isCapturing = true;
    this.emit('capture_started');
    
    try {
      // In a real implementation, this would use react-native-camera
      // or expo-camera for actual photo capture
      const mockCapture: CapturedImage = {
        uri: `file://photo_${Date.now()}.jpg`,
        width: 1920,
        height: 1080,
        timestamp: Date.now(),
        metadata: {
          cameraType: this.currentCamera,
          deviceOrientation: 'portrait',
        },
      };
      
      // Add to history
      this.addToHistory(mockCapture);
      
      this.emit('capture_complete', mockCapture);
      return mockCapture;
    } catch (error) {
      this.emit('error', { error: `Capture failed: ${error}` });
      return null;
    } finally {
      this.isCapturing = false;
    }
  }
  
  /**
   * Capture photo and analyze with vision AI.
   */
  public async captureAndAnalyze(): Promise<{
    image: CapturedImage;
    analysis: VisionAnalysisResult;
  } | null> {
    const image = await this.capturePhoto();
    if (!image) {
      return null;
    }
    
    const analysis = await this.analyzeImage(image.uri);
    if (!analysis) {
      return null;
    }
    
    return { image, analysis };
  }
  
  // ===========================================================================
  // Vision Analysis
  // ===========================================================================
  
  /**
   * Analyze an image using the PC's vision AI.
   */
  public async analyzeImage(
    imageUri: string,
    prompt?: string
  ): Promise<VisionAnalysisResult | null> {
    if (!this.config.serverUrl) {
      this.emit('error', { error: 'Server URL not configured' });
      return null;
    }
    
    this.isAnalyzing = true;
    this.emit('analysis_started', { uri: imageUri });
    
    try {
      const startTime = Date.now();
      
      // Prepare image for upload
      const base64Image = await this.imageToBase64(imageUri);
      if (!base64Image) {
        throw new Error('Failed to convert image to base64');
      }
      
      // Send to server for analysis
      const response = await fetch(
        `${this.config.serverUrl}/api/vision/analyze`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Image,
            prompt: prompt || 'Describe this image in detail',
            include_objects: true,
            include_text: true,
            include_faces: true,
            include_colors: true,
          }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }
      
      const data = await response.json();
      const processingTime = Date.now() - startTime;
      
      const result: VisionAnalysisResult = {
        description: data.description || data.response || '',
        objects: data.objects || [],
        text: data.text,
        faces: data.faces,
        colors: data.colors,
        tags: data.tags || [],
        confidence: data.confidence || 0.9,
        processingTime,
      };
      
      this.emit('analysis_complete', result);
      return result;
    } catch (error) {
      this.emit('error', { error: `Analysis failed: ${error}` });
      return null;
    } finally {
      this.isAnalyzing = false;
    }
  }
  
  /**
   * Ask a question about an image.
   */
  public async askAboutImage(
    imageUri: string,
    question: string
  ): Promise<string | null> {
    if (!this.config.serverUrl) {
      return null;
    }
    
    try {
      const base64Image = await this.imageToBase64(imageUri);
      
      const response = await fetch(
        `${this.config.serverUrl}/api/vision/ask`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Image,
            question,
          }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`Vision ask failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data.answer || data.response || null;
    } catch (error) {
      this.emit('error', { error: `Vision ask failed: ${error}` });
      return null;
    }
  }
  
  // ===========================================================================
  // OCR / Text Extraction
  // ===========================================================================
  
  /**
   * Extract text from an image using OCR.
   */
  public async extractText(imageUri: string): Promise<OCRResult | null> {
    if (!this.config.serverUrl) {
      return null;
    }
    
    try {
      const base64Image = await this.imageToBase64(imageUri);
      
      const response = await fetch(
        `${this.config.serverUrl}/api/vision/ocr`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Image }),
        }
      );
      
      if (!response.ok) {
        throw new Error(`OCR failed: ${response.status}`);
      }
      
      const data = await response.json();
      return {
        fullText: data.text || '',
        blocks: data.blocks || [],
        language: data.language,
      };
    } catch (error) {
      this.emit('error', { error: `OCR failed: ${error}` });
      return null;
    }
  }
  
  // ===========================================================================
  // QR/Barcode Scanning
  // ===========================================================================
  
  /**
   * Scan QR code or barcode from image.
   */
  public async scanCode(imageUri: string): Promise<ScanResult | null> {
    try {
      // In a real implementation, this would use a local scanning library
      // like react-native-camera's barcode scanner or expo-barcode-scanner
      
      // For server-side scanning:
      if (this.config.serverUrl) {
        const base64Image = await this.imageToBase64(imageUri);
        
        const response = await fetch(
          `${this.config.serverUrl}/api/vision/scan`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image }),
          }
        );
        
        if (!response.ok) {
          return null;
        }
        
        const data = await response.json();
        if (data.found) {
          const result: ScanResult = {
            type: data.type || 'qr',
            data: data.data,
            format: data.format,
            timestamp: Date.now(),
          };
          
          this.emit('scan_detected', result);
          return result;
        }
      }
      
      return null;
    } catch (error) {
      this.emit('error', { error: `Scan failed: ${error}` });
      return null;
    }
  }
  
  /**
   * Start continuous QR/barcode scanning.
   */
  public async startScanning(
    onScan: (result: ScanResult) => void
  ): Promise<() => void> {
    // In a real implementation, this would start the camera in scanning mode
    // and call the callback when codes are detected
    
    console.log('Started QR/barcode scanning...');
    
    // Return stop function
    return () => {
      console.log('Stopped scanning');
    };
  }
  
  // ===========================================================================
  // Screenshot Capture
  // ===========================================================================
  
  /**
   * Capture screenshot and analyze.
   */
  public async captureScreenshot(): Promise<CapturedImage | null> {
    try {
      // In a real implementation, this would use a screenshot library
      // like react-native-view-shot
      
      const { width, height } = Dimensions.get('screen');
      
      const screenshot: CapturedImage = {
        uri: `file://screenshot_${Date.now()}.png`,
        width,
        height,
        timestamp: Date.now(),
      };
      
      this.addToHistory(screenshot);
      return screenshot;
    } catch (error) {
      this.emit('error', { error: `Screenshot failed: ${error}` });
      return null;
    }
  }
  
  // ===========================================================================
  // Image Gallery
  // ===========================================================================
  
  /**
   * Pick image from gallery.
   */
  public async pickFromGallery(): Promise<CapturedImage | null> {
    try {
      // In a real implementation, this would use image picker
      // like react-native-image-picker or expo-image-picker
      
      console.log('Opening image picker...');
      
      // Mock result
      return {
        uri: `file://gallery_${Date.now()}.jpg`,
        width: 1920,
        height: 1080,
        timestamp: Date.now(),
      };
    } catch (error) {
      this.emit('error', { error: `Gallery pick failed: ${error}` });
      return null;
    }
  }
  
  /**
   * Pick multiple images from gallery.
   */
  public async pickMultipleFromGallery(
    maxImages: number = 10
  ): Promise<CapturedImage[]> {
    try {
      console.log(`Opening multi-image picker (max: ${maxImages})...`);
      return [];
    } catch (error) {
      this.emit('error', { error: `Multi-pick failed: ${error}` });
      return [];
    }
  }
  
  // ===========================================================================
  // History Management
  // ===========================================================================
  
  /**
   * Get capture history.
   */
  public getHistory(): CapturedImage[] {
    return [...this.captureHistory];
  }
  
  /**
   * Clear capture history.
   */
  public clearHistory(): void {
    this.captureHistory = [];
  }
  
  private addToHistory(image: CapturedImage): void {
    this.captureHistory.unshift(image);
    
    // Limit history size
    if (this.captureHistory.length > this.maxHistorySize) {
      this.captureHistory = this.captureHistory.slice(0, this.maxHistorySize);
    }
  }
  
  // ===========================================================================
  // Utility Methods
  // ===========================================================================
  
  /**
   * Convert image URI to base64.
   */
  private async imageToBase64(uri: string): Promise<string | null> {
    try {
      // In a real implementation, this would read the file and convert to base64
      // using react-native-fs or expo-file-system
      
      // Mock implementation
      if (uri.startsWith('data:')) {
        return uri.split(',')[1];
      }
      
      // Placeholder - actual implementation would read the file
      return `base64_encoded_image_${Date.now()}`;
    } catch (error) {
      console.error('Base64 conversion failed:', error);
      return null;
    }
  }
  
  /**
   * Resize image for upload.
   */
  private async resizeImage(
    uri: string,
    maxDimension: number
  ): Promise<string | null> {
    try {
      // In a real implementation, this would use image manipulation
      // like react-native-image-resizer
      
      console.log(`Resizing image to max ${maxDimension}px...`);
      return uri; // Return original for now
    } catch (error) {
      console.error('Image resize failed:', error);
      return null;
    }
  }
  
  /**
   * Request camera permission.
   */
  private async requestCameraPermission(): Promise<boolean> {
    try {
      // In a real implementation, this would use permission library
      // like react-native-permissions or expo-permissions
      
      console.log('Requesting camera permission...');
      return true;
    } catch (error) {
      console.error('Permission error:', error);
      return false;
    }
  }
  
  // ===========================================================================
  // State Getters
  // ===========================================================================
  
  public getIsCapturing(): boolean {
    return this.isCapturing;
  }
  
  public getIsAnalyzing(): boolean {
    return this.isAnalyzing;
  }
  
  public getCurrentCamera(): 'front' | 'back' {
    return this.currentCamera;
  }
  
  public getIsFlashEnabled(): boolean {
    return this.config.flashEnabled;
  }
}

// Export singleton instance getter
export const getCameraService = (): CameraService => CameraService.getInstance();

export {
  CameraService,
  CameraConfig,
  CapturedImage,
  ImageMetadata,
  VisionAnalysisResult,
  DetectedObject,
  BoundingBox,
  DetectedFace,
  OCRResult,
  TextBlock,
  ColorInfo,
  ScanResult,
  CameraEventType,
};

export default CameraService;
