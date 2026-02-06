/**
 * Local Model Runner
 * 
 * Run small AI models directly on mobile devices.
 * Uses ONNX Runtime or TensorFlow Lite for inference.
 * 
 * FILE: mobile/src/services/LocalModelService.ts
 * TYPE: Mobile Service
 */

import * as FileSystem from 'expo-file-system';

// Types
export interface ModelConfig {
  name: string;
  path: string;
  vocabPath: string;
  maxTokens: number;
  contextLength: number;
  quantization: 'int4' | 'int8' | 'fp16' | 'fp32';
}

export interface GenerationOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export interface TokenizerConfig {
  vocabSize: number;
  bosToken: string;
  eosToken: string;
  padToken: string;
  unkToken: string;
}

// Simple BPE Tokenizer
export class SimpleTokenizer {
  private vocab: Map<string, number> = new Map();
  private inverseVocab: Map<number, string> = new Map();
  private merges: Map<string, string> = new Map();
  private config: TokenizerConfig;

  constructor(config: Partial<TokenizerConfig> = {}) {
    this.config = {
      vocabSize: 32000,
      bosToken: '<s>',
      eosToken: '</s>',
      padToken: '<pad>',
      unkToken: '<unk>',
      ...config,
    };
  }

  async load(vocabPath: string): Promise<void> {
    try {
      const content = await FileSystem.readAsStringAsync(vocabPath);
      const data = JSON.parse(content);
      
      // Load vocab
      if (data.vocab) {
        Object.entries(data.vocab).forEach(([token, id]) => {
          this.vocab.set(token, id as number);
          this.inverseVocab.set(id as number, token);
        });
      }
      
      // Load merges
      if (data.merges) {
        data.merges.forEach((merge: string) => {
          const [a, b] = merge.split(' ');
          this.merges.set(`${a} ${b}`, `${a}${b}`);
        });
      }
    } catch (error) {
      console.error('Failed to load tokenizer:', error);
      this.initBasicVocab();
    }
  }

  private initBasicVocab(): void {
    // Basic ASCII vocab fallback
    const specialTokens = [this.config.padToken, this.config.unkToken, this.config.bosToken, this.config.eosToken];
    specialTokens.forEach((token, i) => {
      this.vocab.set(token, i);
      this.inverseVocab.set(i, token);
    });

    // ASCII characters
    for (let i = 0; i < 256; i++) {
      const char = String.fromCharCode(i);
      const id = i + specialTokens.length;
      this.vocab.set(char, id);
      this.inverseVocab.set(id, char);
    }
  }

  encode(text: string): number[] {
    const tokens: number[] = [this.vocab.get(this.config.bosToken) ?? 2];
    
    // Character-level fallback
    for (const char of text) {
      const id = this.vocab.get(char) ?? this.vocab.get(this.config.unkToken) ?? 3;
      tokens.push(id);
    }
    
    return tokens;
  }

  decode(tokens: number[]): string {
    return tokens
      .map(id => this.inverseVocab.get(id) ?? '')
      .filter(t => t !== this.config.bosToken && t !== this.config.eosToken && t !== this.config.padToken)
      .join('');
  }

  get vocabSize(): number {
    return this.vocab.size;
  }
}

// Model Runner Interface
export interface ModelRunner {
  load(path: string): Promise<void>;
  generate(inputIds: number[], options: GenerationOptions): Promise<number[]>;
  unload(): void;
}

// ONNX Runtime Runner (when native module available)
export class ONNXModelRunner implements ModelRunner {
  private session: any = null;
  private isLoaded: boolean = false;

  async load(path: string): Promise<void> {
    try {
      // Check if file exists
      const info = await FileSystem.getInfoAsync(path);
      if (!info.exists) {
        throw new Error(`Model file not found: ${path}`);
      }

      // In production, use react-native-onnxruntime
      // this.session = await InferenceSession.create(path);
      this.isLoaded = true;
      console.log('ONNX model loaded (simulation)');
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  async generate(inputIds: number[], options: GenerationOptions = {}): Promise<number[]> {
    if (!this.isLoaded) {
      throw new Error('Model not loaded');
    }

    const maxTokens = options.maxTokens ?? 50;
    const temperature = options.temperature ?? 1.0;
    const outputIds = [...inputIds];

    // Simulated generation (in production, run actual inference)
    for (let i = 0; i < maxTokens; i++) {
      // In production:
      // const tensor = new Tensor('int64', inputIds, [1, inputIds.length]);
      // const result = await this.session.run({ input_ids: tensor });
      // const logits = result.logits.data;
      // const nextToken = this.sampleToken(logits, temperature);
      
      // Simulation: generate random tokens
      const nextToken = Math.floor(Math.random() * 1000) + 100;
      outputIds.push(nextToken);

      // Check for EOS
      if (nextToken === 1) break; // Assuming 1 is EOS
    }

    return outputIds;
  }

  private sampleToken(logits: Float32Array, temperature: number): number {
    // Apply temperature
    const scaled = new Float32Array(logits.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      scaled[i] = logits[i] / temperature;
      maxLogit = Math.max(maxLogit, scaled[i]);
    }

    // Softmax
    let sum = 0;
    for (let i = 0; i < scaled.length; i++) {
      scaled[i] = Math.exp(scaled[i] - maxLogit);
      sum += scaled[i];
    }
    for (let i = 0; i < scaled.length; i++) {
      scaled[i] /= sum;
    }

    // Sample
    const rand = Math.random();
    let cumSum = 0;
    for (let i = 0; i < scaled.length; i++) {
      cumSum += scaled[i];
      if (rand < cumSum) return i;
    }

    return scaled.length - 1;
  }

  unload(): void {
    this.session = null;
    this.isLoaded = false;
  }
}

// TensorFlow Lite Runner
export class TFLiteModelRunner implements ModelRunner {
  private interpreter: any = null;
  private isLoaded: boolean = false;

  async load(path: string): Promise<void> {
    try {
      const info = await FileSystem.getInfoAsync(path);
      if (!info.exists) {
        throw new Error(`Model file not found: ${path}`);
      }

      // In production, use react-native-tflite
      // this.interpreter = await TFLite.loadModel(path);
      this.isLoaded = true;
      console.log('TFLite model loaded (simulation)');
    } catch (error) {
      console.error('Failed to load TFLite model:', error);
      throw error;
    }
  }

  async generate(inputIds: number[], options: GenerationOptions = {}): Promise<number[]> {
    if (!this.isLoaded) {
      throw new Error('Model not loaded');
    }

    const maxTokens = options.maxTokens ?? 50;
    const outputIds = [...inputIds];

    for (let i = 0; i < maxTokens; i++) {
      // Simulation
      const nextToken = Math.floor(Math.random() * 1000) + 100;
      outputIds.push(nextToken);
      if (nextToken === 1) break;
    }

    return outputIds;
  }

  unload(): void {
    this.interpreter = null;
    this.isLoaded = false;
  }
}

// Local Model Service
export class LocalModelService {
  private tokenizer: SimpleTokenizer;
  private modelRunner: ModelRunner | null = null;
  private config: ModelConfig | null = null;
  private isLoaded: boolean = false;

  constructor() {
    this.tokenizer = new SimpleTokenizer();
  }

  async loadModel(config: ModelConfig): Promise<void> {
    this.config = config;

    // Load tokenizer
    await this.tokenizer.load(config.vocabPath);

    // Select model runner based on file extension
    if (config.path.endsWith('.onnx')) {
      this.modelRunner = new ONNXModelRunner();
    } else if (config.path.endsWith('.tflite')) {
      this.modelRunner = new TFLiteModelRunner();
    } else {
      throw new Error('Unsupported model format. Use .onnx or .tflite');
    }

    await this.modelRunner.load(config.path);
    this.isLoaded = true;
  }

  async generate(prompt: string, options: GenerationOptions = {}): Promise<string> {
    if (!this.isLoaded || !this.modelRunner) {
      throw new Error('Model not loaded');
    }

    // Encode prompt
    const inputIds = this.tokenizer.encode(prompt);

    // Truncate if needed
    const maxContext = this.config?.contextLength ?? 512;
    const truncatedIds = inputIds.slice(-maxContext);

    // Generate
    const outputIds = await this.modelRunner.generate(truncatedIds, {
      ...options,
      maxTokens: options.maxTokens ?? this.config?.maxTokens ?? 50,
    });

    // Decode
    const newTokens = outputIds.slice(truncatedIds.length);
    return this.tokenizer.decode(newTokens);
  }

  async chat(messages: { role: string; content: string }[], options: GenerationOptions = {}): Promise<string> {
    // Format messages as prompt
    const prompt = messages
      .map(m => {
        if (m.role === 'system') return `System: ${m.content}`;
        if (m.role === 'user') return `User: ${m.content}`;
        return `Assistant: ${m.content}`;
      })
      .join('\n') + '\nAssistant:';

    return this.generate(prompt, options);
  }

  unload(): void {
    this.modelRunner?.unload();
    this.modelRunner = null;
    this.config = null;
    this.isLoaded = false;
  }

  get loaded(): boolean {
    return this.isLoaded;
  }

  get modelInfo(): ModelConfig | null {
    return this.config;
  }
}

// Model Registry
export interface AvailableModel {
  id: string;
  name: string;
  description: string;
  size: number; // bytes
  quantization: 'int4' | 'int8' | 'fp16';
  downloadUrl: string;
  vocabUrl: string;
}

export const AVAILABLE_MODELS: AvailableModel[] = [
  {
    id: 'forge-nano',
    name: 'Forge Nano',
    description: 'Tiny model for basic tasks (~5MB)',
    size: 5 * 1024 * 1024,
    quantization: 'int4',
    downloadUrl: 'https://models.forgeai.dev/forge-nano.onnx',
    vocabUrl: 'https://models.forgeai.dev/forge-nano-vocab.json',
  },
  {
    id: 'forge-micro',
    name: 'Forge Micro',
    description: 'Small model for everyday use (~15MB)',
    size: 15 * 1024 * 1024,
    quantization: 'int4',
    downloadUrl: 'https://models.forgeai.dev/forge-micro.onnx',
    vocabUrl: 'https://models.forgeai.dev/forge-micro-vocab.json',
  },
  {
    id: 'forge-tiny',
    name: 'Forge Tiny',
    description: 'Capable model for complex tasks (~50MB)',
    size: 50 * 1024 * 1024,
    quantization: 'int8',
    downloadUrl: 'https://models.forgeai.dev/forge-tiny.onnx',
    vocabUrl: 'https://models.forgeai.dev/forge-tiny-vocab.json',
  },
];

// Model Downloader
export async function downloadModel(
  model: AvailableModel,
  progressCallback?: (progress: number) => void
): Promise<{ modelPath: string; vocabPath: string }> {
  const modelDir = `${FileSystem.documentDirectory}models/`;
  
  // Ensure directory exists
  const dirInfo = await FileSystem.getInfoAsync(modelDir);
  if (!dirInfo.exists) {
    await FileSystem.makeDirectoryAsync(modelDir, { intermediates: true });
  }

  const modelPath = `${modelDir}${model.id}.onnx`;
  const vocabPath = `${modelDir}${model.id}-vocab.json`;

  // Download model
  const modelDownload = FileSystem.createDownloadResumable(
    model.downloadUrl,
    modelPath,
    {},
    (downloadProgress) => {
      const progress = downloadProgress.totalBytesWritten / downloadProgress.totalBytesExpectedToWrite;
      progressCallback?.(progress * 0.9); // 90% for model
    }
  );

  await modelDownload.downloadAsync();

  // Download vocab
  const vocabDownload = FileSystem.createDownloadResumable(
    model.vocabUrl,
    vocabPath,
    {},
    () => {
      progressCallback?.(0.95); // 95% for vocab
    }
  );

  await vocabDownload.downloadAsync();
  progressCallback?.(1.0);

  return { modelPath, vocabPath };
}

// Check available disk space
export async function checkDiskSpace(): Promise<number> {
  try {
    const info = await FileSystem.getFreeDiskStorageAsync();
    return info;
  } catch {
    return 0;
  }
}

// List downloaded models
export async function listDownloadedModels(): Promise<string[]> {
  const modelDir = `${FileSystem.documentDirectory}models/`;
  
  try {
    const files = await FileSystem.readDirectoryAsync(modelDir);
    return files.filter(f => f.endsWith('.onnx') || f.endsWith('.tflite'));
  } catch {
    return [];
  }
}

// Delete model
export async function deleteModel(modelId: string): Promise<void> {
  const modelDir = `${FileSystem.documentDirectory}models/`;
  
  await FileSystem.deleteAsync(`${modelDir}${modelId}.onnx`, { idempotent: true });
  await FileSystem.deleteAsync(`${modelDir}${modelId}.tflite`, { idempotent: true });
  await FileSystem.deleteAsync(`${modelDir}${modelId}-vocab.json`, { idempotent: true });
}

// Default export
export default LocalModelService;
