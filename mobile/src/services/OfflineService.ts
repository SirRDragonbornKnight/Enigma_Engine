/**
 * Offline Mode Service
 * 
 * Enables offline usage by caching conversations and queuing
 * messages for later sync when online.
 * 
 * Features:
 * - Cache last N conversations locally
 * - Queue messages when offline, replay when online
 * - Store common responses for offline use
 * - Background sync when WiFi available
 * 
 * FILE: mobile/src/services/OfflineService.ts
 * TYPE: Mobile Service
 */

import * as FileSystem from 'expo-file-system';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';

// Types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  synced: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  synced: boolean;
}

export interface QueuedRequest {
  id: string;
  type: 'message' | 'sync' | 'training';
  payload: any;
  createdAt: number;
  retryCount: number;
  maxRetries: number;
}

export interface CachedResponse {
  prompt: string;
  promptHash: string;
  response: string;
  timestamp: number;
  useCount: number;
}

export interface OfflineConfig {
  maxCachedConversations: number;
  maxCachedResponses: number;
  maxQueuedRequests: number;
  syncIntervalMs: number;
  wifiOnlySync: boolean;
  autoCache: boolean;
}

const DEFAULT_CONFIG: OfflineConfig = {
  maxCachedConversations: 50,
  maxCachedResponses: 200,
  maxQueuedRequests: 100,
  syncIntervalMs: 60000, // 1 minute
  wifiOnlySync: false,
  autoCache: true,
};

const STORAGE_KEYS = {
  CONVERSATIONS: '@enigma/conversations',
  QUEUE: '@enigma/offline_queue',
  CACHE: '@enigma/response_cache',
  CONFIG: '@enigma/offline_config',
  LAST_SYNC: '@enigma/last_sync',
};

// Simple hash function for prompts
function hashString(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString(36);
}

// Generate unique ID
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}


export class OfflineService {
  private config: OfflineConfig;
  private conversations: Map<string, Conversation> = new Map();
  private requestQueue: QueuedRequest[] = [];
  private responseCache: Map<string, CachedResponse> = new Map();
  private isOnline: boolean = true;
  private syncTimer: NodeJS.Timeout | null = null;
  private syncInProgress: boolean = false;
  private serverUrl: string = '';
  
  // Callbacks
  public onOnlineStatusChange?: (isOnline: boolean) => void;
  public onSyncComplete?: (success: boolean, synced: number) => void;
  public onQueueChange?: (queueSize: number) => void;

  constructor(serverUrl: string, config: Partial<OfflineConfig> = {}) {
    this.serverUrl = serverUrl;
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // =========================================================================
  // Initialization
  // =========================================================================

  async initialize(): Promise<void> {
    // Load saved data
    await this.loadFromStorage();
    
    // Set up network listener
    this.setupNetworkListener();
    
    // Start sync timer
    this.startSyncTimer();
    
    console.log('OfflineService initialized');
  }

  private async loadFromStorage(): Promise<void> {
    try {
      // Load config
      const configStr = await AsyncStorage.getItem(STORAGE_KEYS.CONFIG);
      if (configStr) {
        this.config = { ...this.config, ...JSON.parse(configStr) };
      }

      // Load conversations
      const convStr = await AsyncStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
      if (convStr) {
        const convArray: Conversation[] = JSON.parse(convStr);
        convArray.forEach(conv => this.conversations.set(conv.id, conv));
      }

      // Load queue
      const queueStr = await AsyncStorage.getItem(STORAGE_KEYS.QUEUE);
      if (queueStr) {
        this.requestQueue = JSON.parse(queueStr);
      }

      // Load response cache
      const cacheStr = await AsyncStorage.getItem(STORAGE_KEYS.CACHE);
      if (cacheStr) {
        const cacheArray: CachedResponse[] = JSON.parse(cacheStr);
        cacheArray.forEach(item => this.responseCache.set(item.promptHash, item));
      }

      console.log(`Loaded ${this.conversations.size} conversations, ${this.requestQueue.length} queued requests, ${this.responseCache.size} cached responses`);
    } catch (error) {
      console.error('Failed to load offline data:', error);
    }
  }

  private async saveToStorage(): Promise<void> {
    try {
      // Save conversations
      const convArray = Array.from(this.conversations.values());
      await AsyncStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(convArray));

      // Save queue
      await AsyncStorage.setItem(STORAGE_KEYS.QUEUE, JSON.stringify(this.requestQueue));

      // Save cache
      const cacheArray = Array.from(this.responseCache.values());
      await AsyncStorage.setItem(STORAGE_KEYS.CACHE, JSON.stringify(cacheArray));

      // Save config
      await AsyncStorage.setItem(STORAGE_KEYS.CONFIG, JSON.stringify(this.config));
    } catch (error) {
      console.error('Failed to save offline data:', error);
    }
  }

  private setupNetworkListener(): void {
    NetInfo.addEventListener(state => {
      const wasOnline = this.isOnline;
      this.isOnline = state.isConnected ?? false;
      
      // Check WiFi for sync
      const isWifi = state.type === 'wifi';
      
      if (this.isOnline !== wasOnline) {
        console.log(`Network status changed: ${this.isOnline ? 'online' : 'offline'}`);
        
        if (this.onOnlineStatusChange) {
          this.onOnlineStatusChange(this.isOnline);
        }
        
        // Sync when coming back online
        if (this.isOnline && (!this.config.wifiOnlySync || isWifi)) {
          this.syncWithServer();
        }
      }
    });
  }

  private startSyncTimer(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
    }
    
    this.syncTimer = setInterval(() => {
      if (this.isOnline && !this.syncInProgress) {
        this.syncWithServer();
      }
    }, this.config.syncIntervalMs);
  }

  // =========================================================================
  // Conversation Management
  // =========================================================================

  createConversation(title: string = 'New Chat'): Conversation {
    const conversation: Conversation = {
      id: generateId(),
      title,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      synced: false,
    };
    
    this.conversations.set(conversation.id, conversation);
    this.pruneConversations();
    this.saveToStorage();
    
    return conversation;
  }

  getConversation(id: string): Conversation | undefined {
    return this.conversations.get(id);
  }

  getAllConversations(): Conversation[] {
    return Array.from(this.conversations.values())
      .sort((a, b) => b.updatedAt - a.updatedAt);
  }

  addMessage(conversationId: string, role: Message['role'], content: string): Message {
    const conversation = this.conversations.get(conversationId);
    if (!conversation) {
      throw new Error(`Conversation not found: ${conversationId}`);
    }

    const message: Message = {
      id: generateId(),
      role,
      content,
      timestamp: Date.now(),
      synced: false,
    };

    conversation.messages.push(message);
    conversation.updatedAt = Date.now();
    conversation.synced = false;

    // Update title from first user message if default
    if (conversation.title === 'New Chat' && role === 'user') {
      conversation.title = content.slice(0, 50) + (content.length > 50 ? '...' : '');
    }

    this.saveToStorage();
    return message;
  }

  deleteConversation(id: string): boolean {
    const deleted = this.conversations.delete(id);
    if (deleted) {
      this.saveToStorage();
    }
    return deleted;
  }

  private pruneConversations(): void {
    // Remove oldest conversations if over limit
    const convArray = this.getAllConversations();
    if (convArray.length > this.config.maxCachedConversations) {
      const toRemove = convArray.slice(this.config.maxCachedConversations);
      toRemove.forEach(conv => this.conversations.delete(conv.id));
    }
  }

  // =========================================================================
  // Request Queue
  // =========================================================================

  queueRequest(type: QueuedRequest['type'], payload: any): QueuedRequest {
    const request: QueuedRequest = {
      id: generateId(),
      type,
      payload,
      createdAt: Date.now(),
      retryCount: 0,
      maxRetries: 3,
    };

    this.requestQueue.push(request);
    this.pruneQueue();
    this.saveToStorage();

    if (this.onQueueChange) {
      this.onQueueChange(this.requestQueue.length);
    }

    return request;
  }

  getQueuedRequests(): QueuedRequest[] {
    return [...this.requestQueue];
  }

  removeFromQueue(id: string): boolean {
    const index = this.requestQueue.findIndex(r => r.id === id);
    if (index >= 0) {
      this.requestQueue.splice(index, 1);
      this.saveToStorage();
      
      if (this.onQueueChange) {
        this.onQueueChange(this.requestQueue.length);
      }
      return true;
    }
    return false;
  }

  private pruneQueue(): void {
    // Remove oldest requests if over limit
    if (this.requestQueue.length > this.config.maxQueuedRequests) {
      this.requestQueue = this.requestQueue.slice(-this.config.maxQueuedRequests);
    }
  }

  // =========================================================================
  // Response Cache
  // =========================================================================

  cacheResponse(prompt: string, response: string): void {
    const promptHash = hashString(prompt.toLowerCase().trim());
    
    this.responseCache.set(promptHash, {
      prompt,
      promptHash,
      response,
      timestamp: Date.now(),
      useCount: 0,
    });

    this.pruneCache();
    this.saveToStorage();
  }

  getCachedResponse(prompt: string): string | null {
    const promptHash = hashString(prompt.toLowerCase().trim());
    const cached = this.responseCache.get(promptHash);
    
    if (cached) {
      cached.useCount++;
      return cached.response;
    }
    
    // Try fuzzy match for similar prompts
    return this.fuzzyMatchCache(prompt);
  }

  private fuzzyMatchCache(prompt: string): string | null {
    const normalizedPrompt = prompt.toLowerCase().trim();
    
    // Simple prefix matching for now
    for (const [hash, cached] of this.responseCache) {
      const normalizedCached = cached.prompt.toLowerCase().trim();
      if (normalizedPrompt.startsWith(normalizedCached) || 
          normalizedCached.startsWith(normalizedPrompt)) {
        cached.useCount++;
        return cached.response;
      }
    }
    
    return null;
  }

  private pruneCache(): void {
    // Remove least used responses if over limit
    if (this.responseCache.size > this.config.maxCachedResponses) {
      const entries = Array.from(this.responseCache.entries())
        .sort((a, b) => a[1].useCount - b[1].useCount);
      
      const toRemove = entries.slice(0, this.responseCache.size - this.config.maxCachedResponses);
      toRemove.forEach(([hash]) => this.responseCache.delete(hash));
    }
  }

  clearCache(): void {
    this.responseCache.clear();
    this.saveToStorage();
  }

  // =========================================================================
  // Chat Interface (Online/Offline aware)
  // =========================================================================

  async sendMessage(
    conversationId: string,
    message: string,
    options: { preferCache?: boolean; forceOnline?: boolean } = {}
  ): Promise<{ response: string; fromCache: boolean; queued: boolean }> {
    // Add user message to conversation
    this.addMessage(conversationId, 'user', message);

    // Try cache first if preferred
    if (options.preferCache !== false) {
      const cached = this.getCachedResponse(message);
      if (cached) {
        this.addMessage(conversationId, 'assistant', cached);
        return { response: cached, fromCache: true, queued: false };
      }
    }

    // If offline, queue the request
    if (!this.isOnline && !options.forceOnline) {
      this.queueRequest('message', {
        conversationId,
        message,
      });
      
      // Return placeholder response
      const placeholderResponse = "I'm currently offline. Your message has been saved and I'll respond when we're back online.";
      this.addMessage(conversationId, 'assistant', placeholderResponse);
      
      return { response: placeholderResponse, fromCache: false, queued: true };
    }

    // Send to server
    try {
      const response = await this.sendToServer(message, conversationId);
      
      // Cache successful response
      if (this.config.autoCache) {
        this.cacheResponse(message, response);
      }
      
      this.addMessage(conversationId, 'assistant', response);
      return { response, fromCache: false, queued: false };
      
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Queue for retry
      this.queueRequest('message', {
        conversationId,
        message,
      });
      
      // Try cache as fallback
      const cached = this.getCachedResponse(message);
      if (cached) {
        this.addMessage(conversationId, 'assistant', cached);
        return { response: cached, fromCache: true, queued: true };
      }
      
      const errorResponse = "Sorry, I couldn't process your request. It's been queued for retry.";
      this.addMessage(conversationId, 'assistant', errorResponse);
      return { response: errorResponse, fromCache: false, queued: true };
    }
  }

  private async sendToServer(message: string, conversationId: string): Promise<string> {
    const conversation = this.conversations.get(conversationId);
    const history = conversation?.messages.slice(-10) ?? [];

    const response = await fetch(`${this.serverUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: message,
        history: history.map(m => ({ role: m.role, content: m.content })),
        max_tokens: 200,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    return data.response ?? data.text ?? '';
  }

  // =========================================================================
  // Sync
  // =========================================================================

  async syncWithServer(): Promise<{ success: boolean; syncedCount: number }> {
    if (this.syncInProgress) {
      return { success: false, syncedCount: 0 };
    }

    if (!this.isOnline) {
      return { success: false, syncedCount: 0 };
    }

    this.syncInProgress = true;
    let syncedCount = 0;

    try {
      // Process queued requests
      const toProcess = [...this.requestQueue];
      
      for (const request of toProcess) {
        try {
          if (request.type === 'message') {
            const { conversationId, message } = request.payload;
            const response = await this.sendToServer(message, conversationId);
            
            // Update conversation with real response
            const conversation = this.conversations.get(conversationId);
            if (conversation) {
              // Find and update placeholder message
              const placeholderIdx = conversation.messages.findIndex(
                m => m.role === 'assistant' && m.content.includes('currently offline')
              );
              if (placeholderIdx >= 0) {
                conversation.messages[placeholderIdx].content = response;
              } else {
                this.addMessage(conversationId, 'assistant', response);
              }
              
              // Cache the response
              if (this.config.autoCache) {
                this.cacheResponse(message, response);
              }
            }
            
            this.removeFromQueue(request.id);
            syncedCount++;
            
          } else if (request.type === 'sync') {
            // Handle other sync types
            this.removeFromQueue(request.id);
            syncedCount++;
          }
          
        } catch (error) {
          console.error(`Failed to sync request ${request.id}:`, error);
          request.retryCount++;
          
          if (request.retryCount >= request.maxRetries) {
            this.removeFromQueue(request.id);
          }
        }
      }

      // Sync unsynced conversations
      for (const [id, conv] of this.conversations) {
        if (!conv.synced) {
          // Mark as synced (in production, would upload to server)
          conv.synced = true;
          conv.messages.forEach(m => m.synced = true);
        }
      }

      // Save updated state
      await this.saveToStorage();
      await AsyncStorage.setItem(STORAGE_KEYS.LAST_SYNC, new Date().toISOString());

      if (this.onSyncComplete) {
        this.onSyncComplete(true, syncedCount);
      }

      return { success: true, syncedCount };

    } catch (error) {
      console.error('Sync failed:', error);
      
      if (this.onSyncComplete) {
        this.onSyncComplete(false, syncedCount);
      }
      
      return { success: false, syncedCount };
      
    } finally {
      this.syncInProgress = false;
    }
  }

  // =========================================================================
  // Status & Config
  // =========================================================================

  get online(): boolean {
    return this.isOnline;
  }

  get queueSize(): number {
    return this.requestQueue.length;
  }

  get cacheSize(): number {
    return this.responseCache.size;
  }

  get conversationCount(): number {
    return this.conversations.size;
  }

  async getLastSyncTime(): Promise<string | null> {
    return AsyncStorage.getItem(STORAGE_KEYS.LAST_SYNC);
  }

  updateConfig(config: Partial<OfflineConfig>): void {
    this.config = { ...this.config, ...config };
    this.saveToStorage();
    
    // Restart sync timer with new interval
    if (config.syncIntervalMs) {
      this.startSyncTimer();
    }
  }

  getConfig(): OfflineConfig {
    return { ...this.config };
  }

  // =========================================================================
  // Cleanup
  // =========================================================================

  destroy(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = null;
    }
  }

  async clearAllData(): Promise<void> {
    this.conversations.clear();
    this.requestQueue = [];
    this.responseCache.clear();
    
    await AsyncStorage.multiRemove([
      STORAGE_KEYS.CONVERSATIONS,
      STORAGE_KEYS.QUEUE,
      STORAGE_KEYS.CACHE,
      STORAGE_KEYS.LAST_SYNC,
    ]);
  }
}


// Singleton instance
let offlineServiceInstance: OfflineService | null = null;

export function getOfflineService(serverUrl?: string): OfflineService {
  if (!offlineServiceInstance) {
    offlineServiceInstance = new OfflineService(serverUrl ?? 'http://localhost:8080');
  }
  return offlineServiceInstance;
}

export function initializeOfflineService(serverUrl: string, config?: Partial<OfflineConfig>): OfflineService {
  offlineServiceInstance = new OfflineService(serverUrl, config);
  return offlineServiceInstance;
}
