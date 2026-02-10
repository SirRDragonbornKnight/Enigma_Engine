/**
 * Multi-Server Connection Service for Enigma AI Mobile
 * 
 * Allows connecting to multiple Enigma instances (PCs) from mobile:
 * - Save multiple server configurations
 * - Switch between servers
 * - Automatic server discovery on local network
 * - Health checking and auto-reconnect
 * - QR code pairing support
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

// Types
interface ServerConfig {
  id: string;
  name: string;
  host: string;
  port: number;
  useTLS: boolean;
  authToken?: string;
  lastConnected?: number;
  addedAt: number;
  capabilities?: ServerCapabilities;
  metadata?: ServerMetadata;
}

interface ServerCapabilities {
  hasImageGen: boolean;
  hasCodeGen: boolean;
  hasVideoGen: boolean;
  hasAudioGen: boolean;
  has3DGen: boolean;
  hasVision: boolean;
  hasVoice: boolean;
  hasAvatar: boolean;
  hasTraining: boolean;
  modelSizes: string[];
  loadedModels: string[];
}

interface ServerMetadata {
  version?: string;
  hostname?: string;
  platform?: string;
  gpuName?: string;
  gpuMemory?: number;
  cpuCount?: number;
  totalMemory?: number;
}

interface ServerStatus {
  online: boolean;
  latency: number;
  health: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  lastChecked: number;
  error?: string;
  load?: number;
  activeConnections?: number;
}

interface DiscoveredServer {
  host: string;
  port: number;
  name: string;
  version?: string;
  capabilities?: string[];
}

type ConnectionEventType =
  | 'server_added'
  | 'server_removed'
  | 'server_updated'
  | 'connected'
  | 'disconnected'
  | 'connection_error'
  | 'server_discovered'
  | 'status_changed';

type ConnectionEventCallback = (event: ConnectionEventType, data?: any) => void;

const STORAGE_KEY = '@enigma_servers';
const DEFAULT_PORT = 8080;
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds
const DISCOVERY_TIMEOUT = 5000; // 5 seconds

/**
 * Multi-Server Connection Service
 * 
 * Example usage:
 * ```typescript
 * const serverManager = ServerManager.getInstance();
 * 
 * // Add a server
 * const server = await serverManager.addServer({
 *   name: 'Home PC',
 *   host: '192.168.1.100',
 *   port: 8080
 * });
 * 
 * // Connect to it
 * await serverManager.connect(server.id);
 * 
 * // Get current connection
 * const current = serverManager.getCurrentServer();
 * ```
 */
class ServerManager {
  private static instance: ServerManager;
  
  private servers: Map<string, ServerConfig> = new Map();
  private serverStatus: Map<string, ServerStatus> = new Map();
  private currentServerId: string | null = null;
  private eventListeners: Map<string, Set<ConnectionEventCallback>> = new Map();
  
  private healthCheckTimer: ReturnType<typeof setInterval> | null = null;
  private isDiscovering: boolean = false;
  
  private constructor() {
    this.loadServers();
  }
  
  public static getInstance(): ServerManager {
    if (!ServerManager.instance) {
      ServerManager.instance = new ServerManager();
    }
    return ServerManager.instance;
  }
  
  // ===========================================================================
  // Event Handling
  // ===========================================================================
  
  public addEventListener(
    event: ConnectionEventType,
    callback: ConnectionEventCallback
  ): () => void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
    
    return () => {
      this.eventListeners.get(event)?.delete(callback);
    };
  }
  
  private emit(event: ConnectionEventType, data?: any): void {
    this.eventListeners.get(event)?.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error(`Connection event callback error: ${error}`);
      }
    });
  }
  
  // ===========================================================================
  // Server Management
  // ===========================================================================
  
  /**
   * Add a new server configuration.
   */
  public async addServer(config: {
    name: string;
    host: string;
    port?: number;
    useTLS?: boolean;
    authToken?: string;
  }): Promise<ServerConfig> {
    const id = this.generateId();
    
    const server: ServerConfig = {
      id,
      name: config.name,
      host: config.host,
      port: config.port || DEFAULT_PORT,
      useTLS: config.useTLS || false,
      authToken: config.authToken,
      addedAt: Date.now(),
    };
    
    // Fetch capabilities and metadata
    try {
      const info = await this.fetchServerInfo(server);
      server.capabilities = info.capabilities;
      server.metadata = info.metadata;
    } catch (error) {
      console.warn('Could not fetch server info:', error);
    }
    
    this.servers.set(id, server);
    await this.saveServers();
    
    this.emit('server_added', server);
    
    // Start health checking if this is the first server
    if (this.servers.size === 1) {
      this.startHealthChecking();
    }
    
    return server;
  }
  
  /**
   * Add server from QR code data.
   */
  public async addServerFromQR(qrData: string): Promise<ServerConfig | null> {
    try {
      // Expected format: enigma://host:port?token=xxx&name=xxx
      // Or JSON: {"host":"...", "port":..., "token":"...", "name":"..."}
      
      let config: any;
      
      if (qrData.startsWith('enigma://')) {
        const url = new URL(qrData.replace('enigma://', 'http://'));
        config = {
          host: url.hostname,
          port: parseInt(url.port) || DEFAULT_PORT,
          authToken: url.searchParams.get('token') || undefined,
          name: url.searchParams.get('name') || url.hostname,
        };
      } else if (qrData.startsWith('{')) {
        config = JSON.parse(qrData);
      } else {
        throw new Error('Invalid QR code format');
      }
      
      if (!config.host) {
        throw new Error('Missing host in QR data');
      }
      
      return await this.addServer(config);
    } catch (error) {
      console.error('Failed to parse QR code:', error);
      return null;
    }
  }
  
  /**
   * Remove a server configuration.
   */
  public async removeServer(serverId: string): Promise<boolean> {
    if (!this.servers.has(serverId)) {
      return false;
    }
    
    // Disconnect if this is the current server
    if (this.currentServerId === serverId) {
      await this.disconnect();
    }
    
    this.servers.delete(serverId);
    this.serverStatus.delete(serverId);
    await this.saveServers();
    
    this.emit('server_removed', { id: serverId });
    
    // Stop health checking if no servers left
    if (this.servers.size === 0) {
      this.stopHealthChecking();
    }
    
    return true;
  }
  
  /**
   * Update server configuration.
   */
  public async updateServer(
    serverId: string,
    updates: Partial<Omit<ServerConfig, 'id' | 'addedAt'>>
  ): Promise<ServerConfig | null> {
    const server = this.servers.get(serverId);
    if (!server) {
      return null;
    }
    
    const updated = { ...server, ...updates };
    this.servers.set(serverId, updated);
    await this.saveServers();
    
    this.emit('server_updated', updated);
    return updated;
  }
  
  /**
   * Get all configured servers.
   */
  public getServers(): ServerConfig[] {
    return Array.from(this.servers.values()).sort((a, b) => {
      // Sort by last connected, then by name
      if (a.lastConnected && b.lastConnected) {
        return b.lastConnected - a.lastConnected;
      }
      if (a.lastConnected) return -1;
      if (b.lastConnected) return 1;
      return a.name.localeCompare(b.name);
    });
  }
  
  /**
   * Get server by ID.
   */
  public getServer(serverId: string): ServerConfig | undefined {
    return this.servers.get(serverId);
  }
  
  /**
   * Get server status.
   */
  public getServerStatus(serverId: string): ServerStatus | undefined {
    return this.serverStatus.get(serverId);
  }
  
  /**
   * Get all server statuses.
   */
  public getAllServerStatuses(): Map<string, ServerStatus> {
    return new Map(this.serverStatus);
  }
  
  // ===========================================================================
  // Connection Management
  // ===========================================================================
  
  /**
   * Connect to a server.
   */
  public async connect(serverId: string): Promise<boolean> {
    const server = this.servers.get(serverId);
    if (!server) {
      this.emit('connection_error', { error: 'Server not found' });
      return false;
    }
    
    try {
      // Test connection
      const isOnline = await this.checkServerHealth(server);
      if (!isOnline) {
        this.emit('connection_error', { 
          error: 'Server is offline',
          serverId 
        });
        return false;
      }
      
      // Disconnect from current server if needed
      if (this.currentServerId && this.currentServerId !== serverId) {
        await this.disconnect();
      }
      
      // Set as current
      this.currentServerId = serverId;
      
      // Update last connected
      server.lastConnected = Date.now();
      await this.saveServers();
      
      this.emit('connected', { server });
      return true;
    } catch (error) {
      this.emit('connection_error', { 
        error: `Connection failed: ${error}`,
        serverId 
      });
      return false;
    }
  }
  
  /**
   * Disconnect from current server.
   */
  public async disconnect(): Promise<void> {
    if (!this.currentServerId) {
      return;
    }
    
    const serverId = this.currentServerId;
    this.currentServerId = null;
    
    this.emit('disconnected', { serverId });
  }
  
  /**
   * Get currently connected server.
   */
  public getCurrentServer(): ServerConfig | null {
    if (!this.currentServerId) {
      return null;
    }
    return this.servers.get(this.currentServerId) || null;
  }
  
  /**
   * Get URL for current server.
   */
  public getCurrentServerUrl(): string | null {
    const server = this.getCurrentServer();
    if (!server) {
      return null;
    }
    return this.getServerUrl(server);
  }
  
  /**
   * Build server URL from config.
   */
  public getServerUrl(server: ServerConfig): string {
    const protocol = server.useTLS ? 'https' : 'http';
    return `${protocol}://${server.host}:${server.port}`;
  }
  
  // ===========================================================================
  // Server Discovery
  // ===========================================================================
  
  /**
   * Discover Enigma servers on local network.
   */
  public async discoverServers(): Promise<DiscoveredServer[]> {
    if (this.isDiscovering) {
      return [];
    }
    
    this.isDiscovering = true;
    const discovered: DiscoveredServer[] = [];
    
    try {
      // Method 1: mDNS/Bonjour discovery
      // In a real implementation, this would use react-native-zeroconf
      // or similar library for mDNS service discovery
      
      // Method 2: UDP broadcast
      // Enigma servers listen on UDP port 19847 for discovery broadcasts
      
      // Method 3: Scan common ports on local subnet
      // This is a fallback for networks without mDNS support
      
      // For now, we'll use a simulated discovery
      console.log('Starting server discovery...');
      
      // Try to connect to the discovery endpoint on common local IPs
      const localSubnets = await this.getLocalSubnets();
      
      const scanPromises: Promise<DiscoveredServer | null>[] = [];
      
      for (const subnet of localSubnets) {
        // Scan last octet 1-254
        for (let i = 1; i < 255; i++) {
          const host = `${subnet}.${i}`;
          scanPromises.push(
            this.probeHost(host, DEFAULT_PORT)
              .then(result => result ? { ...result, host, port: DEFAULT_PORT } : null)
              .catch(() => null)
          );
        }
      }
      
      // Wait for all probes with timeout
      const racePromise = Promise.race([
        Promise.all(scanPromises),
        new Promise<(DiscoveredServer | null)[]>(resolve => 
          setTimeout(() => resolve([]), DISCOVERY_TIMEOUT)
        ),
      ]);
      
      const results = await racePromise;
      
      for (const result of results) {
        if (result) {
          discovered.push(result);
          this.emit('server_discovered', result);
        }
      }
      
    } catch (error) {
      console.error('Discovery error:', error);
    } finally {
      this.isDiscovering = false;
    }
    
    return discovered;
  }
  
  /**
   * Probe a host to check if it's running Enigma.
   */
  private async probeHost(
    host: string,
    port: number
  ): Promise<{ name: string; version?: string; capabilities?: string[] } | null> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1000);
      
      const response = await fetch(
        `http://${host}:${port}/api/health`,
        { 
          method: 'GET',
          signal: controller.signal,
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        return null;
      }
      
      const data = await response.json();
      
      // Check if this is an Enigma server
      if (data.service === 'enigma' || data.name?.includes('Enigma')) {
        return {
          name: data.name || `Enigma @ ${host}`,
          version: data.version,
          capabilities: data.capabilities,
        };
      }
      
      return null;
    } catch (error) {
      return null;
    }
  }
  
  /**
   * Get local network subnets.
   */
  private async getLocalSubnets(): Promise<string[]> {
    // In a real implementation, this would use react-native-network-info
    // or similar to get the device's IP address and subnet
    
    // Common home network subnets
    return ['192.168.1', '192.168.0', '10.0.0'];
  }
  
  // ===========================================================================
  // Health Checking
  // ===========================================================================
  
  /**
   * Start periodic health checking for all servers.
   */
  public startHealthChecking(): void {
    if (this.healthCheckTimer) {
      return;
    }
    
    this.healthCheckTimer = setInterval(() => {
      this.checkAllServersHealth();
    }, HEALTH_CHECK_INTERVAL);
    
    // Initial check
    this.checkAllServersHealth();
  }
  
  /**
   * Stop health checking.
   */
  public stopHealthChecking(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }
  }
  
  /**
   * Check health of all servers.
   */
  public async checkAllServersHealth(): Promise<void> {
    const checkPromises = Array.from(this.servers.values()).map(server =>
      this.checkServerHealth(server).then(online => ({ server, online }))
    );
    
    await Promise.all(checkPromises);
  }
  
  /**
   * Check health of a single server.
   */
  public async checkServerHealth(server: ServerConfig): Promise<boolean> {
    const url = this.getServerUrl(server);
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${url}/api/health`, {
        method: 'GET',
        headers: server.authToken ? {
          'Authorization': `Bearer ${server.authToken}`
        } : {},
      });
      
      const latency = Date.now() - startTime;
      
      if (!response.ok) {
        this.updateServerStatus(server.id, {
          online: false,
          latency,
          health: 'unhealthy',
          lastChecked: Date.now(),
          error: `HTTP ${response.status}`,
        });
        return false;
      }
      
      const data = await response.json();
      
      this.updateServerStatus(server.id, {
        online: true,
        latency,
        health: data.health || 'healthy',
        lastChecked: Date.now(),
        load: data.load,
        activeConnections: data.connections,
      });
      
      return true;
    } catch (error) {
      const latency = Date.now() - startTime;
      
      this.updateServerStatus(server.id, {
        online: false,
        latency,
        health: 'unknown',
        lastChecked: Date.now(),
        error: `${error}`,
      });
      
      return false;
    }
  }
  
  /**
   * Update server status and emit event if changed.
   */
  private updateServerStatus(serverId: string, status: ServerStatus): void {
    const previous = this.serverStatus.get(serverId);
    this.serverStatus.set(serverId, status);
    
    // Emit event if status changed significantly
    if (!previous || previous.online !== status.online || previous.health !== status.health) {
      this.emit('status_changed', { serverId, status });
    }
  }
  
  // ===========================================================================
  // Server Info
  // ===========================================================================
  
  /**
   * Fetch detailed server info and capabilities.
   */
  public async fetchServerInfo(server: ServerConfig): Promise<{
    capabilities: ServerCapabilities;
    metadata: ServerMetadata;
  }> {
    const url = this.getServerUrl(server);
    
    const response = await fetch(`${url}/api/info`, {
      headers: server.authToken ? {
        'Authorization': `Bearer ${server.authToken}`
      } : {},
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch server info: ${response.status}`);
    }
    
    const data = await response.json();
    
    const capabilities: ServerCapabilities = {
      hasImageGen: data.capabilities?.includes('image_gen') || false,
      hasCodeGen: data.capabilities?.includes('code_gen') || false,
      hasVideoGen: data.capabilities?.includes('video_gen') || false,
      hasAudioGen: data.capabilities?.includes('audio_gen') || false,
      has3DGen: data.capabilities?.includes('3d_gen') || false,
      hasVision: data.capabilities?.includes('vision') || false,
      hasVoice: data.capabilities?.includes('voice') || false,
      hasAvatar: data.capabilities?.includes('avatar') || false,
      hasTraining: data.capabilities?.includes('training') || false,
      modelSizes: data.model_sizes || [],
      loadedModels: data.loaded_models || [],
    };
    
    const metadata: ServerMetadata = {
      version: data.version,
      hostname: data.hostname,
      platform: data.platform,
      gpuName: data.gpu?.name,
      gpuMemory: data.gpu?.memory,
      cpuCount: data.cpu_count,
      totalMemory: data.total_memory,
    };
    
    return { capabilities, metadata };
  }
  
  /**
   * Refresh server info and capabilities.
   */
  public async refreshServerInfo(serverId: string): Promise<boolean> {
    const server = this.servers.get(serverId);
    if (!server) {
      return false;
    }
    
    try {
      const info = await this.fetchServerInfo(server);
      server.capabilities = info.capabilities;
      server.metadata = info.metadata;
      
      await this.saveServers();
      this.emit('server_updated', server);
      
      return true;
    } catch (error) {
      console.error('Failed to refresh server info:', error);
      return false;
    }
  }
  
  // ===========================================================================
  // Storage
  // ===========================================================================
  
  /**
   * Load servers from async storage.
   */
  private async loadServers(): Promise<void> {
    try {
      const data = await AsyncStorage.getItem(STORAGE_KEY);
      if (data) {
        const parsed = JSON.parse(data);
        
        for (const server of parsed.servers || []) {
          this.servers.set(server.id, server);
        }
        
        this.currentServerId = parsed.currentServerId || null;
      }
      
      // Start health checking if we have servers
      if (this.servers.size > 0) {
        this.startHealthChecking();
      }
    } catch (error) {
      console.error('Failed to load servers:', error);
    }
  }
  
  /**
   * Save servers to async storage.
   */
  private async saveServers(): Promise<void> {
    try {
      const data = {
        servers: Array.from(this.servers.values()),
        currentServerId: this.currentServerId,
      };
      
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      console.error('Failed to save servers:', error);
    }
  }
  
  /**
   * Generate unique server ID.
   */
  private generateId(): string {
    return `server_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Export singleton instance getter
export const getServerManager = (): ServerManager => ServerManager.getInstance();

export {
  ServerManager,
  ServerConfig,
  ServerCapabilities,
  ServerMetadata,
  ServerStatus,
  DiscoveredServer,
  ConnectionEventType,
};

export default ServerManager;
