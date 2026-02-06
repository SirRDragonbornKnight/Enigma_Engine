/**
 * Mobile Widget Components
 * 
 * Home screen widgets for iOS and Android.
 * Uses expo-widget for cross-platform widget support.
 * 
 * FILE: mobile/src/widgets/ChatWidget.tsx
 * TYPE: Mobile Widget
 */

import React from 'react';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Widget Configuration
export interface WidgetConfig {
  widgetId: string;
  widgetType: 'quick-chat' | 'recent-chats' | 'voice-input' | 'status';
  theme: 'light' | 'dark' | 'system';
  refreshInterval: number; // minutes
}

// Widget Data Types
export interface QuickChatData {
  lastResponse: string;
  timestamp: number;
}

export interface RecentChatsData {
  conversations: Array<{
    id: string;
    title: string;
    lastMessage: string;
    timestamp: number;
  }>;
}

export interface StatusData {
  serverOnline: boolean;
  modelLoaded: boolean;
  lastCheck: number;
}

// Storage Keys
const WIDGET_STORAGE_PREFIX = '@forgeai_widget_';

// Widget Data Manager
export class WidgetDataManager {
  static async getWidgetData<T>(widgetId: string): Promise<T | null> {
    try {
      const data = await AsyncStorage.getItem(`${WIDGET_STORAGE_PREFIX}${widgetId}`);
      return data ? JSON.parse(data) : null;
    } catch {
      return null;
    }
  }

  static async setWidgetData<T>(widgetId: string, data: T): Promise<void> {
    try {
      await AsyncStorage.setItem(`${WIDGET_STORAGE_PREFIX}${widgetId}`, JSON.stringify(data));
    } catch (error) {
      console.error('Failed to save widget data:', error);
    }
  }

  static async getRecentConversations(limit: number = 3): Promise<RecentChatsData> {
    try {
      const stored = await AsyncStorage.getItem('@forgeai_conversations');
      if (!stored) return { conversations: [] };

      const conversations = JSON.parse(stored);
      return {
        conversations: conversations.slice(0, limit).map((c: any) => ({
          id: c.id,
          title: c.title,
          lastMessage: c.messages[c.messages.length - 1]?.content?.slice(0, 50) || '',
          timestamp: c.updatedAt,
        })),
      };
    } catch {
      return { conversations: [] };
    }
  }

  static async checkServerStatus(serverUrl: string): Promise<StatusData> {
    try {
      const response = await fetch(`${serverUrl}/health`, {
        method: 'GET',
        timeout: 5000,
      } as any);
      
      return {
        serverOnline: response.ok,
        modelLoaded: response.ok,
        lastCheck: Date.now(),
      };
    } catch {
      return {
        serverOnline: false,
        modelLoaded: false,
        lastCheck: Date.now(),
      };
    }
  }
}

// Quick Chat Widget Component
export function QuickChatWidget({ config }: { config: WidgetConfig }) {
  const [input, setInput] = React.useState('');
  const [response, setResponse] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const isDark = config.theme === 'dark';

  const sendQuickMessage = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    try {
      const settings = await AsyncStorage.getItem('@forgeai_settings');
      const { serverUrl, apiKey } = settings ? JSON.parse(settings) : { serverUrl: 'http://localhost:8000', apiKey: '' };

      const res = await fetch(`${serverUrl}/v1/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          prompt: input,
          max_tokens: 100,
        }),
      });

      const data = await res.json();
      const text = data.choices?.[0]?.text || 'No response';
      setResponse(text);
      
      await WidgetDataManager.setWidgetData<QuickChatData>(config.widgetId, {
        lastResponse: text,
        timestamp: Date.now(),
      });
    } catch (error) {
      setResponse(`Error: ${error}`);
    } finally {
      setLoading(false);
      setInput('');
    }
  };

  return (
    <View style={[styles.widgetContainer, isDark && styles.darkWidget]}>
      <Text style={[styles.widgetTitle, isDark && styles.darkText]}>ForgeAI Quick Chat</Text>
      
      <View style={styles.inputRow}>
        <TextInput
          style={[styles.widgetInput, isDark && styles.darkInput]}
          value={input}
          onChangeText={setInput}
          placeholder="Ask something..."
          placeholderTextColor={isDark ? '#888' : '#999'}
        />
        <TouchableOpacity
          style={[styles.sendBtn, loading && styles.disabledBtn]}
          onPress={sendQuickMessage}
          disabled={loading}
        >
          <Text style={styles.sendBtnText}>{loading ? '...' : 'Go'}</Text>
        </TouchableOpacity>
      </View>

      {response ? (
        <Text style={[styles.responseText, isDark && styles.darkText]} numberOfLines={3}>
          {response}
        </Text>
      ) : null}
    </View>
  );
}

// TextInput import needed
import { TextInput } from 'react-native';

// Recent Chats Widget Component
export function RecentChatsWidget({ config }: { config: WidgetConfig }) {
  const [chats, setChats] = React.useState<RecentChatsData>({ conversations: [] });
  const isDark = config.theme === 'dark';

  React.useEffect(() => {
    loadRecentChats();
  }, []);

  const loadRecentChats = async () => {
    const data = await WidgetDataManager.getRecentConversations(3);
    setChats(data);
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <View style={[styles.widgetContainer, isDark && styles.darkWidget]}>
      <Text style={[styles.widgetTitle, isDark && styles.darkText]}>Recent Chats</Text>
      
      {chats.conversations.length === 0 ? (
        <Text style={[styles.emptyText, isDark && styles.darkText]}>No recent chats</Text>
      ) : (
        chats.conversations.map((chat) => (
          <View key={chat.id} style={[styles.chatItem, isDark && styles.darkChatItem]}>
            <Text style={[styles.chatTitle, isDark && styles.darkText]} numberOfLines={1}>
              {chat.title}
            </Text>
            <Text style={styles.chatMeta}>{formatTime(chat.timestamp)}</Text>
          </View>
        ))
      )}
    </View>
  );
}

// Status Widget Component
export function StatusWidget({ config }: { config: WidgetConfig }) {
  const [status, setStatus] = React.useState<StatusData>({
    serverOnline: false,
    modelLoaded: false,
    lastCheck: 0,
  });
  const isDark = config.theme === 'dark';

  React.useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    const settings = await AsyncStorage.getItem('@forgeai_settings');
    const { serverUrl } = settings ? JSON.parse(settings) : { serverUrl: 'http://localhost:8000' };
    
    const data = await WidgetDataManager.checkServerStatus(serverUrl);
    setStatus(data);
  };

  return (
    <View style={[styles.widgetContainer, styles.statusWidget, isDark && styles.darkWidget]}>
      <Text style={[styles.widgetTitle, isDark && styles.darkText]}>ForgeAI Status</Text>
      
      <View style={styles.statusRow}>
        <View style={[
          styles.statusDot,
          status.serverOnline ? styles.statusOnline : styles.statusOffline
        ]} />
        <Text style={[styles.statusLabel, isDark && styles.darkText]}>
          Server: {status.serverOnline ? 'Online' : 'Offline'}
        </Text>
      </View>

      <View style={styles.statusRow}>
        <View style={[
          styles.statusDot,
          status.modelLoaded ? styles.statusOnline : styles.statusOffline
        ]} />
        <Text style={[styles.statusLabel, isDark && styles.darkText]}>
          Model: {status.modelLoaded ? 'Ready' : 'Not loaded'}
        </Text>
      </View>

      <TouchableOpacity style={styles.refreshBtn} onPress={checkStatus}>
        <Text style={styles.refreshBtnText}>Refresh</Text>
      </TouchableOpacity>
    </View>
  );
}

// Voice Input Widget (placeholder)
export function VoiceInputWidget({ config }: { config: WidgetConfig }) {
  const isDark = config.theme === 'dark';
  const [isListening, setIsListening] = React.useState(false);

  return (
    <View style={[styles.widgetContainer, styles.voiceWidget, isDark && styles.darkWidget]}>
      <TouchableOpacity
        style={[styles.voiceBtn, isListening && styles.voiceBtnActive]}
        onPressIn={() => setIsListening(true)}
        onPressOut={() => setIsListening(false)}
      >
        <Text style={styles.voiceBtnText}>{isListening ? 'Listening...' : 'Hold to Speak'}</Text>
      </TouchableOpacity>
    </View>
  );
}

// Widget Registry
export const WIDGET_REGISTRY = {
  'quick-chat': QuickChatWidget,
  'recent-chats': RecentChatsWidget,
  'status': StatusWidget,
  'voice-input': VoiceInputWidget,
} as const;

// Widget Configuration for native modules
export const WIDGET_CONFIGURATIONS = [
  {
    name: 'ForgeAI Quick Chat',
    description: 'Ask quick questions from your home screen',
    widgetType: 'quick-chat' as const,
    supportedSizes: ['small', 'medium'],
    defaultSize: 'medium',
  },
  {
    name: 'Recent Conversations',
    description: 'See your recent chat conversations',
    widgetType: 'recent-chats' as const,
    supportedSizes: ['small', 'medium', 'large'],
    defaultSize: 'medium',
  },
  {
    name: 'Server Status',
    description: 'Check ForgeAI server status',
    widgetType: 'status' as const,
    supportedSizes: ['small'],
    defaultSize: 'small',
  },
  {
    name: 'Voice Input',
    description: 'Quick voice input to ForgeAI',
    widgetType: 'voice-input' as const,
    supportedSizes: ['small'],
    defaultSize: 'small',
  },
];

// Styles
const styles = StyleSheet.create({
  widgetContainer: {
    backgroundColor: '#ffffff',
    borderRadius: 16,
    padding: 16,
    margin: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  darkWidget: {
    backgroundColor: '#2a2a2a',
  },
  widgetTitle: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 12,
    color: '#333',
  },
  darkText: {
    color: '#fff',
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  widgetInput: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginRight: 8,
  },
  darkInput: {
    backgroundColor: '#3a3a3a',
    color: '#fff',
  },
  sendBtn: {
    backgroundColor: '#007AFF',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  disabledBtn: {
    opacity: 0.5,
  },
  sendBtnText: {
    color: '#fff',
    fontWeight: '600',
  },
  responseText: {
    marginTop: 12,
    fontSize: 13,
    color: '#666',
  },
  emptyText: {
    color: '#888',
    textAlign: 'center',
    padding: 16,
  },
  chatItem: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  darkChatItem: {
    borderBottomColor: '#444',
  },
  chatTitle: {
    fontSize: 14,
    fontWeight: '500',
  },
  chatMeta: {
    fontSize: 11,
    color: '#888',
    marginTop: 2,
  },
  statusWidget: {
    minWidth: 150,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusOnline: {
    backgroundColor: '#4CAF50',
  },
  statusOffline: {
    backgroundColor: '#F44336',
  },
  statusLabel: {
    fontSize: 13,
  },
  refreshBtn: {
    marginTop: 12,
    padding: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 6,
    alignItems: 'center',
  },
  refreshBtnText: {
    fontSize: 12,
    color: '#007AFF',
  },
  voiceWidget: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceBtn: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#007AFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceBtnActive: {
    backgroundColor: '#FF3B30',
  },
  voiceBtnText: {
    color: '#fff',
    fontSize: 12,
    textAlign: 'center',
  },
});

export default {
  QuickChatWidget,
  RecentChatsWidget,
  StatusWidget,
  VoiceInputWidget,
  WidgetDataManager,
  WIDGET_REGISTRY,
  WIDGET_CONFIGURATIONS,
};
