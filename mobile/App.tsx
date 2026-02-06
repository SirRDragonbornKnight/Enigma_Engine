/**
 * ForgeAI Mobile App
 * 
 * Cross-platform React Native app for iOS and Android.
 * 
 * FILE: mobile/App.tsx
 * TYPE: Mobile
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  SafeAreaView,
  Alert,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import 'react-native-get-random-values';
import { v4 as uuidv4 } from 'uuid';

// Types
interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

interface Settings {
  serverUrl: string;
  apiKey: string;
  darkMode: boolean;
  voiceEnabled: boolean;
  hapticFeedback: boolean;
  fontSize: 'small' | 'medium' | 'large';
}

// API Service
class ForgeAPIService {
  private serverUrl: string;
  private apiKey: string;

  constructor(serverUrl: string, apiKey: string = '') {
    this.serverUrl = serverUrl;
    this.apiKey = apiKey;
  }

  async generate(prompt: string, options: any = {}): Promise<string> {
    const response = await fetch(`${this.serverUrl}/v1/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.apiKey ? { 'Authorization': `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens || 256,
        temperature: options.temperature || 0.7,
        ...options,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.text || data.text || '';
  }

  async chat(messages: Message[], options: any = {}): Promise<string> {
    const response = await fetch(`${this.serverUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.apiKey ? { 'Authorization': `Bearer ${this.apiKey}` } : {}),
      },
      body: JSON.stringify({
        messages: messages.map(m => ({ role: m.role, content: m.content })),
        max_tokens: options.maxTokens || 256,
        temperature: options.temperature || 0.7,
        ...options,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '';
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.serverUrl}/health`, {
        method: 'GET',
        timeout: 5000,
      } as any);
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Storage helpers
const STORAGE_KEYS = {
  SETTINGS: '@forgeai_settings',
  CONVERSATIONS: '@forgeai_conversations',
  CURRENT_CONVERSATION: '@forgeai_current_conversation',
};

const defaultSettings: Settings = {
  serverUrl: 'http://localhost:8000',
  apiKey: '',
  darkMode: true,
  voiceEnabled: true,
  hapticFeedback: true,
  fontSize: 'medium',
};

// Chat Screen
function ChatScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [conversationId, setConversationId] = useState<string>('');

  useEffect(() => {
    loadSettings();
    loadOrCreateConversation();
  }, []);

  const loadSettings = async () => {
    try {
      const stored = await AsyncStorage.getItem(STORAGE_KEYS.SETTINGS);
      if (stored) {
        setSettings({ ...defaultSettings, ...JSON.parse(stored) });
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const loadOrCreateConversation = async () => {
    try {
      const currentId = await AsyncStorage.getItem(STORAGE_KEYS.CURRENT_CONVERSATION);
      if (currentId) {
        const conversations = await AsyncStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
        if (conversations) {
          const parsed = JSON.parse(conversations) as Conversation[];
          const current = parsed.find(c => c.id === currentId);
          if (current) {
            setConversationId(current.id);
            setMessages(current.messages);
            return;
          }
        }
      }
      // Create new conversation
      const newId = uuidv4();
      setConversationId(newId);
      setMessages([]);
      await AsyncStorage.setItem(STORAGE_KEYS.CURRENT_CONVERSATION, newId);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const saveConversation = async (newMessages: Message[]) => {
    try {
      const stored = await AsyncStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
      let conversations: Conversation[] = stored ? JSON.parse(stored) : [];
      
      const existingIndex = conversations.findIndex(c => c.id === conversationId);
      const conversation: Conversation = {
        id: conversationId,
        title: newMessages[0]?.content.slice(0, 50) || 'New Chat',
        messages: newMessages,
        createdAt: existingIndex >= 0 ? conversations[existingIndex].createdAt : Date.now(),
        updatedAt: Date.now(),
      };

      if (existingIndex >= 0) {
        conversations[existingIndex] = conversation;
      } else {
        conversations.unshift(conversation);
      }

      await AsyncStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
    } catch (error) {
      console.error('Failed to save conversation:', error);
    }
  };

  const sendMessage = useCallback(async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: inputText.trim(),
      timestamp: Date.now(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputText('');
    setIsLoading(true);

    if (settings.hapticFeedback) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }

    try {
      const api = new ForgeAPIService(settings.serverUrl, settings.apiKey);
      const response = await api.chat(newMessages);

      const assistantMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: response,
        timestamp: Date.now(),
      };

      const updatedMessages = [...newMessages, assistantMessage];
      setMessages(updatedMessages);
      await saveConversation(updatedMessages);

      if (settings.voiceEnabled) {
        Speech.speak(response, { language: 'en' });
      }

      if (settings.hapticFeedback) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }
    } catch (error) {
      Alert.alert('Error', `Failed to get response: ${error}`);
      if (settings.hapticFeedback) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }
    } finally {
      setIsLoading(false);
    }
  }, [inputText, messages, settings, isLoading, conversationId]);

  const renderMessage = ({ item }: { item: Message }) => (
    <View style={[
      styles.messageContainer,
      item.role === 'user' ? styles.userMessage : styles.assistantMessage,
      settings.darkMode && styles.darkMessage
    ]}>
      <Text style={[
        styles.messageText,
        settings.darkMode && styles.darkText,
        { fontSize: settings.fontSize === 'small' ? 14 : settings.fontSize === 'large' ? 18 : 16 }
      ]}>
        {item.content}
      </Text>
      <Text style={styles.timestamp}>
        {new Date(item.timestamp).toLocaleTimeString()}
      </Text>
    </View>
  );

  const newConversation = async () => {
    const newId = uuidv4();
    setConversationId(newId);
    setMessages([]);
    await AsyncStorage.setItem(STORAGE_KEYS.CURRENT_CONVERSATION, newId);
  };

  return (
    <SafeAreaView style={[styles.container, settings.darkMode && styles.darkContainer]}>
      <StatusBar style={settings.darkMode ? 'light' : 'dark'} />
      
      <View style={styles.header}>
        <Text style={[styles.headerTitle, settings.darkMode && styles.darkText]}>ForgeAI</Text>
        <TouchableOpacity onPress={newConversation} style={styles.newChatButton}>
          <Ionicons name="add-circle-outline" size={28} color={settings.darkMode ? '#fff' : '#000'} />
        </TouchableOpacity>
      </View>

      <FlatList
        data={messages}
        renderItem={renderMessage}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messageList}
        inverted={false}
      />

      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={[styles.loadingText, settings.darkMode && styles.darkText]}>Thinking...</Text>
        </View>
      )}

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.inputContainer}
      >
        <TextInput
          style={[styles.input, settings.darkMode && styles.darkInput]}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type a message..."
          placeholderTextColor={settings.darkMode ? '#888' : '#999'}
          multiline
          maxLength={4000}
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity 
          style={[styles.sendButton, isLoading && styles.disabledButton]}
          onPress={sendMessage}
          disabled={isLoading}
        >
          <Ionicons name="send" size={24} color="#fff" />
        </TouchableOpacity>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// History Screen
function HistoryScreen() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [settings, setSettings] = useState<Settings>(defaultSettings);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const settingsData = await AsyncStorage.getItem(STORAGE_KEYS.SETTINGS);
      if (settingsData) setSettings(JSON.parse(settingsData));

      const conversationsData = await AsyncStorage.getItem(STORAGE_KEYS.CONVERSATIONS);
      if (conversationsData) setConversations(JSON.parse(conversationsData));
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  const selectConversation = async (id: string) => {
    await AsyncStorage.setItem(STORAGE_KEYS.CURRENT_CONVERSATION, id);
    Alert.alert('Conversation Loaded', 'Switch to Chat tab to continue.');
  };

  const deleteConversation = async (id: string) => {
    const updated = conversations.filter(c => c.id !== id);
    setConversations(updated);
    await AsyncStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(updated));
  };

  const renderConversation = ({ item }: { item: Conversation }) => (
    <TouchableOpacity 
      style={[styles.conversationItem, settings.darkMode && styles.darkCard]}
      onPress={() => selectConversation(item.id)}
      onLongPress={() => {
        Alert.alert('Delete?', 'Delete this conversation?', [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Delete', style: 'destructive', onPress: () => deleteConversation(item.id) }
        ]);
      }}
    >
      <Text style={[styles.conversationTitle, settings.darkMode && styles.darkText]} numberOfLines={1}>
        {item.title}
      </Text>
      <Text style={styles.conversationMeta}>
        {item.messages.length} messages - {new Date(item.updatedAt).toLocaleDateString()}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={[styles.container, settings.darkMode && styles.darkContainer]}>
      <Text style={[styles.screenTitle, settings.darkMode && styles.darkText]}>History</Text>
      <FlatList
        data={conversations}
        renderItem={renderConversation}
        keyExtractor={item => item.id}
        ListEmptyComponent={
          <Text style={[styles.emptyText, settings.darkMode && styles.darkText]}>
            No conversations yet
          </Text>
        }
      />
    </SafeAreaView>
  );
}

// Settings Screen
function SettingsScreen() {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    loadSettings();
  }, []);

  useEffect(() => {
    checkServerStatus();
  }, [settings.serverUrl]);

  const loadSettings = async () => {
    try {
      const stored = await AsyncStorage.getItem(STORAGE_KEYS.SETTINGS);
      if (stored) {
        setSettings({ ...defaultSettings, ...JSON.parse(stored) });
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const saveSettings = async (newSettings: Settings) => {
    setSettings(newSettings);
    await AsyncStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(newSettings));
  };

  const checkServerStatus = async () => {
    setServerStatus('checking');
    try {
      const api = new ForgeAPIService(settings.serverUrl, settings.apiKey);
      const isOnline = await api.healthCheck();
      setServerStatus(isOnline ? 'online' : 'offline');
    } catch {
      setServerStatus('offline');
    }
  };

  const SettingRow = ({ label, children }: { label: string; children: React.ReactNode }) => (
    <View style={[styles.settingRow, settings.darkMode && styles.darkCard]}>
      <Text style={[styles.settingLabel, settings.darkMode && styles.darkText]}>{label}</Text>
      {children}
    </View>
  );

  const Toggle = ({ value, onToggle }: { value: boolean; onToggle: () => void }) => (
    <TouchableOpacity 
      style={[styles.toggle, value && styles.toggleActive]}
      onPress={onToggle}
    >
      <View style={[styles.toggleKnob, value && styles.toggleKnobActive]} />
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={[styles.container, settings.darkMode && styles.darkContainer]}>
      <Text style={[styles.screenTitle, settings.darkMode && styles.darkText]}>Settings</Text>
      
      <View style={styles.settingsSection}>
        <Text style={[styles.sectionTitle, settings.darkMode && styles.darkText]}>Server</Text>
        
        <SettingRow label="Server URL">
          <TextInput
            style={[styles.settingInput, settings.darkMode && styles.darkInput]}
            value={settings.serverUrl}
            onChangeText={(text) => saveSettings({ ...settings, serverUrl: text })}
            placeholder="http://localhost:8000"
            autoCapitalize="none"
            autoCorrect={false}
          />
        </SettingRow>

        <SettingRow label="API Key">
          <TextInput
            style={[styles.settingInput, settings.darkMode && styles.darkInput]}
            value={settings.apiKey}
            onChangeText={(text) => saveSettings({ ...settings, apiKey: text })}
            placeholder="Optional"
            secureTextEntry
          />
        </SettingRow>

        <View style={styles.statusRow}>
          <Text style={settings.darkMode ? styles.darkText : undefined}>Status: </Text>
          <View style={[
            styles.statusDot,
            serverStatus === 'online' && styles.statusOnline,
            serverStatus === 'offline' && styles.statusOffline,
            serverStatus === 'checking' && styles.statusChecking,
          ]} />
          <Text style={settings.darkMode ? styles.darkText : undefined}>
            {serverStatus === 'checking' ? 'Checking...' : serverStatus}
          </Text>
          <TouchableOpacity onPress={checkServerStatus} style={styles.refreshButton}>
            <Ionicons name="refresh" size={20} color={settings.darkMode ? '#fff' : '#000'} />
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.settingsSection}>
        <Text style={[styles.sectionTitle, settings.darkMode && styles.darkText]}>Appearance</Text>
        
        <SettingRow label="Dark Mode">
          <Toggle 
            value={settings.darkMode} 
            onToggle={() => saveSettings({ ...settings, darkMode: !settings.darkMode })} 
          />
        </SettingRow>

        <SettingRow label="Font Size">
          <View style={styles.fontSizeButtons}>
            {(['small', 'medium', 'large'] as const).map((size) => (
              <TouchableOpacity
                key={size}
                style={[
                  styles.fontSizeButton,
                  settings.fontSize === size && styles.fontSizeButtonActive
                ]}
                onPress={() => saveSettings({ ...settings, fontSize: size })}
              >
                <Text style={settings.fontSize === size ? styles.fontSizeTextActive : undefined}>
                  {size.charAt(0).toUpperCase()}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </SettingRow>
      </View>

      <View style={styles.settingsSection}>
        <Text style={[styles.sectionTitle, settings.darkMode && styles.darkText]}>Features</Text>
        
        <SettingRow label="Voice Output">
          <Toggle 
            value={settings.voiceEnabled} 
            onToggle={() => saveSettings({ ...settings, voiceEnabled: !settings.voiceEnabled })} 
          />
        </SettingRow>

        <SettingRow label="Haptic Feedback">
          <Toggle 
            value={settings.hapticFeedback} 
            onToggle={() => saveSettings({ ...settings, hapticFeedback: !settings.hapticFeedback })} 
          />
        </SettingRow>
      </View>

      <TouchableOpacity 
        style={styles.dangerButton}
        onPress={() => {
          Alert.alert('Clear All Data?', 'This will delete all conversations and settings.', [
            { text: 'Cancel', style: 'cancel' },
            { 
              text: 'Clear', 
              style: 'destructive', 
              onPress: async () => {
                await AsyncStorage.clear();
                setSettings(defaultSettings);
              }
            }
          ]);
        }}
      >
        <Text style={styles.dangerButtonText}>Clear All Data</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

// Navigation
const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName: keyof typeof Ionicons.glyphMap;

            if (route.name === 'Chat') {
              iconName = focused ? 'chatbubbles' : 'chatbubbles-outline';
            } else if (route.name === 'History') {
              iconName = focused ? 'time' : 'time-outline';
            } else {
              iconName = focused ? 'settings' : 'settings-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#007AFF',
          tabBarInactiveTintColor: 'gray',
          headerShown: false,
        })}
      >
        <Tab.Screen name="Chat" component={ChatScreen} />
        <Tab.Screen name="History" component={HistoryScreen} />
        <Tab.Screen name="Settings" component={SettingsScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  darkContainer: {
    backgroundColor: '#1a1a1a',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  newChatButton: {
    padding: 4,
  },
  messageList: {
    padding: 16,
    paddingBottom: 80,
  },
  messageContainer: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
    marginVertical: 4,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#e5e5ea',
  },
  darkMessage: {
    backgroundColor: '#333',
  },
  messageText: {
    fontSize: 16,
    color: '#fff',
  },
  darkText: {
    color: '#fff',
  },
  timestamp: {
    fontSize: 10,
    color: 'rgba(255,255,255,0.7)',
    marginTop: 4,
    textAlign: 'right',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 8,
  },
  loadingText: {
    marginLeft: 8,
    color: '#666',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    backgroundColor: '#fff',
  },
  input: {
    flex: 1,
    backgroundColor: '#f0f0f0',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    marginRight: 8,
    maxHeight: 100,
  },
  darkInput: {
    backgroundColor: '#333',
    color: '#fff',
  },
  sendButton: {
    backgroundColor: '#007AFF',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  disabledButton: {
    opacity: 0.5,
  },
  screenTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    padding: 16,
  },
  conversationItem: {
    padding: 16,
    marginHorizontal: 16,
    marginVertical: 4,
    backgroundColor: '#fff',
    borderRadius: 12,
  },
  darkCard: {
    backgroundColor: '#2a2a2a',
  },
  conversationTitle: {
    fontSize: 16,
    fontWeight: '600',
  },
  conversationMeta: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  emptyText: {
    textAlign: 'center',
    color: '#888',
    marginTop: 40,
  },
  settingsSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#888',
    paddingHorizontal: 16,
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginVertical: 2,
    borderRadius: 8,
  },
  settingLabel: {
    fontSize: 16,
  },
  settingInput: {
    flex: 1,
    marginLeft: 16,
    padding: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    textAlign: 'right',
  },
  toggle: {
    width: 50,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#e0e0e0',
    padding: 2,
  },
  toggleActive: {
    backgroundColor: '#007AFF',
  },
  toggleKnob: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: '#fff',
  },
  toggleKnobActive: {
    alignSelf: 'flex-end',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginHorizontal: 8,
  },
  statusOnline: {
    backgroundColor: '#4CAF50',
  },
  statusOffline: {
    backgroundColor: '#F44336',
  },
  statusChecking: {
    backgroundColor: '#FFC107',
  },
  refreshButton: {
    marginLeft: 'auto',
    padding: 8,
  },
  fontSizeButtons: {
    flexDirection: 'row',
  },
  fontSizeButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#e0e0e0',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  fontSizeButtonActive: {
    backgroundColor: '#007AFF',
  },
  fontSizeTextActive: {
    color: '#fff',
  },
  dangerButton: {
    margin: 16,
    padding: 16,
    backgroundColor: '#F44336',
    borderRadius: 8,
    alignItems: 'center',
  },
  dangerButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
});
