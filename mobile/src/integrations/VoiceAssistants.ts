/**
 * Voice Assistant Integration
 * 
 * Siri Shortcuts and Google Assistant integration for ForgeAI.
 * 
 * FILE: mobile/src/integrations/VoiceAssistants.ts
 * TYPE: Mobile Integration
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Linking from 'expo-linking';

// Types
export interface ShortcutAction {
  id: string;
  title: string;
  description: string;
  invocationPhrase: string;
  action: 'generate' | 'chat' | 'summarize' | 'translate' | 'custom';
  parameters?: Record<string, any>;
}

export interface AssistantResponse {
  text: string;
  success: boolean;
  action?: string;
  data?: any;
}

export interface IntentData {
  intent: string;
  parameters: Record<string, string>;
  source: 'siri' | 'google' | 'alexa' | 'unknown';
}

// Predefined Shortcuts
export const PREDEFINED_SHORTCUTS: ShortcutAction[] = [
  {
    id: 'ask-forge',
    title: 'Ask ForgeAI',
    description: 'Ask ForgeAI any question',
    invocationPhrase: 'Ask Forge',
    action: 'generate',
  },
  {
    id: 'summarize-text',
    title: 'Summarize Text',
    description: 'Summarize text from clipboard',
    invocationPhrase: 'Summarize this',
    action: 'summarize',
  },
  {
    id: 'translate-text',
    title: 'Translate Text',
    description: 'Translate text to another language',
    invocationPhrase: 'Translate this',
    action: 'translate',
    parameters: { targetLanguage: 'Spanish' },
  },
  {
    id: 'continue-chat',
    title: 'Continue Chat',
    description: 'Continue your last conversation',
    invocationPhrase: 'Continue my chat',
    action: 'chat',
  },
  {
    id: 'quick-note',
    title: 'Quick Note to AI',
    description: 'Add a quick note for AI context',
    invocationPhrase: 'Note for Forge',
    action: 'custom',
    parameters: { customAction: 'add_note' },
  },
];

// Siri Shortcuts Manager
export class SiriShortcutsManager {
  private shortcuts: Map<string, ShortcutAction> = new Map();

  constructor() {
    PREDEFINED_SHORTCUTS.forEach(s => this.shortcuts.set(s.id, s));
  }

  /**
   * Register a shortcut with Siri (iOS only)
   * In production, use react-native-siri-shortcut
   */
  async registerShortcut(shortcut: ShortcutAction): Promise<boolean> {
    try {
      this.shortcuts.set(shortcut.id, shortcut);
      
      // Store for persistence
      const stored = await this.getAllShortcuts();
      stored.push(shortcut);
      await AsyncStorage.setItem('@forgeai_shortcuts', JSON.stringify(stored));

      // In production:
      // import { ShortcutsEmitter, SiriShortcutsEvent } from 'react-native-siri-shortcut';
      // ShortcutsEmitter.addListener('SiriShortcutListener', this.handleShortcut);
      // addShortcut({ activityType: shortcut.id, ... });

      console.log(`Registered Siri shortcut: ${shortcut.title}`);
      return true;
    } catch (error) {
      console.error('Failed to register shortcut:', error);
      return false;
    }
  }

  /**
   * Handle incoming Siri shortcut
   */
  async handleShortcut(shortcutId: string, userInput?: string): Promise<AssistantResponse> {
    const shortcut = this.shortcuts.get(shortcutId);
    if (!shortcut) {
      return {
        text: 'Shortcut not found',
        success: false,
      };
    }

    return this.executeAction(shortcut, userInput);
  }

  /**
   * Execute shortcut action
   */
  private async executeAction(shortcut: ShortcutAction, userInput?: string): Promise<AssistantResponse> {
    const settings = await AsyncStorage.getItem('@forgeai_settings');
    const { serverUrl, apiKey } = settings 
      ? JSON.parse(settings) 
      : { serverUrl: 'http://localhost:8000', apiKey: '' };

    try {
      switch (shortcut.action) {
        case 'generate':
          return this.generateResponse(serverUrl, apiKey, userInput || 'Hello');

        case 'chat':
          return this.continueChat(serverUrl, apiKey, userInput);

        case 'summarize':
          return this.summarizeText(serverUrl, apiKey, userInput || '');

        case 'translate':
          const targetLang = shortcut.parameters?.targetLanguage || 'Spanish';
          return this.translateText(serverUrl, apiKey, userInput || '', targetLang);

        case 'custom':
          return this.handleCustomAction(shortcut.parameters?.customAction, userInput);

        default:
          return { text: 'Unknown action', success: false };
      }
    } catch (error) {
      return {
        text: `Error: ${error}`,
        success: false,
      };
    }
  }

  private async generateResponse(serverUrl: string, apiKey: string, prompt: string): Promise<AssistantResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
    
    try {
      const response = await fetch(`${serverUrl}/v1/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          prompt,
          max_tokens: 150,
          temperature: 0.7,
        }),
        signal: controller.signal,
      });

      const data = await response.json();
      return {
        text: data.choices?.[0]?.text || 'No response',
        success: true,
        action: 'generate',
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async continueChat(serverUrl: string, apiKey: string, userInput?: string): Promise<AssistantResponse> {
    // Load last conversation
    const conversations = await AsyncStorage.getItem('@forgeai_conversations');
    const parsed = conversations ? JSON.parse(conversations) : [];
    const lastConvo = parsed[0];

    if (!lastConvo) {
      return { text: 'No previous conversation found', success: false };
    }

    const messages = lastConvo.messages || [];
    if (userInput) {
      messages.push({ role: 'user', content: userInput });
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
    
    try {
      const response = await fetch(`${serverUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify({
          messages: messages.slice(-10), // Last 10 messages for context
          max_tokens: 150,
        }),
        signal: controller.signal,
      });

      const data = await response.json();
      return {
        text: data.choices?.[0]?.message?.content || 'No response',
        success: true,
        action: 'chat',
      };
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private async summarizeText(serverUrl: string, apiKey: string, text: string): Promise<AssistantResponse> {
    const prompt = `Summarize the following text concisely:\n\n${text}\n\nSummary:`;
    return this.generateResponse(serverUrl, apiKey, prompt);
  }

  private async translateText(serverUrl: string, apiKey: string, text: string, targetLang: string): Promise<AssistantResponse> {
    const prompt = `Translate the following text to ${targetLang}:\n\n${text}\n\nTranslation:`;
    return this.generateResponse(serverUrl, apiKey, prompt);
  }

  private async handleCustomAction(action: string | undefined, userInput?: string): Promise<AssistantResponse> {
    switch (action) {
      case 'add_note':
        // Save note for AI context
        const notes = await AsyncStorage.getItem('@forgeai_notes');
        const parsed = notes ? JSON.parse(notes) : [];
        parsed.push({ text: userInput, timestamp: Date.now() });
        await AsyncStorage.setItem('@forgeai_notes', JSON.stringify(parsed));
        return { text: 'Note saved', success: true, action: 'add_note' };

      default:
        return { text: 'Unknown custom action', success: false };
    }
  }

  async getAllShortcuts(): Promise<ShortcutAction[]> {
    try {
      const stored = await AsyncStorage.getItem('@forgeai_shortcuts');
      return stored ? JSON.parse(stored) : PREDEFINED_SHORTCUTS;
    } catch {
      return PREDEFINED_SHORTCUTS;
    }
  }

  async removeShortcut(shortcutId: string): Promise<boolean> {
    try {
      this.shortcuts.delete(shortcutId);
      const shortcuts = await this.getAllShortcuts();
      const filtered = shortcuts.filter(s => s.id !== shortcutId);
      await AsyncStorage.setItem('@forgeai_shortcuts', JSON.stringify(filtered));
      return true;
    } catch {
      return false;
    }
  }
}

// Google Assistant Actions Manager
export class GoogleAssistantManager {
  private actionHandlers: Map<string, (params: any) => Promise<AssistantResponse>> = new Map();

  constructor() {
    this.registerDefaultHandlers();
  }

  private registerDefaultHandlers(): void {
    // actions.intent.TEXT - General text query
    this.actionHandlers.set('actions.intent.TEXT', async (params) => {
      const query = params.queryText || params.text || '';
      const siri = new SiriShortcutsManager();
      return siri.handleShortcut('ask-forge', query);
    });

    // Custom ForgeAI intents
    this.actionHandlers.set('forge.ask', async (params) => {
      const siri = new SiriShortcutsManager();
      return siri.handleShortcut('ask-forge', params.question);
    });

    this.actionHandlers.set('forge.summarize', async (params) => {
      const siri = new SiriShortcutsManager();
      return siri.handleShortcut('summarize-text', params.text);
    });

    this.actionHandlers.set('forge.translate', async (params) => {
      const siri = new SiriShortcutsManager();
      // Set target language in shortcut params
      const shortcut = PREDEFINED_SHORTCUTS.find(s => s.id === 'translate-text');
      if (shortcut) {
        shortcut.parameters = { targetLanguage: params.language || 'Spanish' };
      }
      return siri.handleShortcut('translate-text', params.text);
    });
  }

  /**
   * Handle incoming Google Assistant action
   */
  async handleAction(intent: string, parameters: Record<string, any>): Promise<AssistantResponse> {
    const handler = this.actionHandlers.get(intent);
    if (handler) {
      return handler(parameters);
    }

    // Default fallback - treat as general query
    const defaultHandler = this.actionHandlers.get('actions.intent.TEXT');
    if (defaultHandler) {
      return defaultHandler(parameters);
    }

    return {
      text: 'Action not supported',
      success: false,
    };
  }

  /**
   * Register custom action handler
   */
  registerHandler(intent: string, handler: (params: any) => Promise<AssistantResponse>): void {
    this.actionHandlers.set(intent, handler);
  }

  /**
   * Generate Action.json for Google Assistant
   */
  generateActionsJson(): object {
    return {
      actions: [
        {
          name: 'actions.intent.MAIN',
          fulfillment: {
            conversationName: 'ForgeAI'
          },
          intent: {
            name: 'actions.intent.MAIN'
          }
        },
        {
          name: 'forge.ask',
          fulfillment: {
            conversationName: 'ForgeAI'
          },
          intent: {
            name: 'forge.ask',
            parameters: [
              {
                name: 'question',
                type: 'SchemaOrg_Text'
              }
            ],
            trigger: {
              queryPatterns: [
                'ask forge $SchemaOrg_Text:question',
                'tell forge $SchemaOrg_Text:question',
                'forge ai $SchemaOrg_Text:question'
              ]
            }
          }
        },
        {
          name: 'forge.summarize',
          fulfillment: {
            conversationName: 'ForgeAI'
          },
          intent: {
            name: 'forge.summarize',
            parameters: [
              {
                name: 'text',
                type: 'SchemaOrg_Text'
              }
            ],
            trigger: {
              queryPatterns: [
                'summarize with forge $SchemaOrg_Text:text',
                'forge summarize $SchemaOrg_Text:text'
              ]
            }
          }
        }
      ],
      conversations: {
        ForgeAI: {
          name: 'ForgeAI',
          url: 'https://api.forgeai.dev/assistant'
        }
      }
    };
  }
}

// Deep Link Handler
export class DeepLinkHandler {
  private siriManager: SiriShortcutsManager;
  private googleManager: GoogleAssistantManager;

  constructor() {
    this.siriManager = new SiriShortcutsManager();
    this.googleManager = new GoogleAssistantManager();
  }

  /**
   * Parse and handle incoming deep link
   */
  async handleDeepLink(url: string): Promise<AssistantResponse> {
    const parsed = Linking.parse(url);
    
    // forgeai://action?type=generate&prompt=hello
    if (parsed.hostname === 'action') {
      const { type, prompt, text, language, intent } = parsed.queryParams || {};

      switch (type) {
        case 'generate':
          return this.siriManager.handleShortcut('ask-forge', prompt as string);
        
        case 'summarize':
          return this.siriManager.handleShortcut('summarize-text', text as string);
        
        case 'translate':
          const shortcut = PREDEFINED_SHORTCUTS.find(s => s.id === 'translate-text');
          if (shortcut) {
            shortcut.parameters = { targetLanguage: language || 'Spanish' };
          }
          return this.siriManager.handleShortcut('translate-text', text as string);
        
        case 'google':
          return this.googleManager.handleAction(intent as string, parsed.queryParams || {});
        
        default:
          return { text: 'Unknown action type', success: false };
      }
    }

    return { text: 'Invalid deep link', success: false };
  }

  /**
   * Get deep link URL for action
   */
  getDeepLink(action: string, params: Record<string, string> = {}): string {
    const queryString = Object.entries(params)
      .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
      .join('&');
    
    return `forgeai://action?type=${action}${queryString ? '&' + queryString : ''}`;
  }
}

// Intent Parser (for voice input)
export function parseIntent(utterance: string): IntentData {
  const lower = utterance.toLowerCase();
  
  // Summarize patterns
  if (lower.includes('summarize') || lower.includes('summary')) {
    return {
      intent: 'forge.summarize',
      parameters: { text: utterance.replace(/summarize|summary|please|can you/gi, '').trim() },
      source: 'unknown',
    };
  }
  
  // Translate patterns
  if (lower.includes('translate')) {
    const langMatch = lower.match(/to (spanish|french|german|italian|portuguese|chinese|japanese|korean)/i);
    const language = langMatch ? langMatch[1] : 'Spanish';
    return {
      intent: 'forge.translate',
      parameters: { 
        text: utterance.replace(/translate|to \w+|please|can you/gi, '').trim(),
        language,
      },
      source: 'unknown',
    };
  }
  
  // Default - general question
  return {
    intent: 'forge.ask',
    parameters: { question: utterance },
    source: 'unknown',
  };
}

// Export managers
export const siriShortcuts = new SiriShortcutsManager();
export const googleAssistant = new GoogleAssistantManager();
export const deepLinks = new DeepLinkHandler();

export default {
  SiriShortcutsManager,
  GoogleAssistantManager,
  DeepLinkHandler,
  parseIntent,
  PREDEFINED_SHORTCUTS,
  siriShortcuts,
  googleAssistant,
  deepLinks,
};
