/**
 * Message Bubble Component
 * 
 * Displays chat messages with proper styling for user/assistant.
 */

import React from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
} from 'react-native';
import * as Clipboard from 'expo-clipboard';
import * as Haptics from 'expo-haptics';

interface MessageBubbleProps {
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp?: number;
  darkMode?: boolean;
  showCopy?: boolean;
  onLongPress?: () => void;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({
  content,
  role,
  timestamp,
  darkMode = false,
  showCopy = true,
  onLongPress,
}) => {
  const isUser = role === 'user';
  const isSystem = role === 'system';

  const handleCopy = async () => {
    await Clipboard.setStringAsync(content);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const handleLongPress = () => {
    if (onLongPress) {
      onLongPress();
    } else if (showCopy) {
      handleCopy();
    }
  };

  const formatTime = (ts: number): string => {
    const date = new Date(ts);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (isSystem) {
    return (
      <View style={styles.systemContainer}>
        <Text style={[styles.systemText, darkMode && styles.darkSystemText]}>
          {content}
        </Text>
      </View>
    );
  }

  return (
    <TouchableOpacity
      style={[
        styles.container,
        isUser ? styles.userContainer : styles.assistantContainer,
      ]}
      onLongPress={handleLongPress}
      activeOpacity={0.9}
    >
      <View
        style={[
          styles.bubble,
          isUser
            ? styles.userBubble
            : [styles.assistantBubble, darkMode && styles.darkAssistantBubble],
        ]}
      >
        <Text
          style={[
            styles.content,
            isUser
              ? styles.userContent
              : [styles.assistantContent, darkMode && styles.darkText],
          ]}
          selectable
        >
          {content}
        </Text>

        {timestamp && (
          <Text
            style={[
              styles.timestamp,
              isUser
                ? styles.userTimestamp
                : [styles.assistantTimestamp, darkMode && styles.darkTimestamp],
            ]}
          >
            {formatTime(timestamp)}
          </Text>
        )}
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 12,
    marginVertical: 4,
  },
  userContainer: {
    alignItems: 'flex-end',
  },
  assistantContainer: {
    alignItems: 'flex-start',
  },
  bubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 18,
  },
  userBubble: {
    backgroundColor: '#007AFF',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: '#fff',
    borderBottomLeftRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  darkAssistantBubble: {
    backgroundColor: '#2a2a2a',
  },
  content: {
    fontSize: 16,
    lineHeight: 22,
  },
  userContent: {
    color: '#fff',
  },
  assistantContent: {
    color: '#000',
  },
  darkText: {
    color: '#fff',
  },
  timestamp: {
    fontSize: 10,
    marginTop: 4,
    alignSelf: 'flex-end',
  },
  userTimestamp: {
    color: 'rgba(255, 255, 255, 0.7)',
  },
  assistantTimestamp: {
    color: '#888',
  },
  darkTimestamp: {
    color: '#666',
  },
  systemContainer: {
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  systemText: {
    fontSize: 12,
    color: '#888',
    fontStyle: 'italic',
  },
  darkSystemText: {
    color: '#666',
  },
});

export default MessageBubble;
