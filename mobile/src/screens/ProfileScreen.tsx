/**
 * Profile Screen
 * 
 * User profile management, authentication, and account settings.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Image,
  Platform,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as ImagePicker from 'expo-image-picker';

// Types
interface UserProfile {
  id: string;
  username: string;
  email: string;
  avatarUri?: string;
  createdAt: number;
  preferences: UserPreferences;
  stats: UserStats;
}

interface UserPreferences {
  defaultModel: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  preferredLanguage: string;
  notifications: boolean;
  dataCollection: boolean;
}

interface UserStats {
  totalMessages: number;
  totalConversations: number;
  favoriteTopics: string[];
  usageMinutes: number;
  imagesGenerated: number;
}

interface ProfileScreenProps {
  darkMode?: boolean;
  serverUrl?: string;
}

const STORAGE_KEY = '@enigma_user_profile';

const DEFAULT_PROFILE: UserProfile = {
  id: '',
  username: 'User',
  email: '',
  createdAt: Date.now(),
  preferences: {
    defaultModel: 'enigma-small',
    temperature: 0.7,
    maxTokens: 256,
    systemPrompt: 'You are a helpful AI assistant.',
    preferredLanguage: 'en',
    notifications: true,
    dataCollection: false,
  },
  stats: {
    totalMessages: 0,
    totalConversations: 0,
    favoriteTopics: [],
    usageMinutes: 0,
    imagesGenerated: 0,
  },
};

const ProfileScreen: React.FC<ProfileScreenProps> = ({ darkMode = false, serverUrl }) => {
  const [profile, setProfile] = useState<UserProfile>(DEFAULT_PROFILE);
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [editedProfile, setEditedProfile] = useState<UserProfile>(DEFAULT_PROFILE);

  // Load profile on mount
  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      const stored = await AsyncStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        setProfile(parsed);
        setEditedProfile(parsed);
      }
    } catch (error) {
      console.error('Failed to load profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveProfile = async () => {
    setIsSaving(true);
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(editedProfile));
      setProfile(editedProfile);
      setIsEditing(false);
      
      // Sync with server if connected
      if (serverUrl) {
        try {
          await fetch(`${serverUrl}/v1/user/profile`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(editedProfile),
          });
        } catch {
          // Offline - will sync later
        }
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to save profile');
    } finally {
      setIsSaving(false);
    }
  };

  const pickAvatar = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission Required', 'Please allow access to your photo library');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setEditedProfile(prev => ({
        ...prev,
        avatarUri: result.assets[0].uri,
      }));
    }
  };

  const resetProfile = () => {
    Alert.alert(
      'Reset Profile',
      'This will clear all your profile data. Are you sure?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.removeItem(STORAGE_KEY);
            setProfile(DEFAULT_PROFILE);
            setEditedProfile(DEFAULT_PROFILE);
          },
        },
      ]
    );
  };

  const exportData = async () => {
    try {
      const data = JSON.stringify(profile, null, 2);
      // In a real app, would use Share API or file system
      Alert.alert('Profile Data', data.substring(0, 500) + '...');
    } catch (error) {
      Alert.alert('Error', 'Failed to export data');
    }
  };

  const styles = createStyles(darkMode);

  if (isLoading) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator size="large" color={darkMode ? '#fff' : '#007AFF'} />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Avatar and Name */}
      <View style={styles.header}>
        <TouchableOpacity onPress={isEditing ? pickAvatar : undefined}>
          {profile.avatarUri ? (
            <Image source={{ uri: profile.avatarUri }} style={styles.avatar} />
          ) : (
            <View style={[styles.avatar, styles.avatarPlaceholder]}>
              <Text style={styles.avatarText}>
                {profile.username.charAt(0).toUpperCase()}
              </Text>
            </View>
          )}
          {isEditing && (
            <View style={styles.editAvatarBadge}>
              <Text style={styles.editAvatarText}>Edit</Text>
            </View>
          )}
        </TouchableOpacity>

        {isEditing ? (
          <TextInput
            style={[styles.usernameInput, darkMode && styles.darkInput]}
            value={editedProfile.username}
            onChangeText={text =>
              setEditedProfile(prev => ({ ...prev, username: text }))
            }
            placeholder="Username"
            placeholderTextColor="#888"
          />
        ) : (
          <Text style={[styles.username, darkMode && styles.darkText]}>
            {profile.username}
          </Text>
        )}

        {isEditing ? (
          <TextInput
            style={[styles.emailInput, darkMode && styles.darkInput]}
            value={editedProfile.email}
            onChangeText={text =>
              setEditedProfile(prev => ({ ...prev, email: text }))
            }
            placeholder="Email"
            keyboardType="email-address"
            placeholderTextColor="#888"
          />
        ) : (
          profile.email && (
            <Text style={[styles.email, darkMode && styles.darkSubtext]}>
              {profile.email}
            </Text>
          )
        )}
      </View>

      {/* Stats */}
      <View style={styles.statsSection}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          USAGE STATISTICS
        </Text>
        <View style={styles.statsGrid}>
          <StatCard
            label="Messages"
            value={profile.stats.totalMessages}
            darkMode={darkMode}
          />
          <StatCard
            label="Conversations"
            value={profile.stats.totalConversations}
            darkMode={darkMode}
          />
          <StatCard
            label="Minutes"
            value={profile.stats.usageMinutes}
            darkMode={darkMode}
          />
          <StatCard
            label="Images"
            value={profile.stats.imagesGenerated}
            darkMode={darkMode}
          />
        </View>
      </View>

      {/* AI Preferences */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          AI PREFERENCES
        </Text>

        <View style={[styles.card, darkMode && styles.darkCard]}>
          <View style={styles.settingRow}>
            <Text style={[styles.label, darkMode && styles.darkText]}>
              Default Model
            </Text>
            {isEditing ? (
              <TextInput
                style={[styles.valueInput, darkMode && styles.darkInput]}
                value={editedProfile.preferences.defaultModel}
                onChangeText={text =>
                  setEditedProfile(prev => ({
                    ...prev,
                    preferences: { ...prev.preferences, defaultModel: text },
                  }))
                }
              />
            ) : (
              <Text style={[styles.value, darkMode && styles.darkSubtext]}>
                {profile.preferences.defaultModel}
              </Text>
            )}
          </View>

          <View style={styles.settingRow}>
            <Text style={[styles.label, darkMode && styles.darkText]}>
              Temperature
            </Text>
            {isEditing ? (
              <TextInput
                style={[styles.valueInput, darkMode && styles.darkInput]}
                value={editedProfile.preferences.temperature.toString()}
                onChangeText={text =>
                  setEditedProfile(prev => ({
                    ...prev,
                    preferences: {
                      ...prev.preferences,
                      temperature: parseFloat(text) || 0.7,
                    },
                  }))
                }
                keyboardType="decimal-pad"
              />
            ) : (
              <Text style={[styles.value, darkMode && styles.darkSubtext]}>
                {profile.preferences.temperature}
              </Text>
            )}
          </View>

          <View style={styles.settingRow}>
            <Text style={[styles.label, darkMode && styles.darkText]}>
              Max Tokens
            </Text>
            {isEditing ? (
              <TextInput
                style={[styles.valueInput, darkMode && styles.darkInput]}
                value={editedProfile.preferences.maxTokens.toString()}
                onChangeText={text =>
                  setEditedProfile(prev => ({
                    ...prev,
                    preferences: {
                      ...prev.preferences,
                      maxTokens: parseInt(text) || 256,
                    },
                  }))
                }
                keyboardType="number-pad"
              />
            ) : (
              <Text style={[styles.value, darkMode && styles.darkSubtext]}>
                {profile.preferences.maxTokens}
              </Text>
            )}
          </View>
        </View>
      </View>

      {/* System Prompt */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          SYSTEM PROMPT
        </Text>
        <View style={[styles.card, darkMode && styles.darkCard]}>
          {isEditing ? (
            <TextInput
              style={[styles.systemPromptInput, darkMode && styles.darkInput]}
              value={editedProfile.preferences.systemPrompt}
              onChangeText={text =>
                setEditedProfile(prev => ({
                  ...prev,
                  preferences: { ...prev.preferences, systemPrompt: text },
                }))
              }
              multiline
              numberOfLines={4}
              placeholder="Enter custom system prompt..."
              placeholderTextColor="#888"
            />
          ) : (
            <Text style={[styles.systemPromptText, darkMode && styles.darkText]}>
              {profile.preferences.systemPrompt}
            </Text>
          )}
        </View>
      </View>

      {/* Privacy */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkMode && styles.darkSubtext]}>
          PRIVACY
        </Text>
        <View style={[styles.card, darkMode && styles.darkCard]}>
          <View style={styles.settingRow}>
            <Text style={[styles.label, darkMode && styles.darkText]}>
              Data Collection
            </Text>
            <TouchableOpacity
              style={[
                styles.toggle,
                editedProfile.preferences.dataCollection && styles.toggleActive,
              ]}
              onPress={() =>
                isEditing &&
                setEditedProfile(prev => ({
                  ...prev,
                  preferences: {
                    ...prev.preferences,
                    dataCollection: !prev.preferences.dataCollection,
                  },
                }))
              }
              disabled={!isEditing}
            >
              <View
                style={[
                  styles.toggleKnob,
                  editedProfile.preferences.dataCollection &&
                    styles.toggleKnobActive,
                ]}
              />
            </TouchableOpacity>
          </View>

          <View style={styles.settingRow}>
            <Text style={[styles.label, darkMode && styles.darkText]}>
              Notifications
            </Text>
            <TouchableOpacity
              style={[
                styles.toggle,
                editedProfile.preferences.notifications && styles.toggleActive,
              ]}
              onPress={() =>
                isEditing &&
                setEditedProfile(prev => ({
                  ...prev,
                  preferences: {
                    ...prev.preferences,
                    notifications: !prev.preferences.notifications,
                  },
                }))
              }
              disabled={!isEditing}
            >
              <View
                style={[
                  styles.toggleKnob,
                  editedProfile.preferences.notifications && styles.toggleKnobActive,
                ]}
              />
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* Actions */}
      <View style={styles.actions}>
        {isEditing ? (
          <>
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={() => {
                setEditedProfile(profile);
                setIsEditing(false);
              }}
            >
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.saveButton, isSaving && styles.disabledButton]}
              onPress={saveProfile}
              disabled={isSaving}
            >
              {isSaving ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.saveButtonText}>Save Changes</Text>
              )}
            </TouchableOpacity>
          </>
        ) : (
          <TouchableOpacity
            style={styles.editButton}
            onPress={() => setIsEditing(true)}
          >
            <Text style={styles.editButtonText}>Edit Profile</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Danger Zone */}
      <View style={styles.dangerZone}>
        <Text style={[styles.sectionTitle, { color: '#F44336' }]}>
          DANGER ZONE
        </Text>
        <TouchableOpacity style={styles.exportButton} onPress={exportData}>
          <Text style={styles.exportButtonText}>Export My Data</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.resetButton} onPress={resetProfile}>
          <Text style={styles.resetButtonText}>Reset Profile</Text>
        </TouchableOpacity>
      </View>

      {/* Spacer */}
      <View style={{ height: 40 }} />
    </ScrollView>
  );
};

// Stat Card Component
const StatCard: React.FC<{
  label: string;
  value: number;
  darkMode: boolean;
}> = ({ label, value, darkMode }) => (
  <View style={[statStyles.card, darkMode && statStyles.darkCard]}>
    <Text style={[statStyles.value, darkMode && statStyles.darkText]}>
      {value.toLocaleString()}
    </Text>
    <Text style={[statStyles.label, darkMode && statStyles.darkLabel]}>
      {label}
    </Text>
  </View>
);

const statStyles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 16,
    margin: 4,
    borderRadius: 12,
    alignItems: 'center',
    minWidth: 80,
  },
  darkCard: {
    backgroundColor: '#2a2a2a',
  },
  value: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  darkText: {
    color: '#4da6ff',
  },
  label: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  darkLabel: {
    color: '#aaa',
  },
});

const createStyles = (darkMode: boolean) =>
  StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: darkMode ? '#1a1a1a' : '#f5f5f5',
    },
    center: {
      justifyContent: 'center',
      alignItems: 'center',
    },
    header: {
      alignItems: 'center',
      paddingVertical: 32,
      paddingHorizontal: 16,
    },
    avatar: {
      width: 100,
      height: 100,
      borderRadius: 50,
    },
    avatarPlaceholder: {
      backgroundColor: '#007AFF',
      justifyContent: 'center',
      alignItems: 'center',
    },
    avatarText: {
      fontSize: 40,
      fontWeight: 'bold',
      color: '#fff',
    },
    editAvatarBadge: {
      position: 'absolute',
      bottom: 0,
      right: 0,
      backgroundColor: '#007AFF',
      paddingHorizontal: 8,
      paddingVertical: 4,
      borderRadius: 10,
    },
    editAvatarText: {
      color: '#fff',
      fontSize: 12,
      fontWeight: '600',
    },
    username: {
      fontSize: 24,
      fontWeight: 'bold',
      marginTop: 16,
      color: '#000',
    },
    usernameInput: {
      fontSize: 24,
      fontWeight: 'bold',
      marginTop: 16,
      textAlign: 'center',
      borderBottomWidth: 1,
      borderBottomColor: '#007AFF',
      paddingBottom: 4,
    },
    email: {
      fontSize: 14,
      color: '#888',
      marginTop: 4,
    },
    emailInput: {
      fontSize: 14,
      marginTop: 4,
      textAlign: 'center',
      borderBottomWidth: 1,
      borderBottomColor: '#007AFF',
      paddingBottom: 4,
      minWidth: 200,
    },
    darkText: {
      color: '#fff',
    },
    darkSubtext: {
      color: '#aaa',
    },
    darkInput: {
      color: '#fff',
      borderBottomColor: '#4da6ff',
    },
    statsSection: {
      marginHorizontal: 12,
      marginBottom: 24,
    },
    statsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      justifyContent: 'space-between',
    },
    section: {
      marginHorizontal: 16,
      marginBottom: 24,
    },
    sectionTitle: {
      fontSize: 12,
      fontWeight: '600',
      color: '#888',
      marginBottom: 8,
      marginLeft: 4,
    },
    card: {
      backgroundColor: '#fff',
      borderRadius: 12,
      overflow: 'hidden',
    },
    darkCard: {
      backgroundColor: '#2a2a2a',
    },
    settingRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 16,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: '#e0e0e0',
    },
    label: {
      fontSize: 16,
      color: '#000',
    },
    value: {
      fontSize: 16,
      color: '#888',
    },
    valueInput: {
      fontSize: 16,
      textAlign: 'right',
      minWidth: 100,
      borderBottomWidth: 1,
      borderBottomColor: '#007AFF',
    },
    systemPromptInput: {
      padding: 16,
      fontSize: 14,
      minHeight: 100,
      textAlignVertical: 'top',
    },
    systemPromptText: {
      padding: 16,
      fontSize: 14,
      color: '#333',
    },
    toggle: {
      width: 50,
      height: 30,
      borderRadius: 15,
      backgroundColor: '#e0e0e0',
      padding: 2,
      justifyContent: 'center',
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
    actions: {
      flexDirection: 'row',
      marginHorizontal: 16,
      marginBottom: 24,
    },
    editButton: {
      flex: 1,
      backgroundColor: '#007AFF',
      padding: 16,
      borderRadius: 12,
      alignItems: 'center',
    },
    editButtonText: {
      color: '#fff',
      fontWeight: '600',
      fontSize: 16,
    },
    cancelButton: {
      flex: 1,
      backgroundColor: '#e0e0e0',
      padding: 16,
      borderRadius: 12,
      alignItems: 'center',
      marginRight: 8,
    },
    cancelButtonText: {
      color: '#333',
      fontWeight: '600',
      fontSize: 16,
    },
    saveButton: {
      flex: 1,
      backgroundColor: '#007AFF',
      padding: 16,
      borderRadius: 12,
      alignItems: 'center',
    },
    saveButtonText: {
      color: '#fff',
      fontWeight: '600',
      fontSize: 16,
    },
    disabledButton: {
      opacity: 0.6,
    },
    dangerZone: {
      marginHorizontal: 16,
      marginTop: 24,
    },
    exportButton: {
      backgroundColor: '#fff',
      padding: 16,
      borderRadius: 12,
      alignItems: 'center',
      marginBottom: 8,
      borderWidth: 1,
      borderColor: '#007AFF',
    },
    exportButtonText: {
      color: '#007AFF',
      fontWeight: '600',
    },
    resetButton: {
      backgroundColor: '#F44336',
      padding: 16,
      borderRadius: 12,
      alignItems: 'center',
    },
    resetButtonText: {
      color: '#fff',
      fontWeight: '600',
    },
  });

export default ProfileScreen;
