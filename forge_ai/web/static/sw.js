/**
 * ForgeAI Service Worker
 * 
 * Provides Progressive Web App functionality:
 * - Offline support
 * - Caching of static assets
 * - Background sync (future)
 */

const CACHE_NAME = 'forgeai-v1';
const STATIC_CACHE = [
    '/',
    '/static/index.html',
    '/static/styles.css',
    '/static/app.js',
    '/static/manifest.json'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('Service Worker installing');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('Caching static assets');
                return cache.addAll(STATIC_CACHE);
            })
            .catch((error) => {
                console.error('Failed to cache static assets:', error);
            })
    );
    
    // Force the waiting service worker to become the active service worker
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker activating');
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    
    // Claim all clients
    return self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    // Skip WebSocket and API requests
    if (event.request.url.includes('/ws/') || 
        event.request.url.includes('/api/') ||
        event.request.method !== 'GET') {
        return;
    }
    
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // Return cached response if found
                if (response) {
                    return response;
                }
                
                // Otherwise fetch from network
                return fetch(event.request)
                    .then((response) => {
                        // Check if valid response
                        if (!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }
                        
                        // Clone the response
                        const responseToCache = response.clone();
                        
                        // Cache the fetched response
                        caches.open(CACHE_NAME)
                            .then((cache) => {
                                cache.put(event.request, responseToCache);
                            });
                        
                        return response;
                    })
                    .catch((error) => {
                        console.error('Fetch failed:', error);
                        
                        // Return offline page if available
                        return caches.match('/');
                    });
            })
    );
});

// Background sync for offline messages
self.addEventListener('sync', (event) => {
    console.log('Background sync:', event.tag);
    
    if (event.tag === 'sync-messages') {
        event.waitUntil(syncOfflineMessages());
    } else if (event.tag === 'sync-settings') {
        event.waitUntil(syncSettings());
    }
});

// Sync offline messages to server
async function syncOfflineMessages() {
    try {
        // Open IndexedDB for offline messages
        const db = await openOfflineDB();
        const messages = await getAllOfflineMessages(db);
        
        for (const message of messages) {
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        content: message.content,
                        timestamp: message.timestamp
                    })
                });
                
                if (response.ok) {
                    // Remove synced message from offline storage
                    await deleteOfflineMessage(db, message.id);
                    console.log('Synced offline message:', message.id);
                }
            } catch (error) {
                console.error('Failed to sync message:', message.id, error);
            }
        }
    } catch (error) {
        console.error('Background sync failed:', error);
    }
}

// Sync settings
async function syncSettings() {
    try {
        const settings = await getLocalSettings();
        if (settings) {
            await fetch('/api/settings', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ settings })
            });
            console.log('Settings synced');
        }
    } catch (error) {
        console.error('Settings sync failed:', error);
    }
}

// IndexedDB helpers for offline storage
function openOfflineDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('ForgeAI_Offline', 1);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('messages')) {
                db.createObjectStore('messages', { keyPath: 'id', autoIncrement: true });
            }
            if (!db.objectStoreNames.contains('settings')) {
                db.createObjectStore('settings', { keyPath: 'key' });
            }
        };
    });
}

function getAllOfflineMessages(db) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('messages', 'readonly');
        const store = tx.objectStore('messages');
        const request = store.getAll();
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result || []);
    });
}

function deleteOfflineMessage(db, id) {
    return new Promise((resolve, reject) => {
        const tx = db.transaction('messages', 'readwrite');
        const store = tx.objectStore('messages');
        const request = store.delete(id);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
    });
}

function getLocalSettings() {
    return new Promise((resolve) => {
        try {
            const settings = localStorage.getItem('forgeai_settings');
            resolve(settings ? JSON.parse(settings) : null);
        } catch (e) {
            resolve(null);
        }
    });
}

// Push notifications with proper data handling
self.addEventListener('push', (event) => {
    console.log('Push notification received');
    
    let notificationData = {
        title: 'ForgeAI',
        body: 'New message from ForgeAI',
        icon: '/static/icons/icon-192.png',
        badge: '/static/icons/badge.png',
        vibrate: [200, 100, 200],
        tag: 'forge-notification',
        requireInteraction: false,
        data: { url: '/' }
    };
    
    // Parse push data if available
    if (event.data) {
        try {
            const pushData = event.data.json();
            notificationData = {
                ...notificationData,
                title: pushData.title || notificationData.title,
                body: pushData.body || pushData.message || notificationData.body,
                data: { 
                    url: pushData.url || '/',
                    action: pushData.action,
                    payload: pushData.payload
                }
            };
        } catch (e) {
            // Plain text fallback
            notificationData.body = event.data.text() || notificationData.body;
        }
    }
    
    event.waitUntil(
        self.registration.showNotification(notificationData.title, {
            body: notificationData.body,
            icon: notificationData.icon,
            badge: notificationData.badge,
            vibrate: notificationData.vibrate,
            tag: notificationData.tag,
            requireInteraction: notificationData.requireInteraction,
            data: notificationData.data,
            actions: [
                { action: 'open', title: 'Open' },
                { action: 'dismiss', title: 'Dismiss' }
            ]
        })
    );
});

// Notification click with action handling
self.addEventListener('notificationclick', (event) => {
    console.log('Notification clicked:', event.action);
    
    event.notification.close();
    
    if (event.action === 'dismiss') {
        return;
    }
    
    const url = event.notification.data?.url || '/';
    
    event.waitUntil(
        clients.matchAll({ type: 'window', includeUncontrolled: true })
            .then((windowClients) => {
                // Focus existing window if available
                for (const client of windowClients) {
                    if (client.url.includes(self.location.origin) && 'focus' in client) {
                        return client.focus();
                    }
                }
                // Open new window
                return clients.openWindow(url);
            })
    );
});
