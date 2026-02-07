/**
 * Memory Viewer JavaScript
 */

// State
let memories = [];
let conversations = [];

// Elements
const memoriesList = document.getElementById('memories-list');
const conversationsList = document.getElementById('conversations-list');
const memoryCount = document.getElementById('memory-count');
const conversationCount = document.getElementById('conversation-count');
const importantCount = document.getElementById('important-count');
const addMemoryBtn = document.getElementById('add-memory-btn');
const newMemoryText = document.getElementById('new-memory-text');
const memoryImportance = document.getElementById('memory-importance');
const importanceValue = document.getElementById('importance-value');
const sortBy = document.getElementById('sort-by');
const refreshBtn = document.getElementById('refresh-btn');

// Load memories
async function loadMemories() {
    try {
        const response = await fetch('/api/memory');
        const data = await response.json();
        
        if (data.success) {
            memories = data.memories || [];
            conversations = data.conversations || [];
            updateUI();
        }
    } catch (error) {
        console.error('Error loading memories:', error);
    }
}

// Update UI
function updateUI() {
    // Update stats
    memoryCount.textContent = memories.length;
    conversationCount.textContent = conversations.length;
    
    const important = memories.filter(m => m.importance >= 0.7).length;
    importantCount.textContent = important;
    
    // Sort memories
    sortMemories();
    
    // Render memories
    renderMemories();
    
    // Render conversations
    renderConversations();
    
    // Load AI preferences
    loadAIPreferences();
}

// Sort memories
function sortMemories() {
    const sortType = sortBy.value;
    
    memories.sort((a, b) => {
        switch (sortType) {
            case 'date':
                return new Date(b.timestamp) - new Date(a.timestamp);
            case 'importance':
                return b.importance - a.importance;
            case 'source':
                return a.source.localeCompare(b.source);
            default:
                return 0;
        }
    });
}

// Render memories
function renderMemories() {
    if (memories.length === 0) {
        memoriesList.innerHTML = '<p class="no-memories">No memories yet. Add your first memory above!</p>';
        return;
    }
    
    memoriesList.innerHTML = '';
    
    memories.forEach(memory => {
        const card = document.createElement('div');
        card.className = 'memory-card';
        
        const importance = Math.round(memory.importance * 10);
        const stars = '⭐'.repeat(Math.min(importance, 5));
        
        card.innerHTML = `
            <div class="memory-header">
                <span class="memory-source">${memory.source}</span>
                <span class="memory-importance">${stars}</span>
            </div>
            <div class="memory-text">${escapeHtml(memory.text)}</div>
            <div class="memory-footer">
                <span class="memory-date">${formatDate(memory.timestamp)}</span>
                <button class="btn-delete" data-id="${memory.id}">🗑️ Delete</button>
            </div>
        `;
        
        memoriesList.appendChild(card);
    });
    
    // Add delete listeners
    document.querySelectorAll('.btn-delete').forEach(btn => {
        btn.addEventListener('click', () => deleteMemory(btn.dataset.id));
    });
}

// Render conversations
function renderConversations() {
    if (conversations.length === 0) {
        conversationsList.innerHTML = '<p class="no-conversations">No conversations saved yet.</p>';
        return;
    }
    
    conversationsList.innerHTML = '';
    
    conversations.forEach(conv => {
        const card = document.createElement('div');
        card.className = 'conversation-card';
        
        card.innerHTML = `
            <div class="conversation-name">${escapeHtml(conv.name)}</div>
            <div class="conversation-info">
                <span>${conv.message_count} messages</span>
                <span>${formatDate(conv.saved_at * 1000)}</span>
            </div>
        `;
        
        conversationsList.appendChild(card);
    });
}

// Add memory
async function addMemory() {
    const text = newMemoryText.value.trim();
    const importance = parseFloat(memoryImportance.value);
    
    if (!text) {
        alert('Please enter memory text');
        return;
    }
    
    try {
        const response = await fetch('/api/memory', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                text: text,
                importance: importance
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            memories.push(data.memory);
            updateUI();
            newMemoryText.value = '';
            memoryImportance.value = 0.5;
            importanceValue.textContent = '0.5';
        }
    } catch (error) {
        console.error('Error adding memory:', error);
        alert('Error adding memory');
    }
}

// Delete memory
async function deleteMemory(memoryId) {
    if (!confirm('Delete this memory?')) return;
    
    try {
        const response = await fetch(`/api/memory/${memoryId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            memories = memories.filter(m => m.id !== memoryId);
            updateUI();
        }
    } catch (error) {
        console.error('Error deleting memory:', error);
        alert('Error deleting memory');
    }
}

// Load AI preferences (topics)
async function loadAIPreferences() {
    try {
        const response = await fetch('/api/ai/preferences');
        const data = await response.json();
        
        if (data.success) {
            // Update favorite topics
            const favTopics = document.getElementById('favorite-topics');
            if (favTopics) {
                favTopics.innerHTML = '';
                data.preferences.favorite_topics.forEach(topic => {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.textContent = topic;
                    favTopics.appendChild(tag);
                });
            }
            
            // Avoided topics (placeholder)
            const avoidTopics = document.getElementById('avoided-topics');
            if (avoidTopics) {
                avoidTopics.innerHTML = '<span class="tag tag-avoid">Spam</span>';
            }
        }
    } catch (error) {
        console.error('Error loading AI preferences:', error);
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Event listeners
memoryImportance.addEventListener('input', () => {
    importanceValue.textContent = memoryImportance.value;
});

addMemoryBtn.addEventListener('click', addMemory);
sortBy.addEventListener('change', updateUI);
refreshBtn.addEventListener('click', loadMemories);

// Load on page load
loadMemories();
