/* Enigma Engine - Web UI Logic */

const API = '/api';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const AppState = {
    modelLoaded: false,
    modelInfo: null,
    loadingModel: false,
    profiles: [],
    activeProfile: null,
    config: {},
    sending: false,
};

// ---------------------------------------------------------------------------
// DOM Helpers
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function addMessage(role, content) {
    const area = $('#chat-messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    // Basic markdown: code blocks and inline code
    let html = escapeHtml(content);
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    div.innerHTML = html;
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
}

function addSystemMessage(text) {
    addMessage('system', text);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function showTyping() {
    const area = $('#chat-messages');
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.id = 'typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    area.appendChild(div);
    area.scrollTop = area.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing');
    if (el) el.remove();
}

function setStatus(text, state) {
    const dot = $('#status-dot');
    const label = $('#status-label');
    dot.className = `status-dot ${state}`;
    label.textContent = text;
}

// ---------------------------------------------------------------------------
// API Calls
// ---------------------------------------------------------------------------

async function apiGet(path) {
    try {
        const res = await fetch(`${API}${path}`);
        if (!res.ok) {
            const text = await res.text();
            try { return JSON.parse(text); } catch { return { error: `HTTP ${res.status}: ${text.slice(0, 200)}` }; }
        }
        return res.json();
    } catch (err) {
        return { error: `Network error: ${err.message}` };
    }
}

async function apiPost(path, body) {
    try {
        const res = await fetch(`${API}${path}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const text = await res.text();
            try { return JSON.parse(text); } catch { return { error: `HTTP ${res.status}: ${text.slice(0, 200)}` }; }
        }
        return res.json();
    } catch (err) {
        return { error: `Network error: ${err.message}` };
    }
}

async function apiDelete(path) {
    try {
        const res = await fetch(`${API}${path}`, { method: 'DELETE' });
        if (!res.ok) {
            const text = await res.text();
            try { return JSON.parse(text); } catch { return { error: `HTTP ${res.status}: ${text.slice(0, 200)}` }; }
        }
        return res.json();
    } catch (err) {
        return { error: `Network error: ${err.message}` };
    }
}

// ---------------------------------------------------------------------------
// Core Actions
// ---------------------------------------------------------------------------

async function loadModels() {
    const data = await apiGet('/models');
    const list = $('#model-list');
    list.innerHTML = '';

    if (data.models.length === 0) {
        list.innerHTML = '<li style="color: var(--text-muted)">No models found</li>';
        return;
    }

    data.models.forEach(m => {
        const li = document.createElement('li');
        li.textContent = m.name;
        li.title = `${m.filename} (${m.size_mb} MB)`;
        li.dataset.path = m.path;
        li.onclick = () => loadModel(m.path, m.name);
        list.appendChild(li);
    });

    // Check if one is already loaded
    const status = await apiGet('/models/status');
    if (status.loaded) {
        AppState.modelLoaded = true;
        AppState.modelInfo = status.model;
        setStatus('Model loaded', 'online');
        updateModelInfo(status.model);
    }
}

async function loadModel(path, name) {
    // Prevent double-clicks
    if (AppState.loadingModel) return;
    AppState.loadingModel = true;

    setStatus('Loading model...', 'loading');
    addSystemMessage(`Loading model: ${name}...`);

    // Disable all model list items during load
    $$('#model-list li').forEach(li => { li.style.pointerEvents = 'none'; li.style.opacity = '0.5'; });

    try {
        const data = await apiPost('/models/load', { path });
        if (data.status === 'ok') {
            AppState.modelLoaded = true;
            AppState.modelInfo = data.model;
            setStatus('Model loaded', 'online');
            addSystemMessage(`Model loaded: ${(data.model.parameters || 0).toLocaleString()} parameters on ${data.model.device}`);
            updateModelInfo(data.model);
            showUnloadBtn(true);

            // Highlight active model in sidebar
            $$('#model-list li').forEach(li => {
                li.classList.toggle('active', li.dataset.path === path);
            });
        } else {
            setStatus('Load failed', 'offline');
            addSystemMessage(`Failed to load model: ${data.error || 'unknown error'}`);
        }
    } catch (err) {
        setStatus('Load failed', 'offline');
        addSystemMessage(`Error: ${err.message}`);
    }

    // Re-enable model list
    $$('#model-list li').forEach(li => { li.style.pointerEvents = ''; li.style.opacity = ''; });
    AppState.loadingModel = false;
}

async function sendMessage() {
    const textarea = $('#chat-input');
    const message = textarea.value.trim();
    if (!message || AppState.sending) return;

    if (!AppState.modelLoaded) {
        addSystemMessage('No model loaded. Select a model from the sidebar first.');
        return;
    }

    AppState.sending = true;
    textarea.value = '';
    textarea.style.height = 'auto';
    $('#send-btn').disabled = true;

    addMessage('user', message);
    showTyping();

    try {
        const data = await apiPost('/chat', {
            message,
            ...AppState.config,
        });

        hideTyping();

        if (data.error) {
            addSystemMessage(`Error: ${data.error}`);
        } else {
            addMessage('assistant', data.message);
        }
    } catch (err) {
        hideTyping();
        addSystemMessage(`Network error: ${err.message}`);
    }

    AppState.sending = false;
    $('#send-btn').disabled = false;
    textarea.focus();
}

async function clearHistory() {
    await apiDelete('/history');
    $('#chat-messages').innerHTML = '';
    addSystemMessage('Chat history cleared.');
}

async function unloadModel() {
    const data = await apiPost('/models/unload', {});
    if (data.status === 'ok') {
        AppState.modelLoaded = false;
        AppState.modelInfo = null;
        setStatus('No model loaded', 'offline');
        addSystemMessage('Model unloaded.');
        showUnloadBtn(false);

        // Clear model highlight
        $$('#model-list li').forEach(li => li.classList.remove('active'));

        // Reset model info panel
        const el = $('#model-info');
        if (el) el.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No model loaded</p>';
    }
}

function showUnloadBtn(visible) {
    const btn = $('#unload-btn');
    if (btn) btn.style.display = visible ? 'inline-block' : 'none';
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

async function loadProfiles() {
    const data = await apiGet('/profiles');
    const list = $('#profile-list');
    list.innerHTML = '';

    AppState.profiles = data.profiles;

    data.profiles.forEach(p => {
        const li = document.createElement('li');
        li.textContent = p.name;
        li.title = p.description;
        li.dataset.id = p.id;
        li.onclick = () => activateProfile(p.id);
        if (data.active === p.id) li.classList.add('active');
        list.appendChild(li);
    });
}

async function activateProfile(id) {
    const data = await apiPost(`/profiles/${id}/activate`, {});
    if (data.status === 'ok') {
        AppState.activeProfile = id;
        addSystemMessage(`Profile activated: ${id}`);
        $$('#profile-list li').forEach(li => {
            li.classList.toggle('active', li.dataset.id === id);
        });
        // Refresh config display
        await loadConfig();
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

async function loadConfig() {
    const data = await apiGet('/config');
    AppState.config = data;
    updateConfigUI(data);
}

function updateConfigUI(config) {
    setNumberInput('temperature', config.temperature, 0, 2, 0.1);
    setNumberInput('top_p', config.top_p, 0, 1, 0.05);
    setNumberInput('top_k', config.top_k, 1, 200, 1);
    setNumberInput('max_tokens', config.max_tokens, 16, 4096, 16);
    setNumberInput('repetition_penalty', config.repetition_penalty, 1.0, 2.0, 0.05);
}

function setNumberInput(name, value, min, max, step) {
    const input = document.getElementById(`cfg-${name}`);
    if (!input) return;
    input.min = min;
    input.max = max;
    input.step = step;
    input.value = value;
}

function onConfigChange(name, value) {
    const num = parseFloat(value);
    if (isNaN(num)) return;

    // Clamp to min/max
    const input = document.getElementById(`cfg-${name}`);
    const min = parseFloat(input?.min);
    const max = parseFloat(input?.max);
    const step = parseFloat(input?.step) || 1;
    const clamped = Math.min(Math.max(num, isNaN(min) ? 0 : min), isNaN(max) ? num : max);

    // Round to step precision
    const decimals = (step.toString().split('.')[1] || '').length;
    const rounded = parseFloat(clamped.toFixed(decimals));

    // Update input to show clamped value
    input.value = rounded;
    AppState.config[name] = rounded;

    // Debounce server update
    clearTimeout(onConfigChange._timer);
    onConfigChange._timer = setTimeout(async () => {
        await apiPost('/config', { [name]: rounded });
    }, 500);
}

// ---------------------------------------------------------------------------
// System Info
// ---------------------------------------------------------------------------

async function loadSystemInfo() {
    const data = await apiGet('/system');
    updateSystemPanel(data);
}

function updateModelInfo(model) {
    const el = $('#model-info');
    if (!el || !model) return;
    el.innerHTML = `
        <div class="info-row"><span class="label">Parameters</span><span class="value">${(model.parameters || 0).toLocaleString()}</span></div>
        <div class="info-row"><span class="label">Device</span><span class="value">${model.device || 'cpu'}</span></div>
        <div class="info-row"><span class="label">Path</span><span class="value" title="${model.path || ''}">${(model.path || '').split(/[/\\]/).pop()}</span></div>
    `;
}

function updateSystemPanel(info) {
    const el = $('#system-info');
    if (!el) return;
    el.innerHTML = `
        <div class="info-row"><span class="label">Python</span><span class="value">${info.python_version}</span></div>
        <div class="info-row"><span class="label">PyTorch</span><span class="value">${info.torch_version || 'N/A'}</span></div>
        <div class="info-row"><span class="label">Device</span><span class="value">${info.device}</span></div>
        <div class="info-row"><span class="label">GPU</span><span class="value">${info.gpu_name || 'None'}</span></div>
        <div class="info-row"><span class="label">VRAM</span><span class="value">${info.vram_gb ? info.vram_gb + ' GB' : 'N/A'}</span></div>
        <div class="info-row"><span class="label">RAM</span><span class="value">${info.ram_gb ? info.ram_gb + ' GB' : 'N/A'}</span></div>
        <div class="info-row"><span class="label">CPU Cores</span><span class="value">${info.cpu_count || 'N/A'}</span></div>
    `;
}

// ---------------------------------------------------------------------------
// Right Panel Tabs
// ---------------------------------------------------------------------------

function switchTab(tabName) {
    $$('.panel-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
    $$('.tab-panel').forEach(p => p.style.display = p.id === `tab-${tabName}` ? 'block' : 'none');
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // Load data
    loadModels();
    loadProfiles();
    loadConfig();
    loadSystemInfo();

    // Chat input
    const textarea = $('#chat-input');
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    textarea.addEventListener('input', () => {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
    });

    // Send button
    $('#send-btn').onclick = sendMessage;

    // Clear button
    const clearBtn = $('#clear-btn');
    if (clearBtn) clearBtn.onclick = clearHistory;

    // Unload button
    const unloadBtn = $('#unload-btn');
    if (unloadBtn) unloadBtn.onclick = unloadModel;

    // Panel tabs
    $$('.panel-tab').forEach(tab => {
        tab.onclick = () => switchTab(tab.dataset.tab);
    });

    // Welcome message
    addSystemMessage('Welcome to Enigma Engine. Select a model from the sidebar to start chatting.');
});
