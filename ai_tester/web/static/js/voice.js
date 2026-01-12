/**
 * Voice Studio JavaScript
 */

// State
let voiceSettings = {
    profile: 'default',
    pitch: 1.0,
    speed: 1.0,
    volume: 1.0,
    effects: []
};

// Elements
const profileSelect = document.getElementById('voice-profile');
const pitchSlider = document.getElementById('pitch-slider');
const speedSlider = document.getElementById('speed-slider');
const volumeSlider = document.getElementById('volume-slider');
const pitchValue = document.getElementById('pitch-value');
const speedValue = document.getElementById('speed-value');
const volumeValue = document.getElementById('volume-value');
const previewBtn = document.getElementById('preview-btn');
const previewText = document.getElementById('preview-text');
const previewStatus = document.getElementById('preview-status');
const saveBtn = document.getElementById('save-voice-btn');
const resetBtn = document.getElementById('reset-voice-btn');
const letAIChooseBtn = document.getElementById('let-ai-choose');

// Current profile display
const currentProfile = document.getElementById('current-profile');
const currentPitch = document.getElementById('current-pitch');
const currentSpeed = document.getElementById('current-speed');
const currentEffects = document.getElementById('current-effects');

// Load voice settings
async function loadVoiceSettings() {
    try {
        const response = await fetch('/api/voice');
        const data = await response.json();
        
        if (data.success) {
            voiceSettings = data.voice;
            updateUI();
        }
    } catch (error) {
        console.error('Error loading voice settings:', error);
    }
}

// Load voice profiles
async function loadVoiceProfiles() {
    try {
        const response = await fetch('/api/voice/profiles');
        const data = await response.json();
        
        if (data.success) {
            profileSelect.innerHTML = '';
            data.profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name} - ${profile.description}`;
                profileSelect.appendChild(option);
            });
            
            // Set current profile
            profileSelect.value = voiceSettings.profile;
        }
    } catch (error) {
        console.error('Error loading voice profiles:', error);
    }
}

// Update UI
function updateUI() {
    pitchSlider.value = voiceSettings.pitch;
    speedSlider.value = voiceSettings.speed;
    volumeSlider.value = voiceSettings.volume;
    
    pitchValue.textContent = voiceSettings.pitch.toFixed(1);
    speedValue.textContent = voiceSettings.speed.toFixed(1);
    volumeValue.textContent = voiceSettings.volume.toFixed(1);
    
    profileSelect.value = voiceSettings.profile;
    
    // Update effects checkboxes
    const effectCheckboxes = document.querySelectorAll('.effect-checkbox input');
    effectCheckboxes.forEach(checkbox => {
        checkbox.checked = voiceSettings.effects.includes(checkbox.value);
    });
    
    updateCurrentDisplay();
}

// Update current voice display
function updateCurrentDisplay() {
    currentProfile.textContent = profileSelect.options[profileSelect.selectedIndex].text.split(' -')[0];
    currentPitch.textContent = voiceSettings.pitch.toFixed(1);
    currentSpeed.textContent = voiceSettings.speed.toFixed(1);
    
    if (voiceSettings.effects.length > 0) {
        currentEffects.textContent = voiceSettings.effects.join(', ');
    } else {
        currentEffects.textContent = 'None';
    }
}

// Preview voice
async function previewVoice() {
    const text = previewText.value.trim();
    
    if (!text) {
        previewStatus.textContent = 'Please enter text to preview';
        previewStatus.className = 'preview-status error';
        return;
    }
    
    previewBtn.disabled = true;
    previewBtn.innerHTML = '<span class="btn-icon">⏳</span> Generating...';
    previewStatus.textContent = 'Generating preview...';
    previewStatus.className = 'preview-status info';
    
    try {
        const response = await fetch('/api/voice/preview', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                text: text,
                settings: voiceSettings
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            previewStatus.textContent = data.note || 'Preview generated!';
            previewStatus.className = 'preview-status success';
        }
    } catch (error) {
        console.error('Error previewing voice:', error);
        previewStatus.textContent = 'Error generating preview';
        previewStatus.className = 'preview-status error';
    } finally {
        previewBtn.disabled = false;
        previewBtn.innerHTML = '<span class="btn-icon">▶️</span> Preview Voice';
    }
}

// Save voice settings
async function saveVoiceSettings() {
    try {
        const response = await fetch('/api/voice', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(voiceSettings)
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Voice settings saved successfully!');
            updateCurrentDisplay();
        }
    } catch (error) {
        console.error('Error saving voice settings:', error);
        alert('Error saving voice settings');
    }
}

// Reset voice settings
async function resetVoiceSettings() {
    if (!confirm('Reset voice settings to default?')) return;
    
    voiceSettings = {
        profile: 'default',
        pitch: 1.0,
        speed: 1.0,
        volume: 1.0,
        effects: []
    };
    
    updateUI();
    await saveVoiceSettings();
}

// Let AI choose voice
function letAIChooseVoice() {
    // Simple algorithm - could be enhanced with personality integration
    const profiles = ['default', 'glados', 'jarvis', 'robot'];
    const randomProfile = profiles[Math.floor(Math.random() * profiles.length)];
    
    voiceSettings.profile = randomProfile;
    voiceSettings.pitch = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
    voiceSettings.speed = 0.9 + Math.random() * 0.2; // 0.9 to 1.1
    
    updateUI();
    alert(`AI chose the ${randomProfile} profile for you!`);
}

// Event listeners
profileSelect.addEventListener('change', () => {
    voiceSettings.profile = profileSelect.value;
    updateCurrentDisplay();
});

pitchSlider.addEventListener('input', () => {
    voiceSettings.pitch = parseFloat(pitchSlider.value);
    pitchValue.textContent = voiceSettings.pitch.toFixed(1);
    updateCurrentDisplay();
});

speedSlider.addEventListener('input', () => {
    voiceSettings.speed = parseFloat(speedSlider.value);
    speedValue.textContent = voiceSettings.speed.toFixed(1);
    updateCurrentDisplay();
});

volumeSlider.addEventListener('input', () => {
    voiceSettings.volume = parseFloat(volumeSlider.value);
    volumeValue.textContent = voiceSettings.volume.toFixed(1);
});

// Effect checkboxes
document.querySelectorAll('.effect-checkbox input').forEach(checkbox => {
    checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
            if (!voiceSettings.effects.includes(checkbox.value)) {
                voiceSettings.effects.push(checkbox.value);
            }
        } else {
            voiceSettings.effects = voiceSettings.effects.filter(e => e !== checkbox.value);
        }
        updateCurrentDisplay();
    });
});

previewBtn.addEventListener('click', previewVoice);
saveBtn.addEventListener('click', saveVoiceSettings);
resetBtn.addEventListener('click', resetVoiceSettings);
letAIChooseBtn.addEventListener('click', letAIChooseVoice);

// Load on page load
loadVoiceSettings();
loadVoiceProfiles();
