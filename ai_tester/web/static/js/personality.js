/**
 * Personality Dashboard JavaScript
 */

// State
let personalityData = null;
let aiEvolutionMode = true;

// Elements
const evolutionToggle = document.getElementById('ai-evolution-toggle');
const modeTitle = document.getElementById('mode-title');
const modeDescription = document.getElementById('mode-description');
const saveBtn = document.getElementById('save-btn');
const resetBtn = document.getElementById('reset-btn');
const explainBtn = document.getElementById('explain-btn');
const explanationBox = document.getElementById('explanation-box');
const explanationText = document.getElementById('explanation-text');

// Trait sliders
const traits = ['humor', 'formality', 'verbosity', 'creativity'];

// Load personality data
async function loadPersonality() {
    try {
        const response = await fetch('/api/personality');
        const data = await response.json();
        
        if (data.success) {
            personalityData = data.personality;
            updateUI();
        }
    } catch (error) {
        console.error('Error loading personality:', error);
    }
}

// Update UI with current personality data
function updateUI() {
    if (!personalityData) return;
    
    // Update sliders and bars
    traits.forEach(trait => {
        const value = personalityData[trait] || 0.5;
        const slider = document.getElementById(`${trait}-slider`);
        const valueSpan = document.getElementById(`${trait}-value`);
        const evolvedBar = document.getElementById(`${trait}-evolved`);
        const userBar = document.getElementById(`${trait}-user`);
        
        if (slider) {
            slider.value = value * 100;
            valueSpan.textContent = value.toFixed(1);
            
            if (personalityData.user_controlled) {
                evolvedBar.style.width = '0%';
                userBar.style.width = (value * 100) + '%';
            } else {
                evolvedBar.style.width = (value * 100) + '%';
                userBar.style.width = '0%';
            }
        }
    });
    
    // Update evolution mode
    aiEvolutionMode = !personalityData.user_controlled;
    evolutionToggle.checked = aiEvolutionMode;
    updateModeDisplay();
    
    // Update interests
    if (personalityData.interests) {
        const interestsDiv = document.getElementById('ai-interests');
        interestsDiv.innerHTML = '';
        personalityData.interests.forEach(interest => {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.textContent = interest;
            interestsDiv.appendChild(tag);
        });
    }
    
    // Update dislikes
    if (personalityData.dislikes) {
        const dislikesDiv = document.getElementById('ai-dislikes');
        dislikesDiv.innerHTML = '';
        personalityData.dislikes.forEach(dislike => {
            const tag = document.createElement('span');
            tag.className = 'tag tag-dislike';
            tag.textContent = dislike;
            dislikesDiv.appendChild(tag);
        });
    }
}

// Update mode display
function updateModeDisplay() {
    if (aiEvolutionMode) {
        modeTitle.textContent = 'Let AI Evolve';
        modeDescription.textContent = 'AI will develop its own personality naturally';
        traits.forEach(trait => {
            document.getElementById(`${trait}-slider`).disabled = true;
        });
    } else {
        modeTitle.textContent = "I'll Control";
        modeDescription.textContent = 'You set the personality traits manually';
        traits.forEach(trait => {
            document.getElementById(`${trait}-slider`).disabled = false;
        });
    }
}

// Save personality
async function savePersonality() {
    const settings = {
        user_controlled: !aiEvolutionMode
    };
    
    // Only include trait values if user controlled
    if (!aiEvolutionMode) {
        traits.forEach(trait => {
            const slider = document.getElementById(`${trait}-slider`);
            settings[trait] = slider.value / 100;
        });
    }
    
    try {
        const response = await fetch('/api/personality', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings)
        });
        
        const data = await response.json();
        
        if (data.success) {
            personalityData = data.personality;
            updateUI();
            showNotification('Personality saved successfully!', 'success');
        }
    } catch (error) {
        console.error('Error saving personality:', error);
        showNotification('Error saving personality', 'error');
    }
}

// Reset personality
async function resetPersonality() {
    if (!confirm('Reset personality to defaults?')) return;
    
    try {
        const response = await fetch('/api/personality/reset', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            personalityData = data.personality;
            updateUI();
            showNotification('Personality reset successfully!', 'success');
        }
    } catch (error) {
        console.error('Error resetting personality:', error);
        showNotification('Error resetting personality', 'error');
    }
}

// Explain personality
async function explainPersonality() {
    try {
        const response = await fetch('/api/ai/explain', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: 'personality'})
        });
        
        const data = await response.json();
        
        if (data.success) {
            explanationText.textContent = data.explanation;
            explanationBox.style.display = 'block';
        }
    } catch (error) {
        console.error('Error getting explanation:', error);
    }
}

// Show notification
function showNotification(message, type) {
    // Simple alert for now - could be enhanced with toast notifications
    console.log(`[${type}] ${message}`);
    alert(message);
}

// Event listeners
evolutionToggle.addEventListener('change', () => {
    aiEvolutionMode = evolutionToggle.checked;
    updateModeDisplay();
});

// Update trait values in real-time
traits.forEach(trait => {
    const slider = document.getElementById(`${trait}-slider`);
    const valueSpan = document.getElementById(`${trait}-value`);
    const userBar = document.getElementById(`${trait}-user`);
    
    slider.addEventListener('input', () => {
        const value = slider.value / 100;
        valueSpan.textContent = value.toFixed(1);
        
        if (!aiEvolutionMode) {
            userBar.style.width = slider.value + '%';
        }
    });
});

saveBtn.addEventListener('click', savePersonality);
resetBtn.addEventListener('click', resetPersonality);
explainBtn.addEventListener('click', explainPersonality);

// Load on page load
loadPersonality();
