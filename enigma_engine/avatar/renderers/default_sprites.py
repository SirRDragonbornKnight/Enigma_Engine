"""
Default Avatar Sprites

Built-in SVG sprite templates that don't require external assets.
"""

import base64

# SVG sprite templates
SPRITE_TEMPLATES = {
    "idle": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - neutral -->
    <path d="M 80 115 Q 100 120 120 115" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>""",
    
    "happy": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - happy closed -->
    <path d="M 70 90 Q 80 85 90 90" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M 110 90 Q 120 85 130 90" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Mouth - big smile -->
    <path d="M 70 110 Q 100 130 130 110" stroke="{accent_color}" stroke-width="4" fill="none" stroke-linecap="round"/>
    
    <!-- Blush marks -->
    <circle cx="60" cy="105" r="8" fill="{secondary_color}" opacity="0.5"/>
    <circle cx="140" cy="105" r="8" fill="{secondary_color}" opacity="0.5"/>
</svg>""",
    
    "thinking": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - looking up -->
    <ellipse cx="75" cy="85" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="115" cy="85" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils - looking up -->
    <circle cx="75" cy="82" r="4" fill="#1e1e2e"/>
    <circle cx="115" cy="82" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - thoughtful -->
    <line x1="85" y1="115" x2="115" y2="115" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Thought bubble -->
    <circle cx="140" cy="60" r="4" fill="{secondary_color}" opacity="0.6"/>
    <circle cx="150" cy="50" r="6" fill="{secondary_color}" opacity="0.6"/>
    <circle cx="165" cy="40" r="10" fill="{secondary_color}" opacity="0.6"/>
</svg>""",
    
    "sad": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - sad -->
    <ellipse cx="80" cy="90" rx="8" ry="14" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="14" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Eyebrows - sad -->
    <path d="M 70 75 Q 80 72 90 75" stroke="{accent_color}" stroke-width="2" fill="none"/>
    <path d="M 110 75 Q 120 72 130 75" stroke="{accent_color}" stroke-width="2" fill="none"/>
    
    <!-- Mouth - frown -->
    <path d="M 80 125 Q 100 115 120 125" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>""",
    
    "surprised": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - wide open -->
    <circle cx="80" cy="90" r="12" fill="{accent_color}"/>
    <circle cx="120" cy="90" r="12" fill="{accent_color}"/>
    
    <!-- Pupils - large -->
    <circle cx="80" cy="90" r="6" fill="#1e1e2e"/>
    <circle cx="120" cy="90" r="6" fill="#1e1e2e"/>
    
    <!-- Mouth - open O -->
    <circle cx="100" cy="120" r="10" fill="{accent_color}"/>
    <circle cx="100" cy="120" r="7" fill="{primary_color}"/>
</svg>""",
    
    "confused": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - asymmetric -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <circle cx="120" cy="88" r="6" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="88" r="3" fill="#1e1e2e"/>
    
    <!-- Eyebrows - confused -->
    <path d="M 70 75 L 90 78" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
    <path d="M 110 78 L 130 75" stroke="{accent_color}" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Mouth - squiggle -->
    <path d="M 75 115 Q 85 110 95 115 Q 105 120 115 115" stroke="{accent_color}" stroke-width="2" fill="none" stroke-linecap="round"/>
    
    <!-- Question mark -->
    <text x="145" y="65" font-size="20" fill="{secondary_color}">?</text>
</svg>""",
    
    "excited": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle with glow -->
    <circle cx="100" cy="100" r="85" fill="{secondary_color}" opacity="0.3"/>
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - sparkly -->
    <circle cx="80" cy="90" r="10" fill="{accent_color}"/>
    <circle cx="120" cy="90" r="10" fill="{accent_color}"/>
    
    <!-- Pupils with highlights -->
    <circle cx="80" cy="90" r="5" fill="#1e1e2e"/>
    <circle cx="120" cy="90" r="5" fill="#1e1e2e"/>
    <circle cx="77" cy="87" r="2" fill="white"/>
    <circle cx="117" cy="87" r="2" fill="white"/>
    
    <!-- Mouth - big excited smile -->
    <path d="M 65 108 Q 100 140 135 108" stroke="{accent_color}" stroke-width="4" fill="none" stroke-linecap="round"/>
    
    <!-- Sparkles -->
    <path d="M 150 70 L 152 75 L 157 73 L 153 78 L 158 80 L 152 82 L 153 87 L 150 82 L 145 84 L 148 79 L 143 77 L 148 75 Z" fill="{secondary_color}"/>
    <path d="M 40 65 L 41 68 L 44 67 L 42 70 L 45 71 L 41 72 L 42 75 L 40 72 L 37 73 L 39 70 L 36 69 L 39 68 Z" fill="{secondary_color}"/>
</svg>""",
    
    "speaking_1": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - speaking position 1 (semi-open) -->
    <ellipse cx="100" cy="118" rx="15" ry="8" fill="{accent_color}"/>
    <ellipse cx="100" cy="116" rx="12" ry="6" fill="{primary_color}"/>
</svg>""",
    
    "speaking_2": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - speaking position 2 (more closed) -->
    <ellipse cx="100" cy="117" rx="12" ry="6" fill="{accent_color}"/>
</svg>""",

    "winking": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Left eye - winking -->
    <path d="M 70 90 Q 80 85 90 90" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Right eye - open -->
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - playful smile -->
    <path d="M 75 115 Q 100 130 125 115" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Blush on winking side -->
    <circle cx="55" cy="105" r="8" fill="{secondary_color}" opacity="0.5"/>
</svg>""",

    "sleeping": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - closed sleeping -->
    <path d="M 65 92 Q 80 88 95 92" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M 105 92 Q 120 88 135 92" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Mouth - relaxed -->
    <ellipse cx="100" cy="120" rx="8" ry="4" fill="{accent_color}" opacity="0.5"/>
    
    <!-- Z's for sleeping -->
    <text x="140" y="55" font-size="16" fill="{secondary_color}" opacity="0.8">Z</text>
    <text x="150" y="45" font-size="14" fill="{secondary_color}" opacity="0.6">z</text>
    <text x="158" y="38" font-size="12" fill="{secondary_color}" opacity="0.4">z</text>
</svg>""",

    "angry": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle - reddish tint -->
    <circle cx="100" cy="100" r="80" fill="#ef4444" opacity="0.15"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - narrowed angry -->
    <ellipse cx="80" cy="92" rx="10" ry="6" fill="{accent_color}"/>
    <ellipse cx="120" cy="92" rx="10" ry="6" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="3" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="3" fill="#1e1e2e"/>
    
    <!-- Angry eyebrows -->
    <path d="M 65 78 L 90 85" stroke="{accent_color}" stroke-width="4" stroke-linecap="round"/>
    <path d="M 135 78 L 110 85" stroke="{accent_color}" stroke-width="4" stroke-linecap="round"/>
    
    <!-- Mouth - angry frown -->
    <path d="M 75 125 Q 100 115 125 125" stroke="{accent_color}" stroke-width="4" fill="none" stroke-linecap="round"/>
</svg>""",

    "love": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle with hearts -->
    <circle cx="100" cy="100" r="80" fill="#ec4899" opacity="0.15"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Heart eyes -->
    <path d="M 68 90 C 68 82 80 82 80 90 C 80 82 92 82 92 90 C 92 102 80 110 80 110 C 80 110 68 102 68 90" fill="#ef4444"/>
    <path d="M 108 90 C 108 82 120 82 120 90 C 120 82 132 82 132 90 C 132 102 120 110 120 110 C 120 110 108 102 108 90" fill="#ef4444"/>
    
    <!-- Mouth - happy smile -->
    <path d="M 75 115 Q 100 130 125 115" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Floating hearts -->
    <path d="M 145 55 C 145 50 150 50 150 55 C 150 50 155 50 155 55 C 155 62 150 67 150 67 C 150 67 145 62 145 55" fill="#ec4899" opacity="0.7"/>
    <path d="M 38 60 C 38 56 42 56 42 60 C 42 56 46 56 46 60 C 46 65 42 69 42 69 C 42 69 38 65 38 60" fill="#ec4899" opacity="0.5"/>
</svg>""",

    "worried": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - worried wide -->
    <ellipse cx="80" cy="90" rx="9" ry="13" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="9" ry="13" fill="{accent_color}"/>
    
    <!-- Pupils - looking to the side nervously -->
    <circle cx="82" cy="91" r="4" fill="#1e1e2e"/>
    <circle cx="122" cy="91" r="4" fill="#1e1e2e"/>
    
    <!-- Worried eyebrows -->
    <path d="M 65 75 Q 80 80 95 78" stroke="{accent_color}" stroke-width="2" fill="none" stroke-linecap="round"/>
    <path d="M 105 78 Q 120 80 135 75" stroke="{accent_color}" stroke-width="2" fill="none" stroke-linecap="round"/>
    
    <!-- Mouth - worried wavy -->
    <path d="M 80 120 Q 90 115 100 120 Q 110 125 120 120" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Sweat drop -->
    <ellipse cx="145" cy="80" rx="4" ry="6" fill="#60a5fa" opacity="0.7"/>
</svg>""",

    "neutral": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes -->
    <ellipse cx="80" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="8" ry="12" fill="{accent_color}"/>
    
    <!-- Pupils -->
    <circle cx="80" cy="92" r="4" fill="#1e1e2e"/>
    <circle cx="120" cy="92" r="4" fill="#1e1e2e"/>
    
    <!-- Mouth - straight neutral line -->
    <line x1="85" y1="118" x2="115" y2="118" stroke="{accent_color}" stroke-width="3" stroke-linecap="round"/>
</svg>""",

    "friendly": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Background circle -->
    <circle cx="100" cy="100" r="80" fill="{primary_color}" opacity="0.2"/>
    
    <!-- Main body/head -->
    <circle cx="100" cy="100" r="60" fill="{primary_color}"/>
    
    <!-- Eyes - warm and friendly -->
    <ellipse cx="80" cy="90" rx="9" ry="11" fill="{accent_color}"/>
    <ellipse cx="120" cy="90" rx="9" ry="11" fill="{accent_color}"/>
    
    <!-- Pupils with highlights -->
    <circle cx="80" cy="91" r="5" fill="#1e1e2e"/>
    <circle cx="120" cy="91" r="5" fill="#1e1e2e"/>
    <circle cx="77" cy="88" r="2" fill="white"/>
    <circle cx="117" cy="88" r="2" fill="white"/>
    
    <!-- Mouth - warm smile -->
    <path d="M 75 112 Q 100 125 125 112" stroke="{accent_color}" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Subtle blush -->
    <circle cx="60" cy="105" r="7" fill="{secondary_color}" opacity="0.3"/>
    <circle cx="140" cy="105" r="7" fill="{secondary_color}" opacity="0.3"/>
</svg>""",

    # ============ ANIME GIRL STYLE ============
    "anime_idle": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Hair back -->
    <ellipse cx="100" cy="85" rx="70" ry="75" fill="{primary_color}"/>
    
    <!-- Face -->
    <ellipse cx="100" cy="100" rx="45" ry="50" fill="#ffecd2"/>
    
    <!-- Hair front/bangs -->
    <path d="M 55 70 Q 70 55 100 50 Q 130 55 145 70 L 140 90 Q 120 80 100 85 Q 80 80 60 90 Z" fill="{primary_color}"/>
    
    <!-- Side hair -->
    <path d="M 55 70 Q 45 100 50 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    <path d="M 145 70 Q 155 100 150 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    
    <!-- Big anime eyes -->
    <ellipse cx="75" cy="100" rx="15" ry="18" fill="white"/>
    <ellipse cx="125" cy="100" rx="15" ry="18" fill="white"/>
    <ellipse cx="75" cy="102" rx="12" ry="14" fill="{accent_color}"/>
    <ellipse cx="125" cy="102" rx="12" ry="14" fill="{accent_color}"/>
    <circle cx="75" cy="100" r="6" fill="#1e1e2e"/>
    <circle cx="125" cy="100" r="6" fill="#1e1e2e"/>
    <!-- Eye highlights -->
    <circle cx="70" cy="96" r="4" fill="white"/>
    <circle cx="120" cy="96" r="4" fill="white"/>
    <circle cx="78" cy="105" r="2" fill="white" opacity="0.6"/>
    <circle cx="128" cy="105" r="2" fill="white" opacity="0.6"/>
    
    <!-- Eyelashes -->
    <path d="M 60 88 Q 75 85 90 88" stroke="#1e1e2e" stroke-width="2" fill="none"/>
    <path d="M 110 88 Q 125 85 140 88" stroke="#1e1e2e" stroke-width="2" fill="none"/>
    
    <!-- Small nose -->
    <path d="M 100 110 L 100 115" stroke="#deb887" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Cute mouth -->
    <path d="M 90 125 Q 100 130 110 125" stroke="#e57373" stroke-width="2" fill="none" stroke-linecap="round"/>
    
    <!-- Blush -->
    <ellipse cx="60" cy="115" rx="10" ry="6" fill="#ffb6c1" opacity="0.5"/>
    <ellipse cx="140" cy="115" rx="10" ry="6" fill="#ffb6c1" opacity="0.5"/>
</svg>""",

    "anime_happy": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Hair back -->
    <ellipse cx="100" cy="85" rx="70" ry="75" fill="{primary_color}"/>
    
    <!-- Face -->
    <ellipse cx="100" cy="100" rx="45" ry="50" fill="#ffecd2"/>
    
    <!-- Hair front/bangs -->
    <path d="M 55 70 Q 70 55 100 50 Q 130 55 145 70 L 140 90 Q 120 80 100 85 Q 80 80 60 90 Z" fill="{primary_color}"/>
    
    <!-- Side hair -->
    <path d="M 55 70 Q 45 100 50 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    <path d="M 145 70 Q 155 100 150 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    
    <!-- Happy closed eyes (^ ^) -->
    <path d="M 60 100 Q 75 90 90 100" stroke="#1e1e2e" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M 110 100 Q 125 90 140 100" stroke="#1e1e2e" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Small nose -->
    <path d="M 100 110 L 100 115" stroke="#deb887" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Big happy smile -->
    <path d="M 80 125 Q 100 140 120 125" stroke="#e57373" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Extra blush when happy -->
    <ellipse cx="55" cy="110" rx="12" ry="7" fill="#ffb6c1" opacity="0.6"/>
    <ellipse cx="145" cy="110" rx="12" ry="7" fill="#ffb6c1" opacity="0.6"/>
    
    <!-- Sparkles -->
    <text x="30" y="60" font-size="16" fill="{secondary_color}">✦</text>
    <text x="160" y="70" font-size="12" fill="{secondary_color}">✦</text>
</svg>""",

    "anime_thinking": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Hair back -->
    <ellipse cx="100" cy="85" rx="70" ry="75" fill="{primary_color}"/>
    
    <!-- Face -->
    <ellipse cx="100" cy="100" rx="45" ry="50" fill="#ffecd2"/>
    
    <!-- Hair front/bangs -->
    <path d="M 55 70 Q 70 55 100 50 Q 130 55 145 70 L 140 90 Q 120 80 100 85 Q 80 80 60 90 Z" fill="{primary_color}"/>
    
    <!-- Side hair -->
    <path d="M 55 70 Q 45 100 50 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    <path d="M 145 70 Q 155 100 150 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    
    <!-- Thinking eyes - looking up -->
    <ellipse cx="75" cy="98" rx="15" ry="18" fill="white"/>
    <ellipse cx="125" cy="98" rx="15" ry="18" fill="white"/>
    <ellipse cx="75" cy="96" rx="12" ry="14" fill="{accent_color}"/>
    <ellipse cx="125" cy="96" rx="12" ry="14" fill="{accent_color}"/>
    <circle cx="75" cy="94" r="6" fill="#1e1e2e"/>
    <circle cx="125" cy="94" r="6" fill="#1e1e2e"/>
    <circle cx="70" cy="90" r="4" fill="white"/>
    <circle cx="120" cy="90" r="4" fill="white"/>
    
    <!-- Eyelashes -->
    <path d="M 60 86 Q 75 83 90 86" stroke="#1e1e2e" stroke-width="2" fill="none"/>
    <path d="M 110 86 Q 125 83 140 86" stroke="#1e1e2e" stroke-width="2" fill="none"/>
    
    <!-- Small nose -->
    <path d="M 100 110 L 100 115" stroke="#deb887" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Thinking mouth (small o) -->
    <ellipse cx="100" cy="128" rx="5" ry="4" fill="#e57373"/>
    
    <!-- Thought bubble -->
    <circle cx="155" cy="50" r="5" fill="{secondary_color}" opacity="0.7"/>
    <circle cx="165" cy="40" r="8" fill="{secondary_color}" opacity="0.7"/>
    <circle cx="178" cy="28" r="12" fill="{secondary_color}" opacity="0.7"/>
    
    <!-- Blush -->
    <ellipse cx="60" cy="115" rx="10" ry="6" fill="#ffb6c1" opacity="0.4"/>
    <ellipse cx="140" cy="115" rx="10" ry="6" fill="#ffb6c1" opacity="0.4"/>
</svg>""",

    "anime_surprised": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Hair back -->
    <ellipse cx="100" cy="85" rx="70" ry="75" fill="{primary_color}"/>
    
    <!-- Face -->
    <ellipse cx="100" cy="100" rx="45" ry="50" fill="#ffecd2"/>
    
    <!-- Hair front/bangs -->
    <path d="M 55 70 Q 70 55 100 50 Q 130 55 145 70 L 140 90 Q 120 80 100 85 Q 80 80 60 90 Z" fill="{primary_color}"/>
    
    <!-- Side hair -->
    <path d="M 55 70 Q 45 100 50 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    <path d="M 145 70 Q 155 100 150 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    
    <!-- Surprised big eyes -->
    <ellipse cx="75" cy="100" rx="18" ry="22" fill="white"/>
    <ellipse cx="125" cy="100" rx="18" ry="22" fill="white"/>
    <ellipse cx="75" cy="102" rx="14" ry="16" fill="{accent_color}"/>
    <ellipse cx="125" cy="102" rx="14" ry="16" fill="{accent_color}"/>
    <circle cx="75" cy="100" r="7" fill="#1e1e2e"/>
    <circle cx="125" cy="100" r="7" fill="#1e1e2e"/>
    <circle cx="70" cy="95" r="5" fill="white"/>
    <circle cx="120" cy="95" r="5" fill="white"/>
    
    <!-- Small nose -->
    <path d="M 100 110 L 100 115" stroke="#deb887" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Surprised mouth (O) -->
    <ellipse cx="100" cy="130" rx="10" ry="12" fill="#e57373"/>
    
    <!-- Blush -->
    <ellipse cx="55" cy="115" rx="12" ry="7" fill="#ffb6c1" opacity="0.5"/>
    <ellipse cx="145" cy="115" rx="12" ry="7" fill="#ffb6c1" opacity="0.5"/>
    
    <!-- Exclamation -->
    <text x="160" y="50" font-size="24" fill="{secondary_color}" font-weight="bold">!</text>
</svg>""",

    "anime_wink": """<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <!-- Hair back -->
    <ellipse cx="100" cy="85" rx="70" ry="75" fill="{primary_color}"/>
    
    <!-- Face -->
    <ellipse cx="100" cy="100" rx="45" ry="50" fill="#ffecd2"/>
    
    <!-- Hair front/bangs -->
    <path d="M 55 70 Q 70 55 100 50 Q 130 55 145 70 L 140 90 Q 120 80 100 85 Q 80 80 60 90 Z" fill="{primary_color}"/>
    
    <!-- Side hair -->
    <path d="M 55 70 Q 45 100 50 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    <path d="M 145 70 Q 155 100 150 150" stroke="{primary_color}" stroke-width="20" fill="none" stroke-linecap="round"/>
    
    <!-- Left eye - winking -->
    <path d="M 60 100 Q 75 95 90 100" stroke="#1e1e2e" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Right eye - open -->
    <ellipse cx="125" cy="100" rx="15" ry="18" fill="white"/>
    <ellipse cx="125" cy="102" rx="12" ry="14" fill="{accent_color}"/>
    <circle cx="125" cy="100" r="6" fill="#1e1e2e"/>
    <circle cx="120" cy="96" r="4" fill="white"/>
    <path d="M 110 88 Q 125 85 140 88" stroke="#1e1e2e" stroke-width="2" fill="none"/>
    
    <!-- Small nose -->
    <path d="M 100 110 L 100 115" stroke="#deb887" stroke-width="2" stroke-linecap="round"/>
    
    <!-- Playful smile -->
    <path d="M 85 125 Q 100 135 115 125" stroke="#e57373" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Blush on wink side -->
    <ellipse cx="55" cy="110" rx="12" ry="7" fill="#ffb6c1" opacity="0.6"/>
    <ellipse cx="145" cy="115" rx="10" ry="6" fill="#ffb6c1" opacity="0.4"/>
    
    <!-- Star -->
    <text x="35" y="85" font-size="14" fill="{secondary_color}">★</text>
</svg>""",
}


def generate_sprite(
    sprite_name: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
) -> str:
    """
    Generate an SVG sprite with custom colors.
    
    Args:
        sprite_name: Name of sprite template to use
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
        
    Returns:
        SVG string with colors applied
    """
    if sprite_name not in SPRITE_TEMPLATES:
        sprite_name = "idle"
    
    template = SPRITE_TEMPLATES[sprite_name]
    
    # Replace color placeholders
    svg = template.format(
        primary_color=primary_color,
        secondary_color=secondary_color,
        accent_color=accent_color
    )
    
    return svg


def generate_sprite_png(
    sprite_name: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981",
    size: int = 200
) -> bytes:
    """
    Generate a PNG sprite from SVG template.
    
    NOTE: Requires cairosvg for PNG conversion. If not available, returns SVG data.
    Install with: pip install cairosvg
    
    Args:
        sprite_name: Name of sprite template
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
        size: Output size in pixels
        
    Returns:
        PNG image data as bytes (or SVG if cairosvg unavailable)
    """
    svg = generate_sprite(sprite_name, primary_color, secondary_color, accent_color)
    
    try:
        # Try cairosvg first (best quality)
        # This is an optional dependency
        import cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        return png_data
    except ImportError:
        # cairosvg not available, return SVG data
        pass
    
    try:
        # Try PIL with svg support
        pass


        # For now, return SVG data - PIL doesn't handle SVG well without extra deps
        return svg.encode('utf-8')
    except ImportError:
        pass
    
    # Fallback: return SVG as bytes
    return svg.encode('utf-8')


def get_sprite_data_url(
    sprite_name: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
) -> str:
    """
    Get sprite as a data URL for use in HTML/CSS.
    
    Args:
        sprite_name: Name of sprite template
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
        
    Returns:
        Data URL string
    """
    svg = generate_sprite(sprite_name, primary_color, secondary_color, accent_color)
    
    # Encode as base64
    svg_bytes = svg.encode('utf-8')
    b64 = base64.b64encode(svg_bytes).decode('utf-8')
    
    return f"data:image/svg+xml;base64,{b64}"


def save_sprite(
    sprite_name: str,
    filepath: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
):
    """
    Save sprite to file.
    
    Args:
        sprite_name: Name of sprite template
        filepath: Path to save to (should end in .svg or .png)
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
    """
    from pathlib import Path
    
    path = Path(filepath)
    
    if path.suffix.lower() == '.svg':
        # Save as SVG
        svg = generate_sprite(sprite_name, primary_color, secondary_color, accent_color)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(svg)
    else:
        # Save as PNG
        png_data = generate_sprite_png(sprite_name, primary_color, secondary_color, accent_color)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(png_data)


def generate_all_sprites(
    output_dir: str,
    primary_color: str = "#6366f1",
    secondary_color: str = "#8b5cf6",
    accent_color: str = "#10b981"
):
    """
    Generate all sprite templates and save to directory.
    
    Args:
        output_dir: Directory to save sprites to
        primary_color: Primary color (hex)
        secondary_color: Secondary color (hex)
        accent_color: Accent color (hex)
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for sprite_name in SPRITE_TEMPLATES.keys():
        save_sprite(
            sprite_name,
            str(output_path / f"{sprite_name}.svg"),
            primary_color,
            secondary_color,
            accent_color
        )
