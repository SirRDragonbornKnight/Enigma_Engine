# Persona Tab UI Layout

## Overview
The Persona tab is located in the "MODEL" section of the sidebar, between "Chat" and "Scale".

## Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  AI Persona Management                                              │
├─────────────────────────────────────────────────────────────────────┤
│  Create, customize, and manage your AI personas. Copy your AI to   │
│  create variants or import personas from others.                    │
├───────────────────────┬─────────────────────────────────────────────┤
│  YOUR PERSONAS        │  PERSONA DETAILS                            │
│                       │                                             │
│  ┌─────────────────┐ │  Name: [Forge Assistant                  ] │
│  │ Forge Assistant │ │                                             │
│  │ (Current)       │ │  Response Style: [balanced    ▼]           │
│  └─────────────────┘ │                                             │
│  ┌─────────────────┐ │  Voice Profile:  [default     ▼]           │
│  │ My Copy         │ │                                             │
│  └─────────────────┘ │  Avatar Preset:  [default     ▼]           │
│  ┌─────────────────┐ │                                             │
│  │ Creative AI     │ │  ┌─────────────────────────────────────┐   │
│  └─────────────────┘ │  │ SYSTEM PROMPT                       │   │
│                       │  │                                     │   │
│  [Set as Current  ]   │  │ You are a helpful AI assistant     │   │
│                       │  │ built with ForgeAI...              │   │
│  [Copy Persona    ]   │  │                                     │   │
│                       │  └─────────────────────────────────────┘   │
│  [Delete          ]   │                                             │
│                       │  ┌─────────────────────────────────────┐   │
│  ┌───────────────────┐│  │ DESCRIPTION                         │   │
│  │ IMPORT/EXPORT     ││  │                                     │   │
│  │                   ││  │ Default ForgeAI assistant persona   │   │
│  │ [Import from File]││  │                                     │   │
│  │ [Export to File  ]││  └─────────────────────────────────────┘   │
│  │ [Load Template   ]││                                             │
│  └───────────────────┘│  [Save Changes]                             │
└───────────────────────┴─────────────────────────────────────────────┘
```

## Features Visible in UI

### Left Panel (Persona List)
- **List of Personas**: Shows all available personas
- **Current Indicator**: Active persona marked with "(Current)" and green text
- **Action Buttons**:
  - "Set as Current" (green) - Activate selected persona
  - "Copy Persona" (gray) - Create a copy
  - "Delete" (red) - Remove persona (disabled for default)

### Import/Export Section
- **Import from File**: Load a .forge-ai file
- **Export to File**: Save current persona to share
- **Load Template**: Import pre-made templates

### Right Panel (Details Editor)
- **Name**: Editable persona name
- **Response Style**: Dropdown (balanced, concise, detailed, casual)
- **Voice Profile**: Dropdown of available voice profiles
- **Avatar Preset**: Dropdown of available avatar presets
- **System Prompt**: Multi-line text editor
- **Description**: Multi-line text editor
- **Save Changes**: Button (enabled when changes are made)

## Chat Tab Integration

In the Chat tab, the header now shows:
```
[AI] small_forge    [Persona] Forge Assistant    [+New Chat] [Clear] [Save]
```

The persona indicator:
- Shows current persona name
- Green background with transparency
- Tooltip: "Current AI persona - manage in Persona tab"
- Clicking opens Persona tab (future enhancement)

## Color Scheme

Following the existing ForgeAI dark theme:
- **Primary (green)**: #a6e3a1 - Active/confirm buttons
- **Secondary (blue)**: #89b4fa - Standard/action buttons
- **Danger (red)**: #f38ba8 - Delete button
- **Disabled (gray + red dashed)**: #313244 with #f38ba8 border - Inactive buttons
- **Background**: #1e1e2e - Dark background
- **Text**: #cdd6f4 - Main text

## Interaction Flow

1. **View Personas**: See all personas in left list
2. **Select**: Click to view details in right panel
3. **Edit**: Modify name, style, prompts, etc.
4. **Save**: Click "Save Changes" to persist
5. **Copy**: Create variant with "Copy Persona"
6. **Export**: Share via "Export to File"
7. **Import**: Load others' personas via "Import from File"
8. **Switch**: Set as current to use in chat

## Template Selection Dialog

When "Load Template" is clicked:
```
┌─────────────────────────────────────┐
│  Load Template                      │
├─────────────────────────────────────┤
│  Choose a template persona:         │
│                                     │
│  ┌───────────────────────────────┐ │
│  │ Helpful Assistant             │ │
│  │ Creative Companion            │ │
│  │ Technical Expert              │ │
│  │ Casual Friend                 │ │
│  └───────────────────────────────┘ │
│                                     │
│            [OK]  [Cancel]           │
└─────────────────────────────────────┘
```

## Copy Dialog

When "Copy Persona" is clicked:
```
┌─────────────────────────────────────┐
│  Copy Persona                       │
├─────────────────────────────────────┤
│  New Name: [Forge Assistant (Copy)]│
│                                     │
│  ☐ Copy learning data               │
│     (Include training data)         │
│                                     │
│            [OK]  [Cancel]           │
└─────────────────────────────────────┘
```
