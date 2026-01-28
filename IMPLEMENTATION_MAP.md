# Self-Improvement System - Implementation Map

## 🗺️ Component Relationships

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FORGEAI GUI                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  enhanced_window.py (MODIFIED)                              │    │
│  │    • Imports LearningTab                                   │    │
│  │    • Adds to sidebar navigation                            │    │
│  │    • Integrates into content stack                         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓ creates                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  gui/tabs/learning_tab.py (NEW)                            │    │
│  │    • Displays metrics dashboard                            │    │
│  │    • Shows training progress                               │    │
│  │    • Autonomous learning toggle                            │    │
│  │    • Manual training trigger                               │    │
│  │    • Activity log                                          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓ uses                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  gui/widgets/feedback_widget.py (NEW)                      │    │
│  │    • Thumbs up/down buttons                                │    │
│  │    • Optional text feedback                                │    │
│  │    • Visual confirmation                                   │    │
│  │    • Emits feedback_given signal                           │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓ reads metrics from
┌─────────────────────────────────────────────────────────────────────┐
│                       LEARNING ENGINE                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  core/self_improvement.py (EXISTING)                       │    │
│  │    • Quality evaluation (relevance, coherence, repetition) │    │
│  │    • Learning queue management                             │    │
│  │    • Performance metrics tracking                          │    │
│  │    • Feedback recording                                    │    │
│  │    • Knowledge graph building                              │    │
│  │    • Training data export                                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                          ↓ feeds examples to                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  learning/training_scheduler.py (NEW)                      │    │
│  │    • Monitors example accumulation                         │    │
│  │    • Checks training criteria                              │    │
│  │    • Filters by quality (>= 0.6)                           │    │
│  │    • Exports training data                                 │    │
│  │    • Tracks training history                               │    │
│  │    • [Future] Executes LoRA training                       │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↓ used by
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS LEARNING                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  core/autonomous.py (EXISTING - VERIFIED REAL)             │    │
│  │    • _reflect_on_conversations() - analyzes past chats    │    │
│  │    • _practice_response() - generates & evaluates         │    │
│  │    • _update_personality() - evolves based on feedback    │    │
│  │    • _research_topic() - web search & learning            │    │
│  │    • _build_knowledge() - connects concepts               │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                   ↑ configured by
┌─────────────────────────────────────────────────────────────────────┐
│                       CONFIGURATION                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  config/defaults.py (MODIFIED)                             │    │
│  │    • self_improvement.enabled                              │    │
│  │    • self_improvement.autonomous_learning                  │    │
│  │    • self_improvement.feedback_learning                    │    │
│  │    • self_improvement.auto_training.*                      │    │
│  │    • self_improvement.lora_config.*                        │    │
│  │    • self_improvement.storage.*                            │    │
│  │    • self_improvement.autonomous.*                         │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 File Summary

### New Files (5)
| File | Lines | Purpose |
|------|-------|---------|
| `forge_ai/gui/widgets/feedback_widget.py` | 211 | Feedback collection UI |
| `forge_ai/learning/training_scheduler.py` | 396 | Automatic training scheduler |
| `forge_ai/gui/tabs/learning_tab.py` | 536 | Metrics dashboard tab |
| `SELF_IMPROVEMENT_INTEGRATION.md` | 242 | Technical documentation |
| `SELF_IMPROVEMENT_QUICKSTART.md` | 285 | User guide |

**Total: 1,670 lines of new code + documentation**

### Modified Files (2)
| File | Changes | Purpose |
|------|---------|---------|
| `forge_ai/config/defaults.py` | +47 lines | Add self_improvement config |
| `forge_ai/gui/enhanced_window.py` | +3 lines | Integrate Learning tab |

**Total: 50 lines modified**

### Existing Files (Verified)
| File | Status | Notes |
|------|--------|-------|
| `forge_ai/core/self_improvement.py` | ✅ Complete | 804 lines, production-ready |
| `forge_ai/core/autonomous.py` | ✅ Real impl | Not stubs, fully functional |
| `forge_ai/gui/tabs/chat_tab.py` | ✅ Integrated | Feedback already connected |

## 🔄 Data Flow

```
┌──────────────┐
│ User clicks  │
│ feedback btn │
└──────┬───────┘
       ↓
┌──────────────────────────────┐
│ FeedbackWidget               │
│ • Emits feedback_given signal│
└──────┬───────────────────────┘
       ↓
┌──────────────────────────────┐
│ Chat handler                 │
│ • Calls record_feedback()    │
└──────┬───────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ LearningEngine                      │
│ • Evaluates quality (0.0-1.0)       │
│ • Creates LearningExample           │
│ • Adds to queue (JSONL)             │
│ • Updates metrics                   │
└──────┬──────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ TrainingScheduler                   │
│ • Monitors example count            │
│ • Checks time since last training   │
│ • Filters by quality (>= 0.6)       │
│ • Triggers when criteria met        │
└──────┬──────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ [Future] LoRA Training              │
│ • Prepares dataset                  │
│ • Trains adapter                    │
│ • Saves weights                     │
└──────┬──────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ LearningTab                         │
│ • Displays updated metrics          │
│ • Shows training progress           │
│ • Logs activity                     │
└─────────────────────────────────────┘
```

## 🎛️ Configuration Options

```python
CONFIG["self_improvement"] = {
    "enabled": True,                    # Master switch
    "autonomous_learning": False,       # Background learning
    "feedback_learning": True,          # Learn from feedback
    
    "auto_training": {
        "enabled": True,                # Auto-trigger training
        "min_examples": 100,            # Examples before training
        "interval_hours": 24,           # Min time between trainings
        "min_quality_score": 0.6,       # Quality threshold
        "max_examples_per_training": 1000
    },
    
    "lora_config": {
        "rank": 8,                      # LoRA rank
        "alpha": 16,                    # LoRA alpha
        "dropout": 0.1,                 # Dropout rate
        "target_modules": ["q_proj", "v_proj"]
    },
    
    "storage": {
        "max_examples": 10000,          # Max queue size
        "max_feedback_entries": 5000,   # Max feedback log
        "cleanup_old_data_days": 30     # Auto-cleanup
    },
    
    "autonomous": {
        "interval": 300,                # Seconds between actions
        "max_actions_per_hour": 12,     # Rate limit
        "min_quality_for_learning": 0.6,
        "reflection_depth": 10,         # Conversations to analyze
        "evolution_rate": 0.02,         # Personality change rate
        "balance_threshold": 0.5
    }
}
```

## 📊 Storage Structure

```
models/<model_name>/learning/
├── learning_queue.jsonl          # All training examples
│   └── One LearningExample per line in JSON format
│
├── performance_metrics.json      # Current metrics
│   └── PerformanceMetrics object
│
├── knowledge_graph.json          # Topic connections
│   └── Dict of topic -> [related topics]
│
├── feedback_log.jsonl           # User feedback history
│   └── One feedback entry per line
│
├── training_state.json          # Scheduler state
│   └── Last training time, etc.
│
└── training_data_*.txt          # Exported training data
    └── Human-readable Q&A pairs with metadata
```

## 🧪 Testing

Run the integration test:
```bash
python test_self_improvement.py
```

Tests verify:
- ✅ Learning engine records feedback
- ✅ Training scheduler monitors examples
- ✅ Configuration loads correctly
- ✅ Components integrate properly

## 🚀 Next Steps

1. **Test with running GUI**:
   ```bash
   python run.py --gui
   ```
   - Navigate to Learning tab
   - Give feedback on responses
   - Watch metrics update

2. **Collect feedback data**:
   - Use the AI naturally
   - Rate responses (good/bad)
   - System accumulates examples

3. **Trigger training** (when ready):
   - Wait for 100+ examples
   - Click "Train Now" in Learning tab
   - OR wait for auto-training (24 hours)

4. **Integrate LoRA training** (future):
   - Connect to `forge_ai/core/training.py`
   - Implement actual training execution
   - Load trained adapters

---

**Status**: ✅ Implementation complete and ready for testing!
