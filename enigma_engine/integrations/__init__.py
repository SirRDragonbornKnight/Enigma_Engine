"""
External Integrations

Integrations with external tools and platforms.

Provides:
- ForgeLLM: LangChain compatibility
- UnityExporter: Unity game engine export
- GameEngineBridge: Generic game engine integration
- OBSController: OBS Studio integration
"""

from .game_engine_bridge import (
    AvatarState,
    EngineMessage,
    GameEngineBridge,
    MessageType,
    Quaternion,
    Transform,
    Vector3,
)
from .langchain_adapter import (
    ForgeChatModel,
    ForgeEmbeddings,
    ForgeLLM,
    ForgeModelConfig,
    create_forge_chat_model,
    create_forge_embeddings,
    create_forge_llm,
)
from .obs_streaming import (
    OBSConfig,
    OBSController,
    OBSIntegration,
    OverlayServer,
    OverlayType,
    SceneInfo,
)
from .unity_export import (
    BlueprintGenerator,
    CSharpGenerator,
    EngineVersion,
    ExportConfig,
    ExportFormat,
    ExportResult,
    UnityExporter,
    UnrealExporter,
)

__all__ = [
    # LangChain
    "ForgeLLM",
    "ForgeChatModel",
    "ForgeEmbeddings",
    "ForgeModelConfig",
    "create_forge_llm",
    "create_forge_chat_model",
    "create_forge_embeddings",
    # Unity/Unreal
    "UnityExporter",
    "UnrealExporter",
    "ExportConfig",
    "ExportFormat",
    "ExportResult",
    "EngineVersion",
    "CSharpGenerator",
    "BlueprintGenerator",
    # Game engines
    "GameEngineBridge",
    "MessageType",
    "Vector3",
    "Quaternion",
    "Transform",
    "AvatarState",
    "EngineMessage",
    # OBS
    "OBSController",
    "OBSIntegration",
    "OBSConfig",
    "OverlayServer",
    "OverlayType",
    "SceneInfo",
]
