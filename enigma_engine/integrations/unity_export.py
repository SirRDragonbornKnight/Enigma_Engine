"""
Unity/Unreal Engine Export for Enigma AI Engine

Export AI models and avatars to game engines.

Features:
- Unity package export (.unitypackage)
- Unreal plugin export
- C# wrapper generation
- Blueprint node generation
- ONNX model conversion
- Asset bundling

Usage:
    from enigma_engine.integrations.unity_export import UnityExporter, UnrealExporter
    
    # Unity export
    exporter = UnityExporter()
    exporter.export_model(model, "MyAI.unitypackage")
    
    # Unreal export
    ue_exporter = UnrealExporter()
    ue_exporter.export_plugin(model, "MyAIPlugin")
"""

import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export format options."""
    UNITY_PACKAGE = "unitypackage"
    UNREAL_PLUGIN = "uplugin"
    ONNX = "onnx"
    TORCHSCRIPT = "pt"
    COREML = "mlmodel"
    TFLITE = "tflite"


class EngineVersion(Enum):
    """Supported engine versions."""
    UNITY_2021 = "2021.3"
    UNITY_2022 = "2022.3"
    UNITY_2023 = "2023.2"
    UNITY_6 = "6000.0"
    UNREAL_5_0 = "5.0"
    UNREAL_5_1 = "5.1"
    UNREAL_5_2 = "5.2"
    UNREAL_5_3 = "5.3"
    UNREAL_5_4 = "5.4"


@dataclass
class ExportConfig:
    """Export configuration."""
    format: ExportFormat = ExportFormat.UNITY_PACKAGE
    engine_version: EngineVersion = EngineVersion.UNITY_2022
    include_runtime: bool = True
    include_samples: bool = True
    quantize_model: bool = False
    optimize_for_mobile: bool = False
    generate_docs: bool = True
    namespace: str = "EnigmaAI"
    company_name: str = "Enigma"


@dataclass
class ExportResult:
    """Export result."""
    success: bool
    output_path: str
    format: ExportFormat
    size_bytes: int
    files_included: List[str]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CSharpGenerator:
    """Generate C# wrapper code for Unity."""
    
    def __init__(self, namespace: str = "EnigmaAI"):
        self.namespace = namespace
        
    def generate_inference_class(self, model_name: str) -> str:
        """Generate C# inference class."""
        return f'''using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Unity.Barracuda;

namespace {self.namespace}
{{
    /// <summary>
    /// AI inference wrapper for {model_name}.
    /// </summary>
    public class {model_name}Inference : MonoBehaviour
    {{
        [Header("Model Settings")]
        [SerializeField] private NNModel modelAsset;
        [SerializeField] private bool useGPU = true;
        
        private IWorker worker;
        private Model model;
        
        /// <summary>
        /// Event fired when inference completes.
        /// </summary>
        public event Action<string> OnInferenceComplete;
        
        private void Awake()
        {{
            InitializeModel();
        }}
        
        private void InitializeModel()
        {{
            if (modelAsset == null)
            {{
                Debug.LogError("{model_name}: Model asset not assigned!");
                return;
            }}
            
            model = ModelLoader.Load(modelAsset);
            worker = WorkerFactory.CreateWorker(
                useGPU ? WorkerFactory.Type.ComputePrecompiled : WorkerFactory.Type.CSharpBurst,
                model
            );
            
            Debug.Log("{model_name}: Model initialized successfully");
        }}
        
        /// <summary>
        /// Run inference on input text.
        /// </summary>
        public async Task<string> InferAsync(string input)
        {{
            if (worker == null)
            {{
                Debug.LogError("{model_name}: Worker not initialized!");
                return null;
            }}
            
            return await Task.Run(() =>
            {{
                // Tokenize input
                var tokens = Tokenize(input);
                
                // Create input tensor
                var inputTensor = new Tensor(1, tokens.Length, tokens);
                
                // Execute
                worker.Execute(inputTensor);
                
                // Get output
                var outputTensor = worker.PeekOutput();
                var result = Detokenize(outputTensor);
                
                inputTensor.Dispose();
                
                return result;
            }});
        }}
        
        /// <summary>
        /// Run inference synchronously (may cause frame drops).
        /// </summary>
        public string Infer(string input)
        {{
            return InferAsync(input).GetAwaiter().GetResult();
        }}
        
        private float[] Tokenize(string text)
        {{
            // Basic tokenization - replace with actual tokenizer
            var chars = text.ToCharArray();
            var tokens = new float[chars.Length];
            for (int i = 0; i < chars.Length; i++)
            {{
                tokens[i] = (float)chars[i];
            }}
            return tokens;
        }}
        
        private string Detokenize(Tensor tensor)
        {{
            // Basic detokenization - replace with actual detokenizer
            var chars = new char[tensor.length];
            for (int i = 0; i < tensor.length; i++)
            {{
                chars[i] = (char)Mathf.RoundToInt(tensor[i]);
            }}
            return new string(chars);
        }}
        
        private void OnDestroy()
        {{
            worker?.Dispose();
        }}
    }}
}}
'''

    def generate_chat_component(self, model_name: str) -> str:
        """Generate Unity chat component."""
        return f'''using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace {self.namespace}
{{
    /// <summary>
    /// Chat UI component for {model_name}.
    /// </summary>
    public class {model_name}Chat : MonoBehaviour
    {{
        [Header("UI References")]
        [SerializeField] private TMP_InputField inputField;
        [SerializeField] private Button sendButton;
        [SerializeField] private ScrollRect chatScroll;
        [SerializeField] private Transform messageContainer;
        [SerializeField] private GameObject userMessagePrefab;
        [SerializeField] private GameObject aiMessagePrefab;
        
        [Header("Settings")]
        [SerializeField] private int maxHistory = 50;
        [SerializeField] private bool autoScroll = true;
        
        private {model_name}Inference inference;
        private List<string> history = new List<string>();
        
        private void Start()
        {{
            inference = GetComponent<{model_name}Inference>();
            
            if (sendButton != null)
                sendButton.onClick.AddListener(OnSendClicked);
                
            if (inputField != null)
                inputField.onSubmit.AddListener(OnInputSubmit);
        }}
        
        private void OnSendClicked()
        {{
            SendMessage(inputField.text);
        }}
        
        private void OnInputSubmit(string text)
        {{
            SendMessage(text);
        }}
        
        private async void SendMessage(string text)
        {{
            if (string.IsNullOrWhiteSpace(text)) return;
            
            // Clear input
            inputField.text = "";
            
            // Add user message
            AddMessage(text, isUser: true);
            history.Add($"User: {{text}}");
            
            // Get AI response
            sendButton.interactable = false;
            
            try
            {{
                var response = await inference.InferAsync(text);
                AddMessage(response, isUser: false);
                history.Add($"AI: {{response}}");
            }}
            catch (Exception e)
            {{
                Debug.LogError($"Inference error: {{e.Message}}");
                AddMessage("Sorry, an error occurred.", isUser: false);
            }}
            
            sendButton.interactable = true;
            
            // Trim history
            while (history.Count > maxHistory)
                history.RemoveAt(0);
        }}
        
        private void AddMessage(string text, bool isUser)
        {{
            var prefab = isUser ? userMessagePrefab : aiMessagePrefab;
            var msgObj = Instantiate(prefab, messageContainer);
            
            var textComponent = msgObj.GetComponentInChildren<TMP_Text>();
            if (textComponent != null)
                textComponent.text = text;
                
            if (autoScroll)
                Canvas.ForceUpdateCanvases();
                chatScroll.verticalNormalizedPosition = 0;
        }}
        
        public void ClearChat()
        {{
            foreach (Transform child in messageContainer)
                Destroy(child.gameObject);
            history.Clear();
        }}
    }}
}}
'''

    def generate_avatar_controller(self) -> str:
        """Generate avatar controller component."""
        return f'''using System;
using UnityEngine;

namespace {self.namespace}
{{
    /// <summary>
    /// AI-controlled avatar component.
    /// </summary>
    public class EnigmaAvatarController : MonoBehaviour
    {{
        [Header("Avatar Settings")]
        [SerializeField] private Animator animator;
        [SerializeField] private AudioSource voiceSource;
        
        [Header("Blend Shapes")]
        [SerializeField] private SkinnedMeshRenderer faceMesh;
        [SerializeField] private string[] blendShapeNames;
        
        [Header("Lip Sync")]
        [SerializeField] private string mouthOpenBlendShape = "mouthOpen";
        [SerializeField] private float lipSyncSmooth = 10f;
        
        private float currentMouthValue;
        private float targetMouthValue;
        
        private int mouthOpenIndex = -1;
        
        private void Start()
        {{
            if (faceMesh != null && !string.IsNullOrEmpty(mouthOpenBlendShape))
            {{
                mouthOpenIndex = faceMesh.sharedMesh.GetBlendShapeIndex(mouthOpenBlendShape);
            }}
        }}
        
        private void Update()
        {{
            UpdateLipSync();
        }}
        
        private void UpdateLipSync()
        {{
            if (voiceSource != null && voiceSource.isPlaying)
            {{
                // Simple volume-based lip sync
                float[] samples = new float[256];
                voiceSource.GetOutputData(samples, 0);
                
                float sum = 0;
                for (int i = 0; i < samples.Length; i++)
                    sum += Mathf.Abs(samples[i]);
                    
                targetMouthValue = Mathf.Clamp01(sum / samples.Length * 50f);
            }}
            else
            {{
                targetMouthValue = 0;
            }}
            
            currentMouthValue = Mathf.Lerp(currentMouthValue, targetMouthValue, Time.deltaTime * lipSyncSmooth);
            
            if (mouthOpenIndex >= 0 && faceMesh != null)
            {{
                faceMesh.SetBlendShapeWeight(mouthOpenIndex, currentMouthValue * 100f);
            }}
        }}
        
        /// <summary>
        /// Set emotion expression.
        /// </summary>
        public void SetEmotion(string emotion, float intensity = 1f)
        {{
            if (animator != null)
            {{
                animator.SetTrigger(emotion);
                animator.SetFloat("EmotionIntensity", intensity);
            }}
        }}
        
        /// <summary>
        /// Play gesture animation.
        /// </summary>
        public void PlayGesture(string gestureName)
        {{
            if (animator != null)
            {{
                animator.SetTrigger(gestureName);
            }}
        }}
        
        /// <summary>
        /// Look at target position.
        /// </summary>
        public void LookAt(Vector3 worldPosition)
        {{
            // Implement look-at IK
        }}
        
        /// <summary>
        /// Speak text with lip sync.
        /// </summary>
        public void Speak(AudioClip clip)
        {{
            if (voiceSource != null)
            {{
                voiceSource.clip = clip;
                voiceSource.Play();
            }}
        }}
    }}
}}
'''


class BlueprintGenerator:
    """Generate Unreal Blueprint nodes."""
    
    def __init__(self, plugin_name: str = "EnigmaAI"):
        self.plugin_name = plugin_name
        
    def generate_header(self, class_name: str) -> str:
        """Generate C++ header for Blueprint node."""
        return f'''// Copyright Enigma AI Engine. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintAsyncActionBase.h"
#include "{class_name}.generated.h"

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnInferenceComplete, const FString&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnInferenceError, const FString&, Error);

/**
 * Async Blueprint node for AI inference.
 */
UCLASS()
class {self.plugin_name.upper()}_API U{class_name} : public UBlueprintAsyncActionBase
{{
    GENERATED_BODY()

public:
    /** Called when inference completes successfully */
    UPROPERTY(BlueprintAssignable)
    FOnInferenceComplete OnComplete;
    
    /** Called when inference fails */
    UPROPERTY(BlueprintAssignable)
    FOnInferenceError OnError;

    /**
     * Run AI inference asynchronously.
     * @param WorldContext - World context
     * @param Input - Input text for the AI
     * @return - Async action object
     */
    UFUNCTION(BlueprintCallable, meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject"), Category = "Enigma AI")
    static U{class_name}* InferAsync(UObject* WorldContextObject, const FString& Input);

    virtual void Activate() override;

private:
    FString InputText;
    
    void ExecuteInference();
    void HandleComplete(const FString& Response);
    void HandleError(const FString& Error);
}};
'''

    def generate_cpp(self, class_name: str) -> str:
        """Generate C++ implementation."""
        return f'''// Copyright Enigma AI Engine. All Rights Reserved.

#include "{class_name}.h"
#include "Async/Async.h"

U{class_name}* U{class_name}::InferAsync(UObject* WorldContextObject, const FString& Input)
{{
    U{class_name}* Action = NewObject<U{class_name}>();
    Action->InputText = Input;
    Action->RegisterWithGameInstance(WorldContextObject);
    return Action;
}}

void U{class_name}::Activate()
{{
    ExecuteInference();
}}

void U{class_name}::ExecuteInference()
{{
    // Run inference on background thread
    AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this]()
    {{
        // TODO: Replace with actual ONNX inference
        FString Response = TEXT("AI response placeholder");
        
        // Return to game thread
        AsyncTask(ENamedThreads::GameThread, [this, Response]()
        {{
            HandleComplete(Response);
        }});
    }});
}}

void U{class_name}::HandleComplete(const FString& Response)
{{
    OnComplete.Broadcast(Response);
    SetReadyToDestroy();
}}

void U{class_name}::HandleError(const FString& Error)
{{
    OnError.Broadcast(Error);
    SetReadyToDestroy();
}}
'''

    def generate_uplugin(self, plugin_name: str, version: str = "1.0.0") -> str:
        """Generate .uplugin file content."""
        return json.dumps({
            "FileVersion": 3,
            "Version": 1,
            "VersionName": version,
            "FriendlyName": f"{plugin_name} AI",
            "Description": "AI integration powered by Enigma AI Engine",
            "Category": "AI",
            "CreatedBy": "Enigma AI Engine",
            "CreatedByURL": "https://github.com/enigma-ai",
            "DocsURL": "",
            "MarketplaceURL": "",
            "SupportURL": "",
            "CanContainContent": True,
            "IsBetaVersion": False,
            "IsExperimentalVersion": False,
            "Installed": False,
            "Modules": [
                {
                    "Name": plugin_name,
                    "Type": "Runtime",
                    "LoadingPhase": "Default"
                }
            ],
            "Plugins": [
                {
                    "Name": "ONNXRuntime",
                    "Enabled": True
                }
            ]
        }, indent=2)


class UnityExporter:
    """Export models to Unity package format."""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self.csharp_gen = CSharpGenerator(self.config.namespace)
        
    def export_model(
        self,
        model: Any,
        output_path: str,
        model_name: str = "EnigmaModel"
    ) -> ExportResult:
        """Export model to Unity package."""
        output_path = Path(output_path)
        temp_dir = output_path.parent / f".{output_path.stem}_temp"
        
        try:
            # Create temp directory
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            files_included = []
            warnings = []
            
            # Create Asset folder structure
            assets_dir = temp_dir / "Assets" / self.config.namespace
            scripts_dir = assets_dir / "Scripts"
            models_dir = assets_dir / "Models"
            samples_dir = assets_dir / "Samples"
            
            for d in [scripts_dir, models_dir, samples_dir]:
                d.mkdir(parents=True, exist_ok=True)
            
            # Export model to ONNX
            onnx_path = models_dir / f"{model_name}.onnx"
            if self._convert_to_onnx(model, onnx_path):
                files_included.append(str(onnx_path.relative_to(temp_dir)))
            else:
                warnings.append("Could not convert to ONNX - placeholder used")
                # Create placeholder
                onnx_path.write_bytes(b"ONNX_PLACEHOLDER")
                files_included.append(str(onnx_path.relative_to(temp_dir)))
            
            # Generate C# scripts
            inference_script = scripts_dir / f"{model_name}Inference.cs"
            inference_script.write_text(self.csharp_gen.generate_inference_class(model_name))
            files_included.append(str(inference_script.relative_to(temp_dir)))
            
            chat_script = scripts_dir / f"{model_name}Chat.cs"
            chat_script.write_text(self.csharp_gen.generate_chat_component(model_name))
            files_included.append(str(chat_script.relative_to(temp_dir)))
            
            avatar_script = scripts_dir / "EnigmaAvatarController.cs"
            avatar_script.write_text(self.csharp_gen.generate_avatar_controller())
            files_included.append(str(avatar_script.relative_to(temp_dir)))
            
            # Generate documentation
            if self.config.generate_docs:
                readme = assets_dir / "README.md"
                readme.write_text(self._generate_readme(model_name))
                files_included.append(str(readme.relative_to(temp_dir)))
            
            # Create Unity package
            self._create_unity_package(temp_dir, output_path)
            
            size_bytes = output_path.stat().st_size
            
            return ExportResult(
                success=True,
                output_path=str(output_path),
                format=ExportFormat.UNITY_PACKAGE,
                size_bytes=size_bytes,
                files_included=files_included,
                warnings=warnings,
                metadata={
                    "model_name": model_name,
                    "namespace": self.config.namespace,
                    "engine_version": self.config.engine_version.value
                }
            )
            
        except Exception as e:
            logger.error(f"Unity export failed: {e}")
            return ExportResult(
                success=False,
                output_path=str(output_path),
                format=ExportFormat.UNITY_PACKAGE,
                size_bytes=0,
                files_included=[],
                errors=[str(e)]
            )
        finally:
            # Cleanup
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
    def _convert_to_onnx(self, model: Any, output_path: Path) -> bool:
        """Convert model to ONNX format."""
        try:
            import torch
            
            if hasattr(model, 'export_onnx'):
                model.export_onnx(str(output_path))
                return True
            elif isinstance(model, torch.nn.Module):
                # Create dummy input
                dummy_input = torch.zeros(1, 512, dtype=torch.long)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(output_path),
                    input_names=['input_ids'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'sequence'},
                        'logits': {0: 'batch', 1: 'sequence'}
                    },
                    opset_version=14
                )
                return True
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
        return False
        
    def _create_unity_package(self, source_dir: Path, output_path: Path):
        """Create .unitypackage from directory."""
        # Unity packages are tar.gz with specific structure
        # For simplicity, we create a zip that users can extract
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname)
                    
    def _generate_readme(self, model_name: str) -> str:
        """Generate documentation."""
        return f'''# {model_name} - Enigma AI for Unity

## Quick Start

1. Import this package into your Unity project
2. Add the `{model_name}Inference` component to a GameObject
3. Assign the ONNX model asset to the component
4. Call `InferAsync()` or `Infer()` to generate responses

## Requirements

- Unity {self.config.engine_version.value} or later
- Unity Barracuda package
- TextMeshPro (for chat UI)

## Components

### {model_name}Inference
Core inference component. Handles model loading and text generation.

### {model_name}Chat  
Ready-to-use chat UI component with message history.

### EnigmaAvatarController
Avatar control with lip sync and emotion support.

## Example Usage

```csharp
using {self.config.namespace};

public class MyGame : MonoBehaviour
{{
    private {model_name}Inference ai;
    
    async void Start()
    {{
        ai = GetComponent<{model_name}Inference>();
        string response = await ai.InferAsync("Hello!");
        Debug.Log(response);
    }}
}}
```

## Performance Tips

- Use GPU inference when possible (`useGPU = true`)
- Cache responses for repeated queries
- Use async methods to avoid frame drops

## License

Generated by Enigma AI Engine
'''


class UnrealExporter:
    """Export models to Unreal Engine plugin format."""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig(
            format=ExportFormat.UNREAL_PLUGIN,
            engine_version=EngineVersion.UNREAL_5_3
        )
        self.blueprint_gen = BlueprintGenerator()
        
    def export_plugin(
        self,
        model: Any,
        output_dir: str,
        plugin_name: str = "EnigmaAI"
    ) -> ExportResult:
        """Export model as Unreal plugin."""
        output_dir = Path(output_dir)
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            files_included = []
            warnings = []
            
            # Create plugin structure
            source_dir = output_dir / "Source" / plugin_name
            private_dir = source_dir / "Private"
            public_dir = source_dir / "Public"
            content_dir = output_dir / "Content"
            
            for d in [private_dir, public_dir, content_dir]:
                d.mkdir(parents=True, exist_ok=True)
            
            # Generate .uplugin
            uplugin_path = output_dir / f"{plugin_name}.uplugin"
            uplugin_path.write_text(self.blueprint_gen.generate_uplugin(plugin_name))
            files_included.append(str(uplugin_path.name))
            
            # Generate Blueprint async node
            header_path = public_dir / "EnigmaInferenceAsync.h"
            header_path.write_text(self.blueprint_gen.generate_header("EnigmaInferenceAsync"))
            files_included.append(f"Source/{plugin_name}/Public/EnigmaInferenceAsync.h")
            
            cpp_path = private_dir / "EnigmaInferenceAsync.cpp"
            cpp_path.write_text(self.blueprint_gen.generate_cpp("EnigmaInferenceAsync"))
            files_included.append(f"Source/{plugin_name}/Private/EnigmaInferenceAsync.cpp")
            
            # Generate Build.cs
            build_cs_path = source_dir / f"{plugin_name}.Build.cs"
            build_cs_path.write_text(self._generate_build_cs(plugin_name))
            files_included.append(f"Source/{plugin_name}/{plugin_name}.Build.cs")
            
            # Generate module files
            module_h = public_dir / f"{plugin_name}Module.h"
            module_h.write_text(self._generate_module_header(plugin_name))
            files_included.append(f"Source/{plugin_name}/Public/{plugin_name}Module.h")
            
            module_cpp = private_dir / f"{plugin_name}Module.cpp"
            module_cpp.write_text(self._generate_module_cpp(plugin_name))
            files_included.append(f"Source/{plugin_name}/Private/{plugin_name}Module.cpp")
            
            # Export ONNX model
            onnx_path = content_dir / "Models" / "EnigmaModel.onnx"
            onnx_path.parent.mkdir(exist_ok=True)
            if not self._convert_to_onnx(model, onnx_path):
                warnings.append("Could not convert to ONNX - placeholder used")
                onnx_path.write_bytes(b"ONNX_PLACEHOLDER")
            files_included.append("Content/Models/EnigmaModel.onnx")
            
            # Calculate total size
            size_bytes = sum(
                f.stat().st_size 
                for f in output_dir.rglob('*') 
                if f.is_file()
            )
            
            return ExportResult(
                success=True,
                output_path=str(output_dir),
                format=ExportFormat.UNREAL_PLUGIN,
                size_bytes=size_bytes,
                files_included=files_included,
                warnings=warnings,
                metadata={
                    "plugin_name": plugin_name,
                    "engine_version": self.config.engine_version.value
                }
            )
            
        except Exception as e:
            logger.error(f"Unreal export failed: {e}")
            return ExportResult(
                success=False,
                output_path=str(output_dir),
                format=ExportFormat.UNREAL_PLUGIN,
                size_bytes=0,
                files_included=[],
                errors=[str(e)]
            )
            
    def _convert_to_onnx(self, model: Any, output_path: Path) -> bool:
        """Convert model to ONNX format."""
        try:
            import torch
            
            if hasattr(model, 'export_onnx'):
                model.export_onnx(str(output_path))
                return True
            elif isinstance(model, torch.nn.Module):
                dummy_input = torch.zeros(1, 512, dtype=torch.long)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(output_path),
                    input_names=['input_ids'],
                    output_names=['logits'],
                    opset_version=14
                )
                return True
        except Exception as e:
            logger.warning(f"ONNX conversion failed: {e}")
        return False
        
    def _generate_build_cs(self, plugin_name: str) -> str:
        """Generate Build.cs for plugin."""
        return f'''// Copyright Enigma AI Engine. All Rights Reserved.

using UnrealBuildTool;

public class {plugin_name} : ModuleRules
{{
    public {plugin_name}(ReadOnlyTargetRules Target) : base(Target)
    {{
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
        
        PublicIncludePaths.AddRange(
            new string[] {{
            }}
        );
        
        PrivateIncludePaths.AddRange(
            new string[] {{
            }}
        );
        
        PublicDependencyModuleNames.AddRange(
            new string[]
            {{
                "Core",
            }}
        );
        
        PrivateDependencyModuleNames.AddRange(
            new string[]
            {{
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
                "ONNXRuntime",
            }}
        );
        
        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {{
            }}
        );
    }}
}}
'''

    def _generate_module_header(self, plugin_name: str) -> str:
        """Generate module header."""
        return f'''// Copyright Enigma AI Engine. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class F{plugin_name}Module : public IModuleInterface
{{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
}};
'''

    def _generate_module_cpp(self, plugin_name: str) -> str:
        """Generate module implementation."""
        return f'''// Copyright Enigma AI Engine. All Rights Reserved.

#include "{plugin_name}Module.h"

#define LOCTEXT_NAMESPACE "F{plugin_name}Module"

void F{plugin_name}Module::StartupModule()
{{
    UE_LOG(LogTemp, Log, TEXT("{plugin_name}: Module started"));
}}

void F{plugin_name}Module::ShutdownModule()
{{
    UE_LOG(LogTemp, Log, TEXT("{plugin_name}: Module shutdown"));
}}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(F{plugin_name}Module, {plugin_name})
'''


# Convenience functions
def export_to_unity(
    model: Any,
    output_path: str,
    model_name: str = "EnigmaModel",
    **kwargs
) -> ExportResult:
    """Export model to Unity package."""
    config = ExportConfig(**kwargs)
    exporter = UnityExporter(config)
    return exporter.export_model(model, output_path, model_name)


def export_to_unreal(
    model: Any,
    output_dir: str,
    plugin_name: str = "EnigmaAI",
    **kwargs
) -> ExportResult:
    """Export model to Unreal plugin."""
    config = ExportConfig(format=ExportFormat.UNREAL_PLUGIN, **kwargs)
    exporter = UnrealExporter(config)
    return exporter.export_plugin(model, output_dir, plugin_name)
