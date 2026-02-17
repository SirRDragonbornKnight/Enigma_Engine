"""Download sample humanoid GLB model for testing avatar system."""
import urllib.request
import os

# Sample humanoid GLB URLs (public domain / CC0)
SAMPLE_MODELS = {
    "robot": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/RiggedSimple/glTF-Binary/RiggedSimple.glb",
    "character": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/RiggedFigure/glTF-Binary/RiggedFigure.glb",
    "cesium_man": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/CesiumMan/glTF-Binary/CesiumMan.glb",
}

def download_sample_model(model_name: str = "character", output_dir: str = "models/avatars"):
    """Download a sample rigged GLB model for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name not in SAMPLE_MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(SAMPLE_MODELS.keys())}")
        return None
    
    url = SAMPLE_MODELS[model_name]
    output_path = os.path.join(output_dir, f"{model_name}.glb")
    
    print(f"Downloading {model_name}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def list_bones_in_glb(glb_path: str):
    """List all bones in a GLB file."""
    try:
        from enigma_engine.avatar.animation_3d_native import GLTFLoader
        
        loader = GLTFLoader()
        data = loader.load(glb_path)
        
        if not data:
            print("Failed to load GLB")
            return
        
        print(f"\nBones in {glb_path}:")
        for bone in data.get("bones", {}).keys():
            print(f"  - {bone}")
        
        return list(data.get("bones", {}).keys())
    except ImportError:
        print("GLTFLoader not available")
        return None


if __name__ == "__main__":
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else "character"
    
    # Download model
    path = download_sample_model(model)
    
    # List bones if successful
    if path:
        list_bones_in_glb(path)
