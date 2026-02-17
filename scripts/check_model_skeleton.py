"""Check model for skeleton/skin data."""
import json
from pathlib import Path

def check_gltf(path):
    """Check a GLTF file for skeleton data."""
    with open(path) as f:
        data = json.load(f)
    
    print(f"=== {path.name} ===")
    print(f"  Nodes: {len(data.get('nodes', []))}")
    print(f"  Meshes: {len(data.get('meshes', []))}")
    print(f"  Materials: {len(data.get('materials', []))}")
    print(f"  Skins: {len(data.get('skins', []))}")
    print(f"  Animations: {len(data.get('animations', []))}")
    
    if 'skins' in data and data['skins']:
        print("\n=== SKELETON DATA ===")
        for i, skin in enumerate(data['skins']):
            joints = skin.get('joints', [])
            print(f"Skin {i}: {len(joints)} joints")
            # Show joint names
            for j_idx in joints[:15]:
                if j_idx < len(data['nodes']):
                    node = data['nodes'][j_idx]
                    print(f"  Joint: {node.get('name', f'node_{j_idx}')}")
            if len(joints) > 15:
                print(f"  ... and {len(joints)-15} more")
        return True
    else:
        print("\nNO SKINS - checking for bone-like nodes...")
        bone_nodes = []
        for i, node in enumerate(data.get('nodes', [])):
            name = node.get('name', '')
            # Check for transform hierarchy that might be bones
            has_children = 'children' in node
            has_transform = 'rotation' in node or 'translation' in node
            if has_children or has_transform:
                if any(x in name.lower() for x in ['bone', 'arm', 'head', 'spine', 'joint', 'root', 'hip']):
                    bone_nodes.append(name)
                    print(f"  Potential bone: {name}")
        
        # Also show nodes with children (skeleton hierarchy)
        print("\n  Nodes with children (hierarchy):")
        for i, node in enumerate(data.get('nodes', [])):
            if 'children' in node:
                name = node.get('name', f'node_{i}')
                print(f"    {name} -> {len(node['children'])} children")
        
        return len(bone_nodes) > 0


def main():
    models_dir = Path("data/avatar/models")
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            print(f"\n{'='*60}")
            print(f"Model: {model_dir.name}")
            print('='*60)
            
            gltf_files = list(model_dir.glob("*.gltf"))
            glb_files = list(model_dir.glob("*.glb"))
            
            for f in gltf_files:
                check_gltf(f)
            
            for f in glb_files:
                # For GLB, extract JSON chunk
                import struct
                with open(f, 'rb') as glb:
                    magic = glb.read(4)
                    version = struct.unpack('<I', glb.read(4))[0]
                    length = struct.unpack('<I', glb.read(4))[0]
                    chunk_length = struct.unpack('<I', glb.read(4))[0]
                    chunk_type = glb.read(4)
                    json_str = glb.read(chunk_length)
                    data = json.loads(json_str)
                
                print(f"\n=== {f.name} (GLB) ===")
                print(f"  Nodes: {len(data.get('nodes', []))}")
                print(f"  Skins: {len(data.get('skins', []))}")
                print(f"  Animations: {len(data.get('animations', []))}")
                
                if data.get('skins'):
                    for skin in data['skins']:
                        joints = skin.get('joints', [])
                        print(f"  Skeleton: {len(joints)} joints")


if __name__ == "__main__":
    main()
