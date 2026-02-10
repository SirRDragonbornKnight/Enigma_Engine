"""
Avatar Control Tool Definition

Allows AI to control avatar through bone animations as a tool call.
"""

from typing import Any

AVATAR_CONTROL_TOOL = {
    "name": "control_avatar_bones",
    "description": "Control the avatar's bones to create natural body language and gestures. Use this to make the avatar move, gesture, or express emotions physically.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action to perform",
                "enum": ["move_bone", "gesture", "reset_pose"]
            },
            "bone_name": {
                "type": "string",
                "description": "Name of the bone to move (only for move_bone action)",
                "enum": [
                    "head", "neck", "spine", "spine1", "spine2", "chest", "hips", "pelvis",
                    "left_shoulder", "left_upper_arm", "left_arm", "left_forearm", 
                    "left_hand", "left_wrist",
                    "right_shoulder", "right_upper_arm", "right_arm", "right_forearm",
                    "right_hand", "right_wrist",
                    "left_upper_leg", "left_leg", "left_lower_leg", "left_foot",
                    "right_upper_leg", "right_leg", "right_lower_leg", "right_foot"
                ]
            },
            "pitch": {
                "type": "number",
                "description": "Pitch rotation in degrees (nodding up/down). Range: -45 to 45 degrees typically"
            },
            "yaw": {
                "type": "number",
                "description": "Yaw rotation in degrees (turning left/right). Range: -80 to 80 degrees typically"
            },
            "roll": {
                "type": "number",
                "description": "Roll rotation in degrees (tilting side to side). Range: -30 to 30 degrees typically"
            },
            "gesture_name": {
                "type": "string",
                "description": "Predefined gesture name (only for gesture action)",
                "enum": ["nod", "shake", "wave", "shrug", "point", "thinking", "bow", "stretch"]
            }
        },
        "required": ["action"]
    }
}


def execute_avatar_control(action: str, bone_name: str = None, pitch: float = None,
                          yaw: float = None, roll: float = None, gesture_name: str = None) -> dict[str, Any]:
    """Execute avatar control command.
    
    Args:
        action: Action to perform (move_bone, gesture, reset_pose)
        bone_name: Name of bone to move
        pitch: Pitch rotation
        yaw: Yaw rotation  
        roll: Roll rotation
        gesture_name: Predefined gesture name
    
    Returns:
        Result dictionary with success status
    """
    try:
        from ..avatar.ai_control import get_ai_avatar_control
        
        ai_control = get_ai_avatar_control()
        
        if action == "move_bone":
            if not bone_name:
                return {"success": False, "error": "bone_name required for move_bone action"}
            
            from ..avatar.ai_control import BoneCommand
            command = BoneCommand(bone_name, pitch=pitch, yaw=yaw, roll=roll)
            ai_control.execute_commands([command])
            
            return {
                "success": True,
                "result": f"Moved {bone_name} (pitch={pitch}, yaw={yaw}, roll={roll})"
            }
        
        elif action == "gesture":
            if not gesture_name:
                return {"success": False, "error": "gesture_name required for gesture action"}
            
            success = ai_control.execute_gesture(gesture_name)
            if success:
                return {"success": True, "result": f"Executed gesture: {gesture_name}"}
            else:
                return {"success": False, "error": f"Unknown gesture: {gesture_name}"}
        
        elif action == "reset_pose":
            ai_control.reset_pose()
            return {"success": True, "result": "Avatar reset to neutral pose"}
        
        else:
            return {"success": False, "error": f"Unknown action: {action}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}
