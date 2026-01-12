"""
Avatar Tab - Main Container

This tab contains three sub-tabs:
  - Avatar: Visual representation of the AI
  - Game: Connect to games the AI can control
  - Robot: Connect to robots the AI can control

IMPORTANT: There is ONE AI being trained. That same AI controls:
  - The avatar display
  - Game connections
  - Robot connections

All control comes from the same trained model.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .avatar.avatar_display import create_avatar_subtab
from .game.game_connection import create_game_subtab
from .robot.robot_control import create_robot_subtab


def create_avatar_tab(parent):
    """
    Create the avatar tab with sub-tabs for avatar, game, and robot control.
    
    All three sub-tabs are controlled by the SAME AI that you train.
    """
    w = QWidget()
    layout = QVBoxLayout()
    
    # Sub-tabs container
    sub_tabs = QTabWidget()
    
    # Avatar sub-tab - the AI's visual representation
    avatar_widget = create_avatar_subtab(parent)
    sub_tabs.addTab(avatar_widget, "Avatar")
    
    # Game connection sub-tab - games the AI can control
    game_widget = create_game_subtab(parent)
    sub_tabs.addTab(game_widget, "Game")
    
    # Robot control sub-tab - robots the AI can control
    robot_widget = create_robot_subtab(parent)
    sub_tabs.addTab(robot_widget, "Robot")
    
    layout.addWidget(sub_tabs)
    w.setLayout(layout)
    
    return w
