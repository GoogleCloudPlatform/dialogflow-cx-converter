from typing import List, Union
from dataclasses import dataclass, field

class Flow:
    """A class to store a Watson workspace"""
    def __init__(self, input_dict: dict) -> None:
        # Check for workspace key inside dict or just use the dict
        self.workspace = input_dict.get("workspace", input_dict)

        # Check for answer mapping in the dict
        self.answers = input_dict.get("answers", [])

        # Common keys
        self.intents = self.workspace["intents"]
        self.entities = self.workspace["entities"]
        self.pages = self.workspace["dialog_nodes"]


class Agent:
    """A class to store multiple Watson workspaces"""
    def __init__(self, assuntos: List[Flow]) -> None:
        self.assuntos = assuntos

@dataclass
class Node:
    """A data class for recreating the Watson tree"""
    parent: Union["Node",None] = None
    children: List["Node"] = field(default_factory=list)
    page: dict = field(default_factory=dict)
    flow_id: str = ""
    jump_to: bool = False