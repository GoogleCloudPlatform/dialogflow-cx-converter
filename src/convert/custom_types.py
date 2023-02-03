# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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