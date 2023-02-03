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

import src.convert.custom_types as custom_types


def test_flow_creation_without_workspaces_field():
    input_dict = {
        "intents": [],
        "entities": [],
        "dialog_nodes": []
    }
    flow = custom_types.Flow(input_dict=input_dict)
    assert hasattr(flow, "intents")
    assert hasattr(flow, "entities")
    assert hasattr(flow, "pages")
    assert hasattr(flow, "workspace")
    assert flow.workspace == input_dict

def test_flow_creation_with_workspaces_field():
    input_dict = {
        "workspace": {
            "intents": [],
            "entities": [],
            "dialog_nodes": []
        },
        "answers": []
    }
    flow = custom_types.Flow(input_dict=input_dict)
    assert hasattr(flow, "intents")
    assert hasattr(flow, "entities")
    assert hasattr(flow, "pages")
    assert hasattr(flow, "workspace")
    assert hasattr(flow, "answers")
    assert flow.workspace == input_dict["workspace"]

def test_agent_creation():
    input_dict = {
        "intents": [],
        "entities": [],
        "dialog_nodes": []
    }
    flow = custom_types.Flow(input_dict=input_dict)
    agent = custom_types.Agent([flow])
    assert agent.assuntos[0] == flow
