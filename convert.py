"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
import json
import time
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field

from google.cloud.dialogflowcx_v3beta1 import types, AgentsClient, EntityTypesClient, FlowsClient, IntentsClient, PagesClient
from google.api_core.exceptions import InvalidArgument, AlreadyExists, FailedPrecondition


#========= Dialogflow Constants ===============================================
QUOTA_REQUESTS = 60
QUOTA_REQUESTS_WINDOW_IN_SECONDS = 60


#========= Help Constants =====================================================
REQUESTS_LIMIT = QUOTA_REQUESTS // 2
REQUESTS_REFRESH = QUOTA_REQUESTS_WINDOW_IN_SECONDS * 2


#========= Classes ============================================================
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


@dataclass
class Expression:
    """A data class for recreating the Watson expression"""
    operations: Dict[str, str] = field(default_factory=dict)
    intents: List[str] = field(default_factory=list)
    condition: str = ""
    events: List[str] = field(default_factory=list)


#========= Help Function ======================================================
def is_number(s: str) -> bool:
    """A help function to check if the string is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def wait_for_time(requests_per_minute: dict):
    """A help function to wait for quota"""
    time.sleep(60 // REQUESTS_LIMIT)
    now = time.time()
    if now - QUOTA_REQUESTS_WINDOW_IN_SECONDS > requests_per_minute["time"]:
        requests_per_minute["time"] = now
        requests_per_minute["request_count"] = 0
    
    if requests_per_minute["request_count"] + 1 > REQUESTS_LIMIT:
        time.sleep(REQUESTS_REFRESH-(now-requests_per_minute["time"]))
        requests_per_minute["time"] = time.time()
        requests_per_minute["request_count"] = 0
    else:
        requests_per_minute["request_count"] += 1


def parse_condition_tokens(token: str):
    """
    A help function to parse condition tokens
    # - intents
    @ - entities
    @{entity-name}:{value} - Value of entity
    $ - context variable
    ${context}:{value} - Value of contex variable
    anything_else: No match
    conversation_start: First Node
    welcome: First Node with user input, greet the user
    false: Always false
    true: Always true
    irrelevant: True if user input is irrelevant (No match/No input)
    now(): function of time
    str.contains(str):
    size(): check size
    intents[0].confidence: Change to true
    entities.size(): Change to true or page form completed
    time.plusHours/plusMinutes(): Not implemented
    """
    # [TODO] plus minutes, plus hours, webhook needed
    token_type = "unknown"
    converted = token

    if len(token) > 0:

        if ".contains(" in token:
            if token[0] == "[":
                tokens = token[:-1].split(".contains(")
                print(tokens)
                item_list = tokens[0].replace("[", "").replace("]", "").split(",")
                print(item_list)
                converted_list = [parse_condition_tokens(item)[1] for item in item_list]
                _, converted_contain = parse_condition_tokens(tokens[1])
                converted = f"$sys.func.CONTAIN([{','.join(converted_list)}], {converted_contain})"
            else:
                tokens: List[str] = token[:-1].split(".contains(")
                if tokens != None and len(tokens) == 2:
                    _, token_string = parse_condition_tokens(tokens[0]) # type: ignore 
                    _, token_substring = parse_condition_tokens(tokens[1]) # type: ignore 
                    token_type = "contains"
                    converted = f"{token_string} : {token_substring}"
        
        # This is not posible in Dialogflow, check for entities
        elif "entities.size()" in token:
            token_type = "params_final"
            converted = '($page.params.status = "FINAL" || true)'
        
        # This is not posible in Dialogflow
        elif token.startswith("intents["):
            token_type = "intents"
            converted = 'true'
        
        elif token[0] == "!":
            token_type = "negation"
            token_type, converted = parse_condition_tokens(token[1:])
            converted = f"NOT {converted}"

        elif token[0] == "[":
            token_type = "string"
            converted = token[1:-1]

        elif token[0] == '#':
            token_type = "intent"
            converted = token[1:]
        
        elif token[0] == '@' or token[0] == '$':
            if "." in token and "(" in token:
                token_type = "function"
                converted = "0"

            elif ":" in token:
                token_type = "entity_value"
                tokens = token.split(":")
                tokens[1] = tokens[1].replace("(", "").replace(")", "")
                converted = f"$session.params.{tokens[0][1:]} = '{tokens[1]}'"
            else:
                token_type = "entity"
                converted = f"$session.params.{token[1:]}"
        
        elif token.startswith("now()"):
            token_type = "checkTime"
            operation = ""
            time = token.split("(")[-1][:-1]
            if "sameOrAfter" in token:
                operation = ">="
            elif "after" in token:
                operation = ">"
            elif "sameOrBefore" in token:
                operation = "<="
            elif "before" in token:
                operation = "<"
            # converted = f"$sys.func.NOW() {operation} {time}"
            converted = "0"
        
        elif token.lower() == "true":
            token_type = "true"
            converted = "true"
        
        elif token.lower() == "false":
            token_type = "false"
            converted = "false"
        
        elif token.lower() == "anything_else" or token.lower() == "irrelevant":
            token_type = "no_match"
        
        elif token.lower() == "conversation_start":
            token_type = "start"
        
        elif token.lower() == "welcome":
            token_type = "intent"
            converted = "Default Welcome Intent"
    
    return token_type, converted


def tokenize_literals(conditions: str) -> Tuple[str, dict, int]:
    """A function to tokenize watson literals with the format: &|number|"""
    token_count = 0
    token_map = {}
    while '"' in conditions:
        first = conditions.find('"')
        second = conditions[first + 1:].find('"') + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = conditions[first:second + 1].replace('"', "'")
        conditions = conditions[:first] + token_key + conditions[second+1:]
    
    while "'" in conditions:
        first = conditions.find("'")
        second = conditions[first + 1:].find("'") + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = conditions[first:second + 1]
        conditions = conditions[:first] + token_key + conditions[second+1:]
    
    while ':(' in conditions:
        first = conditions.find(':(')
        second = conditions[first + 1:].find(')') + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = conditions[first+2:second]
        conditions = conditions[:first+1] + token_key + conditions[second+1:]

    return conditions, token_map, token_count


def parse_expression(
    expression: str,
    subexpressions: Dict[str, Expression] ={},
    subexpression_count: int = 0) -> Tuple[Expression, dict, int]:
    """A function to parse expressions from Watson to Dialogflow format"""
    
    intents = []
    events = []

    if len(expression) > 1:

        # Add a space to help with the check
        if expression[-1] == ")":
            expression += " "
        
        # Count the number of close parenthesis to avoid infinite loop
        parenthesis_count = expression.count(") ")
        
        start = 0
        end = len(expression)

        while parenthesis_count>=0:

            # Check for a close parenthesis
            first_close_parenthesis = expression[start:].find(") ") + start

            if first_close_parenthesis == -1:
                break
            end = min(end, first_close_parenthesis)

            # Check for an open parenthesis
            closest_open_parenthesis = expression[:end].rfind("(")
            if closest_open_parenthesis == -1:
                start = first_close_parenthesis + 1
                parenthesis_count -= 1
                continue

            if closest_open_parenthesis > 0:
                if expression[closest_open_parenthesis-1] != " ":
                    end = closest_open_parenthesis
                    start = first_close_parenthesis + 1
                    parenthesis_count -= 1
                    continue
            
            # If we find parenthesis, then it is a subexpression, recursive call
            subexpression, subexpressions, subexpression_count = parse_expression(
                expression[closest_open_parenthesis+1:first_close_parenthesis],
                subexpressions=subexpressions,
                subexpression_count=subexpression_count)
            

            subexpression_str = ""

            # Tokenize subexpression in the format: &|(number)|&
            if subexpression.condition:
                subexpression_key = f"&|({subexpression_count})|&"
                subexpression_str = f" {subexpression_key} " if subexpression.condition else ""
                subexpression_count += 1
                subexpressions[subexpression_key] = subexpression

            intents += subexpression.intents

            expression = expression[:closest_open_parenthesis] + subexpression_str + expression[first_close_parenthesis+1:]
            expression = expression.replace("    ", " ")
            
            # Reset loop
            start = 0
            end = len(expression)
            parenthesis_count -= 1
        
        expression = expression.replace(") )", ")")
        
        
        # Convert tokens into Dialogflow expression
        tokens = expression.split()
        converted_tokens = []
        empty = True
        comparison_flag = False
        intent_or_event_flag = False
        entity_flag = False
        or_and_flag = False
        remove_next = False
        operation = ""

        for token in tokens:
            is_last_comparison = comparison_flag
            is_last_intent_or_event = intent_or_event_flag
            is_last_entity = entity_flag
            is_last_or_and = or_and_flag
            comparison_flag = False
            intent_or_event_flag = False
            entity_flag = False
            or_and_flag = False


            if remove_next:
                remove_next = False
                continue

            #[TODO] + - 
            if token in ["+", "-"]:
                if empty or operation:
                    remove_next = True
                    operation = ""
                    continue

                if token == "+":
                    operation = f"$sys.func.ADD({converted_tokens.pop()},"
                else:
                    operation = f"$sys.func.MINUS({converted_tokens.pop()},"

                

            elif token in ["||", "&&", ">", ">=", "<", "<=", "==", "!=", "+", "-"]:

                if operation:
                    operation = ""
                    continue

                if token in ["||", "&&"]:
                    if empty or is_last_intent_or_event or is_last_or_and:
                        continue
                    else:
                        or_and_flag = True
                    token = "OR" if token == "||" else "AND"
                else:
                    if empty:
                        remove_next = True
                        continue
                    if token == "==":
                        token = "="
                    if is_last_intent_or_event:
                        remove_next = True
                        continue
                    
                    else:
                        comparison_flag = True
                        is_last_entity = False
                
                if is_last_entity:
                    converted_tokens.append("!=")
                    converted_tokens.append("null")
                
                

                converted_tokens.append(token)
                
            elif (token.startswith("&|(")) or token.startswith("&|"):
                if operation:
                    operation = ""
                    continue

                if empty:
                    if token.startswith("&|("):
                        if subexpressions.get(
                            token, Expression()).condition.strip():
                            empty = False
                    else:
                        empty = False
                converted_tokens.append(token)
            elif is_number(token):
                if operation:
                    token = operation + token + ")"
                    operation = ""
                converted_tokens.append(token)
                empty = False
            else:
                token_type, converted = parse_condition_tokens(token)
                if token_type in ["no_match", "start"]:
                    if operation:
                        operation = ""
                    events.append(token_type)
                    intent_or_event_flag = True
                    if is_last_comparison or is_last_or_and:
                        converted_tokens.pop()
                elif token_type in ["true", "false"]:
                    converted_tokens.append(token)
                elif token_type == "intent":
                    if operation:
                        operation = ""
                    intents.append(converted)
                    intent_or_event_flag = True
                    if is_last_comparison or is_last_or_and:
                        converted_tokens.pop()
                elif token_type in ["intents", "params_final", "checkTime", "function"]:
                    if operation:
                        operation = ""
                    intent_or_event_flag = True
                    if is_last_comparison or is_last_or_and:
                        converted_tokens.pop()
                elif token_type == "entity":
                    if not is_last_comparison:
                        entity_flag = True
                    if operation:
                        token = operation + token + ")"
                        operation = ""
                    converted_tokens.append(converted)
                    empty = False
                elif token_type == "entity_value":
                    if operation:
                        operation = ""
                    converted_tokens.append(converted)
                    empty = False
                else:
                    if operation:
                        operation = ""
                    converted_tokens.append(converted)
        
        if entity_flag:
            converted_tokens.append("!=")
            converted_tokens.append("null")

        expression = " ".join(converted_tokens)
    
    expression_obj = Expression(
        operations={},
        intents=intents,
        events=events,
        condition=expression)

    return expression_obj, subexpressions, subexpression_count


def parse_conditions(conditions: str) -> Expression:
    """Function to parse Watson condition into a Dialogflow condition"""
    
    conditions = conditions.replace("((", "( (").replace("))", ") )")
    
    # Tokenize Literals
    conditions, token_map, token_count = tokenize_literals(conditions)

    expression, subexpressions, _ = parse_expression(conditions, {}, 0)

    condition = expression.condition
    
    if condition:
        while subexpressions:
            converted_tokens = []
            tokens = condition.split()
            for token in tokens:
                if (token.startswith("&|(")):
                    if subexpressions[token].condition.strip():
                        converted_tokens.append("(")
                        converted_tokens.append(subexpressions[token].condition)
                        converted_tokens.append(")")
                    del subexpressions[token]
                else:
                    converted_tokens.append(token)
            
            condition = " ".join(converted_tokens)

        if token_count:
            for k, v in token_map.items():
                condition = condition.replace(k,v)
    
    expression.condition = condition

    return expression


def create_node_tree(assunto: Flow) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Function to create a node tree from Watson Dialog Nodes"""
    start_node = Node()

    # All pages
    pages: Dict[str, Node] = {"start": start_node}

    # Only roots
    roots: Dict[str, Node] = {"start": start_node}


    for page in assunto.pages:
        # Nodes for pages without a parent
        if not page.get("parent", ""):
            if not pages.get(page["dialog_node"], ""):
                parent = None 
                pages[page["dialog_node"]] = Node(
                    page=page,
                    flow_id=page["dialog_node"])
            # If the node already exists, just update the dict
            else:
                pages[page["dialog_node"]].page = page
                pages[page["dialog_node"]].flow_id = page["dialog_node"]
            
            # conversation_start indicates first node
            if "conversation_start" in page.get("conditions", ""):
                pages[page["dialog_node"]].parent = roots["start"]
                roots["start"].children.append(pages[page["dialog_node"]])
            else:
                # Add as root
                roots[page["dialog_node"]] = pages[page["dialog_node"]]
        # Nodes for pages with a parent
        else:
            # If parent node does not exist, creates a placeholder
            if not pages.get(page["parent"], ""):
                pages[page["parent"]] = Node()
            
            parent = pages[page["parent"]]
            flow_id = ""
            parents = []
            
            # If the flow id is defined, updates for all nodes in the tree
            while parent:
                if parent.flow_id:
                    flow_id = parent.flow_id
                    for parent_item in parents:
                        parent_item.flow_id = flow_id
                    break
                parents.append(parent)
                parent = parent.parent

            # Creates a page node or just update it
            if not pages.get(page["dialog_node"], ""):
                pages[page["dialog_node"]] = Node(
                    page=page,
                    flow_id=flow_id)
            else:
                if flow_id:
                    for child in pages[page["dialog_node"]].children:
                        child.flow_id = flow_id
                pages[page["dialog_node"]].page = page
                pages[page["dialog_node"]].flow_id = flow_id

            pages[page["parent"]].children.append(pages[page["dialog_node"]])
        
        if "next_step" in page:
            page_next_step = page["next_step"]
            if page_next_step.get("behavior","") == "jump_to":
                dialog_node_id = page_next_step.get("dialog_node", "")
                if dialog_node_id:
                    if not pages.get(dialog_node_id, ""):
                        pages[dialog_node_id] = Node(
                            jump_to=True)
                    else:
                        pages[page["dialog_node"]].jump_to = True
    
    return pages, roots
        

#========= Creation Functions =================================================
def create_or_get_agent(
    create_agent: bool,
    requests_per_minute: dict,
    display_name: str,
    language_code: str, 
    time_zone: str, 
    project_id: str,
    location: str
    ) -> types.Agent:
    agents_functions = Agents(creds_path=creds_path) # type: ignore
    agents_client = AgentsClient(credentials=agents_functions.creds)
    if create_agent:
        wait_for_time(requests_per_minute)
        agent = agents_client.create_agent(
            agent=types.Agent(
                display_name=display_name,
                default_language_code=language_code,
                time_zone=time_zone
            ),
            parent=f"projects/{project_id}/locations/{location}"
        )
        print(agent)
        return agent
    else:
        wait_for_time(requests_per_minute)
        agents = list(agents_client.list_agents(parent=f"projects/{project_id}/locations/{location}"))
        for agent in agents:
            if agent.display_name == display_name:
                return agent
    
    raise ValueError(f"Agent {display_name} not found")
    


def create_or_get_entity_types(
    create_entities:bool,
    requests_per_minute: dict,
    bot:Agent, 
    agent: types.Agent,
    project_id: str,
    location: str
    ) -> List[types.EntityType]:
    
    entity_types = []
    entity_types_names = []
    
    # entity_types_functions = EntityTypes(creds_path=creds_path)
    entity_type_client = EntityTypesClient()
    if create_entities:
        for assunto in bot.assuntos:
            for entity in assunto.entities:
                if entity["entity"] in entity_types_names:
                    continue
                entities = []
                for value in entity["values"]:
                    if value["synonyms"]:
                        entities.append(
                            types.EntityType.Entity(
                                value=value["value"],
                                synonyms=value["synonyms"]
                            )
                        )
                if not entities:
                    continue
                entity_type = types.EntityType(
                    display_name=entity["entity"],
                    kind=types.EntityType.Kind.KIND_MAP,
                    entities=entities,
                    excluded_phrases=[],
                    enable_fuzzy_extraction=False,
                    redact=False
                )
                wait_for_time(requests_per_minute)
                entity_types.append(
                    entity_type_client.create_entity_type(
                        parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}",
                        entity_type=entity_type)
                )
                entity_types_names.append(entity["entity"])
        print("Finished Entities")
    else:
        wait_for_time(requests_per_minute)
        entity_types = list(entity_type_client.list_entity_types(
            parent=f"projects/{project_id}/locations/global/agents/{str(agent.name)}"))
    
    return entity_types


def create_or_get_intents(
    create_intents: bool,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    entity_types: List[types.EntityType],
    add_description_as_training_phrase: bool, 
    project_id: str,
    location: str
    ):
    intents = []
    intents_client = IntentsClient()
    if create_intents:
        for assunto in bot.assuntos:
            for intent in assunto.intents:
                training_phrases = []
                entity_ids=[]
                parameters=[]
                for example in intent["examples"]:
                    parts=[]
                    if "mentions" in example:
                        # Untested
                        text = example["text"]
                        last_idx = 0
                        for mention in example["mention"]:
                            loc = mention["location"]
                            entity = mention["entity"]
                            entity_id = ""
                            for entity_type in entity_types:
                                if entity_type.display_name == entity:
                                    entity_id = entity_type.name
                            if not entity_id:
                                parts.append(
                                    types.Intent.TrainingPhrase.Part(
                                        text=text[last_idx:loc[1]]
                                    )
                                )
                                last_idx=loc[1]
                                continue
                            parameter = types.Intent.Parameter(
                                entity_type=entity_id,
                                id=entity_id
                            )
                            if entity_id not in entity_ids:
                                entity_ids.append(entity_id)
                                parameters.append(parameter)
                            if last_idx < loc[0]:
                                # Only text part with no parameter annotations
                                parts.append(
                                    types.Intent.TrainingPhrase.Part(
                                        text=text[last_idx:loc[0]]
                                    )
                                )
                            # Part with parameter annotation
                            parts.append(
                                types.Intent.TrainingPhrase.Part(
                                    text=text[loc[0]:loc[1]],
                                    parameter=parameter
                                )
                            )
                            last_idx = loc[1]
                    else:
                        parts.append(
                            types.Intent.TrainingPhrase.Part(
                                text=example["text"]
                            )
                        )
                    training_phrases.append(
                        types.Intent.TrainingPhrase(
                            parts=parts,
                            repeat_count=1
                        )
                    )
                if add_description_as_training_phrase:
                    if "description" in intent:
                        training_phrases.append(
                            types.Intent.TrainingPhrase(
                                parts=[
                                    types.Intent.TrainingPhrase.Part(
                                        text= intent.get("description","")
                                        )],
                                repeat_count=1
                            )
                        )
                intent_obj = types.Intent(
                    display_name=intent["intent"],
                    training_phrases=training_phrases,
                    parameters=parameters,
                    priority=500000, # Normal is 500000
                    is_fallback=False,
                    labels=[],
                    description= intent.get("description","")
                )
                wait_for_time(requests_per_minute)
                intents.append(
                    intents_client.create_intent(
                        parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}",
                        intent=intent_obj)
                )

        print("Finished Intents")
    else:
        intents = list(intents_client.list_intents(
            parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}"))
    
    return intents


def create_or_get_flows(
    create_flows:bool,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    parentless_folders_as_flows: bool,
    project_id: str,
    location: str
    ) -> Dict[str, types.Flow]:
    flows = {}
    flows_client = FlowsClient()
    if create_flows:
        if not parentless_folders_as_flows:
            for assunto in bot.assuntos:
                flow_obj = types.Flow(
                    display_name=assunto.workspace["name"],
                    description=assunto.workspace.get("description",""),
                    transition_routes=[],
                    event_handlers=[],
                    transition_route_groups=[]
                )
                wait_for_time(requests_per_minute)
                flows[assunto.workspace["name"]] = flows_client.create_flow(
                    parent=str(agent.name),flow=flow_obj)
                print("Finished Flows")
        else:
            assunto = bot.assuntos[0]
            for page in assunto.pages:
                if page.get("type", "") == "folder" and not page.get("parent", ""):
                    display_name = page.get("title", page["dialog_node"])
                    flow_obj = types.Flow(
                        display_name=display_name,
                        description=None,
                        transition_routes=[],
                        event_handlers=[],
                        transition_route_groups=[]
                    )
                    flows[str(page["dialog_node"])] = flows_client.create_flow(
                    parent=str(agent.name),flow=flow_obj)
    else:
        wait_for_time(requests_per_minute)
        flows_list = list(flows_client.list_flows(parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}"))
        if parentless_folders_as_flows:
            assunto = bot.assuntos[0]
            for page in assunto.pages:
                if page.get("type", "") == "folder" and not page.get("parent", ""):
                    flow = [flow for flow in flows_list if flow.display_name in [
                        page.get("title", ""),
                        page["dialog_node"]]]
                    if flow:
                        flows[str(page["dialog_node"])] = flow[0]
        else:
            flows = {str(flow.display_name): flow for flow in flows_list}

    return flows


def create_child_page(
    page: Node,
    pages: Dict[str, dict],
    intents,
    flow_name: str,
    flow_id: str,
    flows_client: FlowsClient,
    pages_client: PagesClient,
    requests_per_minute: dict,
    parent: Union[types.Flow, types.Page], 
    agent: types.Agent,
    project_id: str, 
    location: str):
    
    page_dict = page.page
    if page_dict.get("type", "standard") not in [
        "standard", "folder", "response_condition"]:
        print("not the right type")
        return

    if not page_dict.get("conditions", ""):
        print("no condition")
        return

    expression = parse_conditions(page_dict["conditions"])
    
    messages = [
        types.ResponseMessage(
            text=types.ResponseMessage.Text(text=[value])
            ) for value in page_dict.get(
                "output",{"text":{"values":[]}})
                .get("text", {"values":[]})
                .get("values",[])]
    
    parameter_actions = []

    for k, v in page_dict.get("context", {}).items():
        parameter_actions = [
            types.Fulfillment.SetParameterAction(
                parameter=k,
                value=v)]

    fullfillment = types.Fulfillment(
        messages=messages,
        set_parameter_actions=parameter_actions)
    
    dialog_node_id = str(page.page["dialog_node"])

    page_target = False

    if page.children or page.jump_to:
        if page_dict.get("type", "standard") != "response_condition":
            child_page = types.Page(
                display_name = page.page.get(
                    "title",
                    dialog_node_id)
            )
        else:
            child_page = types.Page(
                display_name = dialog_node_id
            )
        
        if not flow_name in pages:
            pages[flow_name] = {
                "pages": {}
            }

        wait_for_time(requests_per_minute)
        try:
            pages[flow_name][
                "pages"][dialog_node_id] = pages_client.create_page(
                parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}/flows/{flow_id}",
                page=child_page)
            page_target = True
        except FailedPrecondition as e:
            print(e)
            return
        except AlreadyExists as e:
            print(e)
            try:
                child_page = types.Page(
                    display_name = dialog_node_id
                )
                pages[flow_name][
                    "pages"][dialog_node_id] = pages_client.create_page(
                parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}/flows/{flow_id}",
                page=child_page)
                page_target = True
            except AlreadyExists as e:
                print(e)
                wait_for_time(requests_per_minute)
                pages_map = {
                    page.display_name: page
                    for page in pages_client.list_pages(
                        parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}/flows/{flow_id}")
                }

                page_id = str(pages_map.get(dialog_node_id, pages_map.get(page.page.get("title", ""))))
                wait_for_time(requests_per_minute)
                pages[flow_name][
                    "pages"][dialog_node_id] = pages_map.get(page_id, None)
    
    if expression.condition or expression.intents or expression.events:
        intent_ids = []
        update_parent = False

        if len(expression.intents) > 0:
            for expression_intent in expression.intents:
                for intent in intents:
                    if intent.display_name == expression_intent:
                        intent_ids.append(intent.name)
        
        if len(expression.events) > 0:
            for expression_event in expression.events:
                if expression_event == "start":
                    if not "Default Welcome Intent" in expression.intents:
                        for intent in intents:
                            if intent.display_name == "Default Welcome Intent":
                                intent_ids.append(intent.name)
        
        if intent_ids:
            for intent_id in intent_ids:
                transition_route = types.TransitionRoute(
                    intent=intent_id,
                    target_page=pages[flow_name]["pages"][dialog_node_id].name if page_target else None,
                    trigger_fulfillment=fullfillment,
                    condition=expression.condition if expression.condition else None
                )
                parent.transition_routes.append(transition_route) # type: ignore
                update_parent = True
        elif expression.condition:
            transition_route = types.TransitionRoute(
                target_page=pages[flow_name]["pages"][dialog_node_id].name if page_target else None,
                trigger_fulfillment=fullfillment,
                condition=expression.condition
            )
            parent.transition_routes.append(transition_route) # type: ignore
            update_parent = True

        if update_parent:
            wait_for_time(requests_per_minute)
            try:
                if type(parent) == types.Flow:
                    parent = flows_functions.update_flow(str(parent.name), parent) # type: ignore
                else:
                    parent = pages_functions.update_page(str(parent.name), parent) # type: ignore
            except InvalidArgument as e:
                print(e)
                if intent_ids:
                    for intent_id in intent_ids:
                        parent.transition_routes.pop() # type: ignore
                else:
                    parent.transition_routes.pop() # type: ignore
                print(f"Original condition: {page_dict['conditions']}")

    for child in page.children:
        create_child_page(
            page=child,
            pages=pages,
            intents=intents,
            flow_name=flow_name,
            flow_id=flow_id,
            flows_client=flows_client,
            pages_client=pages_client,
            requests_per_minute=requests_per_minute,
            parent=pages[flow_name]["pages"][dialog_node_id],
            agent=agent,
            project_id=project_id,
            location=location)


def create_jump_to_routes(
    name: str,
    page: Node,
    pages: Dict[str, dict],
    flows: Dict[str, types.Flow],
    requests_per_minute: dict,
    pages_client: PagesClient):

    if not page.flow_id:
        print("no flow id")
        return
    
    if not pages.get(page.flow_id, ""):
        print("flow not found in pages")
        return

    if not pages[page.flow_id]["pages"].get(name, None):
        print("page not found in pages")
        return
    
    page_dict = page.page
    if "next_step" not in page_dict:
        print("next step not found")
        return
    page_next_step = page_dict["next_step"]
    if page_next_step.get("behavior","") != "jump_to":
        print("jump to not found")
        return
    

    dialog_node_id = page_next_step.get("dialog_node", "")
    if not dialog_node_id:
        print("next page not found")
        return
    
    page_id = ""
    flow_id = ""

    if dialog_node_id == page.flow_id:
        flow_id = str(flows[dialog_node_id].name)
    else:
        page_id = str(pages[page.flow_id]["pages"].get(dialog_node_id, ""))
        if not page_id:
            print("target page id not found")
            return
        


    messages = [
        types.ResponseMessage(
            text=types.ResponseMessage.Text(text=[value])
            ) for value in page_dict.get(
                "output",{"text":{"values":[]}})
                .get("text", {"values":[]})
                .get("values",[])]
    
    parameter_actions = []

    for k, v in page_dict.get("context", {}).items():
        parameter_actions = [
            types.Fulfillment.SetParameterAction(
                parameter=k,
                value=v)]

    fullfillment = types.Fulfillment(
        messages=messages,
        set_parameter_actions=parameter_actions)

    transition_route = types.TransitionRoute(
        target_page=page_id if page_id else None,
        target_flow=flow_id if flow_id else None,
        trigger_fulfillment=fullfillment,
        condition="true"
    )
    pages[page.flow_id]["pages"][name].transition_routes.append(transition_route) # type: ignore
    wait_for_time(requests_per_minute)
    pages[page.flow_id]["pages"][name] = pages_client.update_page(
        pages[page.flow_id]["pages"][name])


def create_pages_folders_as_flows(
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    pages: Dict[str, dict],
    intents: List[types.Intent],
    flows_client: FlowsClient,
    pages_client: PagesClient,
    requests_per_minute: dict, 
    project_id: str, 
    location: str
    ) -> Dict[str, dict]:
    
    page_nodes, trees = create_node_tree(bot.assuntos[0])
    # Create intent and condition routes
    for name, tree in trees.items():
        flow = ""
        # Start flow
        if name == "start":
            print("start")
            wait_for_time(requests_per_minute)
            flow = flows_client.get_flow(name=str(agent.start_flow))
        elif name in flows:
            flow = flows[name]
        
        if flow:
            for child in tree.children:
                print("start_child")
                create_child_page(
                    page=child,
                    pages=pages,
                    intents=intents,
                    flow_name=name,
                    flow_id=str(flow.name),
                    flows_client=flows_client,
                    pages_client=pages_client,
                    requests_per_minute=requests_per_minute,
                    parent=flow,
                    agent=agent,
                    project_id=project_id,
                    location=location)
    
    # Create jump_to routes:
    for name, page in page_nodes.items():
        create_jump_to_routes(name=name, page=page, pages=pages, flows=flows, requests_per_minute=requests_per_minute, pages_client=pages_client)
    
    return pages


def create_pages_multiflow(
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    pages: Dict[str, dict],
    intents: List[types.Intent],
    flows_client: FlowsClient,
    pages_client: PagesClient,
    requests_per_minute: dict, 
    project_id: str, 
    location: str
    ) -> Dict[str, dict]:
    for assunto in bot.assuntos:
        is_parent_pages = {page["dialog_node"]: False for page in assunto.pages}
        for page in assunto.pages:
            parent = page.get("parent", "")
            if parent:
                is_parent_pages[parent] = True
        
        
        flow_id = ""
        flow_obj = types.Flow()
        flow_display_name = assunto.workspace["name"]

    
        for k, flow in flows.items():
            if assunto.workspace["name"]==flow.display_name:
                flow_id = str(flow.name)
                flow_obj = flow
                    
        
        pages[flow_display_name] = {
            "flow_id": flow_id,
            "pages": []
        }

        for page in assunto.pages:
            if len(pages[flow_display_name]["pages"]) > 200:
                continue
            elif page.get("type","") not in ["standard", "folder", "response_condition"]:
                continue

            # Prepare fullfillment
            messages = [
                types.ResponseMessage(
                    text=types.ResponseMessage.Text(
                        text=[answer["answer"]]
                    )
                ) for answer in assunto.answers if (
                    answer["dialog_node"]==page["dialog_node"])] + [
                types.ResponseMessage(
                    text=types.ResponseMessage.Text(text=[value])
                    ) for value in page.get("output",{"text":{"values":[]}}).get("text", {"values":[]}).get("values",[])]
            
            parameter_actions = []

            for k, v in page.get("context", {}).items():
                parameter_actions = [types.Fulfillment.SetParameterAction(parameter=k, value=v)]

            fullfillment = types.Fulfillment(messages=messages, set_parameter_actions=parameter_actions)

            #[TODO] Manage next_steps like jump_to and skip

            # Validate conditions
            conditions = page.get("conditions","")
            if conditions == "":
                continue
            expression = parse_conditions(conditions)
            if not (expression.condition or expression.intents):
                continue
            
            intent_ids = []

            if len(expression.intents) > 0:
                for expression_intent in expression.intents:
                    for intent in intents:
                        if intent.display_name == expression_intent:
                            intent_ids.append(intent.name)

            routes = []
            target_page = ""
            if is_parent_pages.get(page["dialog_node"], False):
                page_obj=types.Page(
                    display_name=page["dialog_node"],
                    entry_fulfillment=None,
                    form=None,
                    transition_route_groups=[],
                    transition_routes=[],
                    event_handlers=[]
                )
                wait_for_time(requests_per_minute)
                page_response = pages_client.create_page(
                        parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}/flows/{flow_id}",
                        page=page_obj
                    )
                target_page = page_response.name
                pages[flow_display_name]["pages"].append(page_response)

            if intent_ids:
                for intent_id in intent_ids:
                    transition_route = types.TransitionRoute(
                        intent=intent_id,
                        target_page=target_page if target_page else None,
                        trigger_fulfillment=fullfillment,
                        condition=expression.condition if expression.condition else None
                    )
                    routes.append(transition_route)
            else:
                transition_route = types.TransitionRoute(
                    target_page=target_page if target_page else None,
                    trigger_fulfillment=fullfillment,
                    condition=expression.condition if expression.condition else None
                )
                routes.append(transition_route)
            

            if page.get("parent", ""):
                parent_page = [p_page for p_page in pages[flow_display_name]["pages"] if p_page.display_name == page["parent"]]
                if parent_page:
                    p_page_obj = parent_page[0]
                    p_page_obj.transition_routes.extend(routes)
                    wait_for_time(requests_per_minute)
                    try:
                        pages_client.update_page(page=p_page_obj)
                    except InvalidArgument as e:
                        print(e)
                else:
                    wait_for_time(requests_per_minute)
                    backup_routes = list(flow_obj.transition_routes).copy() # type: ignore
                    flow_obj.transition_routes.extend(routes) # type: ignore
                    
                    try:
                        flows_client.update_flow(flow=flow_obj)
                    except InvalidArgument as e:
                        print(e)
                        flow_obj.transition_routes = backup_routes # type: ignore
        
        print("Finished flow pages")
    
    return pages


def create_or_get_pages(
    create_pages:bool,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    intents: List[types.Intent],
    parentless_folders_as_flows: bool, 
    project_id: str, 
    location: str
    ) -> Dict[str, dict]:
    """
    dialog_node - supported, used as display name
    conditions - partially supported, it does not work with functions
    parent - supported as route using conditions
    previous_sibling - not supported
    output - supported for text values
    context - partially supported, it does not work with functions
    metadata - maybe
    next_step - maybe
    title - maybe
    type - Possible values: 
        [standard, - supported
        event_handler, - not supported
        frame, - not supported
        slot, - not supported
        response_condition, - supported
        folder] - supported
    event_name - not supported
    actions - not supported
    digress_in - not supported
    digress_out - not supported
    digress_out_slots - not supported
    user_label - not supported
    disambiguation_opt_out - not supported
    disbaled - not supported
    created - not supported
    updated - not supported
    """
    
    pages: Dict[str, dict] = {}
    pages_client = PagesClient()
    flows_client = FlowsClient()

    if create_pages:
        if parentless_folders_as_flows:
            return create_pages_folders_as_flows(
                bot=bot,
                agent=agent,
                flows=flows,
                pages=pages, 
                intents=intents,
                flows_client=flows_client,
                pages_client=pages_client,
                requests_per_minute=requests_per_minute, 
                project_id=project_id,
                location=location)
        else:
            return create_pages_multiflow(
                bot=bot,
                agent=agent,
                flows=flows,
                pages=pages, 
                intents=intents,
                flows_client=flows_client,
                pages_client=pages_client,
                requests_per_minute=requests_per_minute, 
                project_id=project_id,
                location=location)

    else:
        for flow in flows:
            wait_for_time(requests_per_minute)
            pages[str(flows[flow].name)]["pages"] = pages_client.list_pages(
                parent=f"projects/{project_id}/locations/{location}/agents/{str(agent.name)}/flows/{str(flows[flow].name)}")

    
    return pages


# ========= Main ==============================================================

def main(
    input_paths: str,
    display_name: str,
    language_code: str,
    time_zone: str,
    location:str,
    project_id: str,
    create_agent: bool = True,
    create_entities:  bool = True,
    create_intents: bool = True,
    add_description_as_training_phrase: bool = True,
    create_flows: bool = True,
    create_pages: bool = True,
    parentless_folders_as_flows = False
    ) -> None:
    """Main function"""
    
    # Prepare counter for rate limiting
    requests_per_minute={
        "request_count": 0,
        "time": time.time()
    }

    # Import Watson jsons
    assuntos: List[Flow] = []
    for input_path in input_paths:
        with open(input_path, "r") as f:
            assuntos.append(Flow(json.loads(f.read())))
    
    bot = Agent(assuntos)
    # Create Agent
    agent = create_or_get_agent(
        create_agent=create_agent,
        requests_per_minute=requests_per_minute,
        display_name=display_name,
        language_code=language_code,
        time_zone=time_zone,
        project_id=project_id,
        location=location
        )
    
    # Create Entity Types
    entity_types = create_or_get_entity_types(
        create_entities=create_entities,
        requests_per_minute=requests_per_minute,
        bot=bot, 
        agent=agent,
        project_id=project_id,
        location=location
        )
    
    # Create Intents
    intents = create_or_get_intents(
        create_intents=create_intents,
        requests_per_minute=requests_per_minute,
        bot=bot,
        agent=agent,
        entity_types=entity_types,
        add_description_as_training_phrase=add_description_as_training_phrase, 
        project_id=project_id,
        location=location
        )

    # Create Flows
    flows = create_or_get_flows(
        create_flows=create_flows,
        requests_per_minute=requests_per_minute,
        bot=bot,
        agent=agent,
        parentless_folders_as_flows=parentless_folders_as_flows, 
        project_id=project_id,
        location=location
        )
    
    # Create Pages
    create_or_get_pages(
        create_pages=create_pages,
        requests_per_minute=requests_per_minute,
        bot=bot,
        agent=agent,
        flows=flows,
        intents=intents,
        parentless_folders_as_flows=parentless_folders_as_flows, 
        project_id=project_id,
        location=location
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        dest="input",
        default="watson.json",
        help="Input json to process.",
    )

    parser.add_argument(
        "--bot_display_name",
        dest="bot_name",
        default="Converted_Bot",
        help="Display name of the bot.",
    )

    parser.add_argument(
        "--bot_language_code",
        dest="bot_language_code",
        default="pt-br",
        help="Language of the bot.",
    )

    parser.add_argument(
        "--bot_time_zone",
        dest="bot_time_zone",
        default="America/Buenos_Aires",
        help="Time zone of the bot.",
    )

    parser.add_argument(
        "--bot_location",
        dest="bot_location",
        default="global",
        help="Location of the bot.",
    )

    parser.add_argument(
        "--project_id",
        dest="project_id",
        default="project_id",
        help="Google Cloud Project ID.",
    )

    parser.add_argument(
        "--skip_agent_creation",
        dest="create_agent",
        default=True,
        action="store_false",
        help="Flag to skip agent creation. The script will just get the existing agent.",
    )

    parser.add_argument(
        "--skip_intents_creation",
        dest="create_intents",
        default=True,
        action="store_false",
        help="Flag to skip intent creation. The script will just get the existing intents.",
    )

    parser.add_argument(
        "--skip_entities_creation",
        dest="create_entities",
        default=True,
        action="store_false",
        help="Flag to skip entities creation. The script will just get the existing entities.",
    )

    parser.add_argument(
        "--skip_flows_creation",
        dest="create_flows",
        default=True,
        action="store_false",
        help="Flag to skip flow creation. The script will just get the existing flows.",
    )

    parser.add_argument(
        "--skip_pages_creation",
        dest="create_pages",
        default=True,
        action="store_false",
        help="Flag to skip page creation.",
    )

    parser.add_argument(
        "--parentless_folders_as_flows",
        dest="parentless_folders_as_flows",
        default=False,
        action="store_true",
        help="Flag to use parentless folders as flows inside Dialogflow CX.",
    )
    
    args = parser.parse_args()

    main(
        input_paths=args.input,
        display_name=args.bot_name,
        language_code=args.bot_language_code,
        time_zone=args.bot_time_zone,
        location=args.bot_location,
        project_id=args.project_id,
        create_agent=args.create_agent,
        create_intents=args.create_intents,
        create_entities=args.create_entities,
        create_flows=args.create_flows,
        create_pages=args.create_pages,
        parentless_folders_as_flows=args.parentless_folders_as_flows)