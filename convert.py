import argparse
import json
import time
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass

from google.cloud.dialogflowcx_v3beta1 import types, FlowsClient, AgentsClient
from google.api_core.exceptions import InvalidArgument
from dfcx_scrapi.core.agents import Agents
from dfcx_scrapi.core.entity_types import EntityTypes
from dfcx_scrapi.core.intents import Intents
from dfcx_scrapi.core.flows import Flows
from dfcx_scrapi.core.pages import Pages


#========= Dialogflow Constants ===============================================
QUOTA_ADMIN_REQUESTS_PER_MINUTE = 60
QUOTA_REFRESH_SECONDS_ADMIN_REQUESTS_PER_MINUTE = 120


#========= Help Constants =====================================================
REQUESTS_LIMIT = QUOTA_ADMIN_REQUESTS_PER_MINUTE
REQUESTS_REFRESH = QUOTA_REFRESH_SECONDS_ADMIN_REQUESTS_PER_MINUTE + 10
REQUESTS_BUFFER = QUOTA_REFRESH_SECONDS_ADMIN_REQUESTS_PER_MINUTE


#========= Classes ============================================================
class Flow():
    """A class to store a Watson workspace"""
    def __init__(self, input_dict: dict) -> None:
        self.workspace = input_dict.get("workspace", input_dict)
        # A map for text outputs
        self.answers = input_dict.get("answers", [])
        self.intents = self.workspace["intents"]
        self.entities = self.workspace["entities"]
        self.pages = self.workspace["dialog_nodes"]


class Agent():
    """A class to store multiple Watson workspaces"""
    def __init__(self, assuntos: List[Flow]) -> None:
        self.assuntos = assuntos


@dataclass
class Node():
    parent: Union["Node",None] = None
    children: List["Node"] = []
    page: dict = {}
    flow_id: str = ""


@dataclass
class Expression:
    operations: Dict[str, str]
    intents: List[str]
    condition: str
    events: List[str]


#========= Help Function ======================================================
def wait_for_time(requests_per_minute: dict):
    """Function to wait for quota"""
    time.sleep(1)
    now = time.time()
    if now - REQUESTS_BUFFER > requests_per_minute["time"]:
        requests_per_minute["time"] = now
        requests_per_minute["request_count"] = 0
    elif requests_per_minute["request_count"] + 1 > REQUESTS_LIMIT:
        time.sleep(REQUESTS_REFRESH-(now-requests_per_minute["time"]))
        requests_per_minute["time"] = time.time()
        requests_per_minute["request_count"] = 0
    else:
        requests_per_minute["request_count"] += 1


def parse_condition_tokens(token: str):
    """
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
    # [TODO] plus minutes, plus hours
    token_type = "unknown"
    converted = token

    if len(token) > 0:

        if ".contains(" in token:
            tokens: List[str] = token[:-1].split(".contains(")
            if tokens != None and len(tokens) == 2:
                _, token_string = parse_condition_tokens(tokens[0]) # type: ignore 
                _, token_substring = parse_condition_tokens(tokens[1]) # type: ignore 
                token_type = "contains"
                converted = f"{token_string} : {token_substring}"
        
        elif "entities.size()" in token:
            token_type = "params_final"
            converted = '($page.params.status = "FINAL" || true)'

        elif token.startswith("intents["):
            token_type = "intents"
            converted = 'true'
        
        elif token[0] == "!":
            token_type = "negation"
            _, converted = parse_condition_tokens(token[1:])
            converted = f"NOT {converted}"

        elif token[0] == "[":
            token_type = "string"
            converted = token[1:-1]

        elif token[0] == '#':
            token_type = "intent"
            converted = token[1:]
        
        elif token[0] == '@' or token[0] == '$':
            if "." in token:
                token_type = "function"
                converted = "0"


            elif ":" in token:
                token_type = "entity_value"
                tokens = token.split(":")
                tokens[1] = tokens[1].replace("(", "").replace(")", "")
                converted = f"$session.params.{tokens[0]} == '{tokens[1]}'"
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
    token_count = 0
    token_map = {}
    while '"' in conditions:
        first = conditions.find('"')
        second = conditions[first + 1:].find('"') + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = conditions[first:second + 1]
        conditions = conditions[:first] + conditions[second+1:]

    return conditions, token_map, token_count


def parse_expression(
    expression: str,
    subexpressions: dict ={},
    subexpression_count: int = 0) -> Tuple[Expression, dict, int]:
    
    intents = []
    events = []

    if len(expression) > 0:
        if expression[-1] == ")":
            expression += " "
        parenthesis_count = expression.count(") ")
        
        start = 0
        end = len(expression)

        while parenthesis_count>=0:

            first_close_parenthesis = expression[start:].find(") ") + start
            if first_close_parenthesis == -1:
                break
            end = min(end, first_close_parenthesis)
            closest_open_parenthesis = expression[:end].rfind("(")
            if closest_open_parenthesis == -1:
                break

            if closest_open_parenthesis > 0:
                if expression[closest_open_parenthesis-1] != " ":
                    parenthesis_count -= 1
                    start = first_close_parenthesis
                    end = closest_open_parenthesis
                    continue
            
            subexpression, subexpressions, subexpression_count = parse_expression(
                expression[closest_open_parenthesis+1:first_close_parenthesis],
                subexpressions=subexpressions,
                subexpression_count=subexpression_count)

            subexpression_key = f"&|({subexpression_count})|&"
            subexpression_count += 1
            subexpressions[subexpression_key] = subexpression

            intents += subexpression.intents

            print(expression)
            print(closest_open_parenthesis)
            print(first_close_parenthesis)

            expression = expression[:closest_open_parenthesis] + " " + subexpression_key + " " + expression[first_close_parenthesis+1:]

            print(expression)
            
            start = 0
            end = len(expression)
            parenthesis_count -= 1
        

        tokens = expression.split()


        converted_tokens = []
        comparison_flag = False
        intent_or_event_flag = False
        remove_next = False
        for token in tokens:
            is_last_comparison = comparison_flag
            is_last_intent_or_event = intent_or_event_flag
            comparison_flag = False
            intent_or_event_flag = False

            if remove_next:
                remove_next = False
                continue
            if token in ["||", "&&", ">", ">=", "<", "<="]:
                if is_last_intent_or_event:
                    remove_next = True
                    continue

                if token == "||":
                    token = "OR"
                elif token == "&&":
                    token = "AND"
                
                converted_tokens.append(token)
                comparison_flag = True
            elif (token.startswith("&|(")):
                converted_tokens.append(token)
            else:
                token_type, converted = parse_condition_tokens(token)
                if token_type in ["no_match", "start"]:
                    events.append(token_type)
                    intent_or_event_flag = True
                    if is_last_comparison:
                        converted_tokens.pop()
                elif token_type == "intent":
                    intents.append(converted)
                    intent_or_event_flag = True
                    if is_last_comparison:
                        converted_tokens.pop()
                elif token_type in ["intents", "params_final", "checkTime", "function"]:
                    intent_or_event_flag = True
                    if is_last_comparison:
                        converted_tokens.pop()
                else:
                    converted_tokens.append(converted)
        
        expression = " ".join(converted_tokens)
    
    expression_obj = Expression(
        operations={},
        intents=intents,
        events=events,
        condition=expression)

    return expression_obj, subexpressions, subexpression_count


def parse_conditions(conditions: str) -> Expression:
    
    conditions = conditions.replace("((", "( (").replace("))", ") )")
    # Tokenize Literals
    conditions, token_map, token_count = tokenize_literals(conditions)

    expression, subexpressions, _ = parse_expression(conditions)

    condition = expression.condition

    while subexpressions:
        converted_tokens = []
        tokens = condition.split()
        for token in tokens:
            if (token.startswith("&|(")):
                converted_tokens.append(subexpressions[token].condition)
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
            
            # Add as root
            roots[page["dialog_node"]] = pages[page["dialog_node"]]
        # Nodes for pages with a parent
        else:
            # If parent node does not exist, creates a placeholder
            if not pages.get(page["parent"], ""):
                pages[page["parent"]] = Node()
            
            parent = pages.get(page["parent"])
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
    
    return pages, roots
        

#========= Creation Functions =================================================
def create_or_get_agent(
    create_agent: bool,
    creds_path: str,
    requests_per_minute: dict,
    display_name: str,
    project_id: str,
    ) -> types.Agent:
    agents_functions = Agents(creds_path=creds_path) # type: ignore
    agents_client = AgentsClient(credentials=agents_functions.creds)
    if create_agent:
        wait_for_time(requests_per_minute)
        agent = agents_client.create_agent(
            agent=types.Agent(
                display_name=display_name,
                default_language_code="pt-br",
                time_zone="America/Buenos_Aires"
            ),
            parent=f"projects/{project_id}/locations/global"
        )
        print(agent)
    else:
        agent = agents_functions.get_agent_by_display_name(
            project_id, display_name)
    return agent


def create_or_get_entity_types(
    create_entities:bool,
    creds_path: str,
    requests_per_minute: dict,
    bot:Agent, 
    agent: types.Agent):
    
    entity_types = []
    entity_types_names = []
    
    entity_types_functions = EntityTypes(creds_path=creds_path)
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
                    entity_types_functions.create_entity_type(
                        agent_id=str(agent.name),
                        obj=entity_type)
                )
                entity_types_names.append(entity["entity"])
        print("Finished Entities")
    else:
        entity_types = entity_types_functions.list_entity_types(
            agent_id=str(agent.name))
    
    return entity_types


def create_or_get_intents(
    create_intents: bool,
    creds_path: str,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    entity_types: List[types.EntityType],
    add_description_as_training_phrase: bool
    ):
    intents = []
    intents_functions = Intents(creds_path=creds_path)
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
                    intents_functions.create_intent(
                        agent_id=str(agent.name),
                        obj=intent_obj)
                )

        print("Finished Intents")
    else:
        intents = intents_functions.list_intents(
            str(agent.name), language_code="pt-br")
    
    return intents


def create_or_get_flows(
    create_flows:bool,
    creds_path:str,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    intents: List[types.Intent],
    intents_as_routes: bool,
    parentless_folders_as_flows: bool
    ) -> Dict[str, types.Flow]:
    flows = {}
    flows_functions = Flows(creds_path=creds_path)
    if create_flows:
        flows_client = FlowsClient(
            credentials=flows_functions.creds)
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
        flows_list = flows_functions.list_flows(str(agent.name))
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

    
    if intents_as_routes:
        start_flow_obj = flows_functions.get_flow(str(agent.start_flow))
        transition_routes = start_flow_obj.transition_routes
        for intent in intents:
            if "00000000-0000-0000-0000-000000000001" in str(intent.name):
                continue
            transition_route = types.TransitionRoute(
                    intent=intent.name,
                    trigger_fulfillment=types.Fulfillment(
                        messages=[
                            types.ResponseMessage(
                                text=types.ResponseMessage.Text(
                                    text=[intent.display_name]
                                )
                            )
                        ]
                    )
                )
            transition_routes.append(transition_route) # type: ignore
        start_flow_obj.transition_routes = transition_routes
        start_flow_obj = flows_functions.update_flow(
                    flow_id=str(start_flow_obj.name),
                    obj=start_flow_obj
                )
    
    return flows


def create_child_page(page, pages, flow_name, flows_functions, pages_functions, flow, agent, requests_per_minute, parent):
    page_dict = page.page
    if page_dict.get("type", "standard") not in [
        "standard", "folder", "response_condition"]:
        return

    if not page_dict.get("conditions", ""):
        return

    expression = parse_conditions(page_dict["conditions"])

    # [TODO] Group Response Condition
    if page_dict.get("type", "standard") == "response_condition":
        pass
    
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
    
    dialog_node_id = page.page["dialog_node"]

    child_page = types.Page(
        display_name = page.page.get(
            "title",
            dialog_node_id)
    )

    wait_for_time(requests_per_minute)
    pages[flow_name][
        "pages"][dialog_node_id] = pages_functions.create_page(
        flow_name,
        child_page)
    
    transition_route = types.TransitionRoute(
        target_page=pages[flow_name]["pages"][dialog_node_id].name,
        trigger_fulfillment=fullfillment,
        condition=expression.condition if expression.condition else None
    )

    parent.transition_routes.append(transition_route) # type: ignore

    wait_for_time(requests_per_minute)

    if type(parent) == types.Flow:
        flows_functions.update_flow(parent.name, parent)
    else:
        pages_functions.update_page(parent.name, parent)

    for child in page.children:
        create_child_page(
            child,
            pages,
            flow_name,
            flows_functions,
            pages_functions,
            flow, agent,
            requests_per_minute,
            pages[flow_name]["pages"][dialog_node_id])


def create_jump_to_routes(name, page, pages, requests_per_minute, pages_functions):
    # [TODO] Create Jump to Flow
    if not pages.get(name, None):
        return
    page_dict = page.page
    if "next_step" not in page_dict:
        return
    page_next_step = page_dict["next_step"]
    if page_next_step.get("behavior","") != "jump_to":
        return
    
    # [TODO] search for id
    page_id = page_next_step.get("dialog_node", "")
    if not page_id:
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
        target_page=page_id,
        trigger_fulfillment=fullfillment,
        condition="true"
    )
    pages[page.flow_id]["pages"][name].transition_routes.append(transition_route) # type: ignore
    wait_for_time(requests_per_minute)
    pages[page.flow_id]["pages"][name] = pages_functions.update_page(
        str(pages[page.flow_id]["pages"][name].name), pages[page.flow_id]["pages"][name])


def create_pages_folders_as_flows(
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    flows_functions: Flows,
    pages_functions: Pages,
    pages: Dict[str, dict],
    requests_per_minute: dict
    ) -> Dict[str, dict]:
    
    page_nodes, trees = create_node_tree(bot.assuntos[0])
    # Create intent and condition routes
    for name, tree in trees.items():
        # Start flow
        if name == "start":
            start_flow = flows_functions.get_flow(str(agent.start_flow))
            start_flow_name = str(start_flow.name)
            for child in tree.children:
                create_child_page(child, pages, start_flow_name, flows_functions, pages_functions, start_flow, agent, requests_per_minute, start_flow)

        # Creation for other  flows
        elif name in flows:
            flow = flows_functions.get_flow(name)
            flow_name = str(flow.name)
            for child in tree.children:
                create_child_page(child, pages, flow_name, flows_functions, pages_functions, flow, agent, requests_per_minute, flow)
    
    # Create jump_to routes:
    for name, page in page_nodes.items():
        create_jump_to_routes(name, page, pages, requests_per_minute, pages_functions)
    
    return pages


def create_pages_multiflow(bot, flows, pages, intents, requests_per_minute, pages_functions, flows_functions) -> Dict[str, dict]:
    for assunto in bot.assuntos:
        is_parent_pages = {page["dialog_node"]: False for page in assunto.pages}
        for page in assunto.pages:
            parent = page.get("parent", "")
            if parent:
                is_parent_pages[parent] = True
        
        
        flow_id = ""
        flow_obj = types.Flow()
        flow_idx = 0
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
                page_response = pages_functions.create_page(
                        flow_id=flow_id,
                        obj=page_obj
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
                        pages_functions.update_page(str(p_page_obj.name), p_page_obj)
                    except InvalidArgument as e:
                        pass
                else:
                    wait_for_time(requests_per_minute)
                    backup_routes = list(flow_obj.transition_routes).copy() # type: ignore
                    flow_obj.transition_routes.extend(routes) # type: ignore
                    
                    try:
                        flows_functions.update_flow(str(flow_obj.name), flow_obj)
                        flows[flow_idx] = flow_obj
                    except InvalidArgument as e:
                        flow_obj.transition_routes = backup_routes # type: ignore
        
        print("Finished flow pages")
    
    return pages


def create_or_get_pages(
    create_pages:bool,
    creds_path: str,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    intents: List[types.Intent],
    parentless_folders_as_flows
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
    pages_functions = Pages(creds_path=creds_path)
    flows_functions = Flows(creds_path=creds_path)

    if create_pages:
        if parentless_folders_as_flows:
            return create_pages_folders_as_flows(bot, agent, flows, flows_functions,pages_functions, pages, requests_per_minute)
        else:
            return create_pages_multiflow(bot, flows, pages, intents, requests_per_minute, pages_functions, flows_functions)

    else:
        for flow in flows:
            pages[str(flows[flow].name)]["pages"] = pages_functions.list_pages(flow_id=str(flows[flow].name))

    
    return pages


# ========= Main ==============================================================

def main(
    input_paths: str,
    output_path: str,
    display_name: str,
    project_id: str,
    creds_path: str,
    create_agent: bool = True,
    create_entities:  bool = True,
    create_intents: bool = True,
    add_description_as_training_phrase: bool = True,
    create_flows: bool = True,
    create_pages: bool = True,
    intents_as_routes: bool = False,
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
    agent = types.Agent(
        display_name=display_name,
        default_language_code="pt-br",
        time_zone="America/Buenos_Aires"
    )
    # Create Agent
    agent = create_or_get_agent(
        create_agent,
        creds_path,
        requests_per_minute,
        display_name,
        project_id,
        )
    
    # Create Entity Types
    entity_types = create_or_get_entity_types(
        create_entities,
        creds_path,
        requests_per_minute,
        bot, 
        agent
        )
    
    # Create Intents
    intents = create_or_get_intents(
        create_intents,
        creds_path,
        requests_per_minute,
        bot,
        agent,
        entity_types,
        add_description_as_training_phrase
        )

    # Create Flows
    flows = create_or_get_flows(
        create_flows,
        creds_path,
        requests_per_minute,
        bot,
        agent,
        intents,
        intents_as_routes,
        parentless_folders_as_flows
        )
    
    # Create Pages
    create_or_get_pages(
        create_pages,
        creds_path,
        requests_per_minute,
        bot,
        agent,
        flows,
        intents,
        parentless_folders_as_flows
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
        "--output",
        dest="output",
        default="dialogflow.json",
        help="Output json.",
    )
    parser.add_argument(
        "--bot_name",
        dest="bot_name",
        default="Converted_Bot",
        help="Display name of the bot.",
    )
    parser.add_argument(
        "--project_id",
        dest="project_id",
        default="egon-ongcp-demos",
        help="Project ID.",
    )
    parser.add_argument(
        "--creds_path",
        dest="creds_path",
        default="credentials.json",
        help="Credentials Path.",
    )
    parser.add_argument(
        "--skip_agent",
        dest="create_agent",
        default=True,
        action="store_false",
        help="Create Agent.",
    )
    parser.add_argument(
        "--skip_intents",
        dest="create_intents",
        default=True,
        action="store_false",
        help="Create Intents.",
    )
    parser.add_argument(
        "--skip_entities",
        dest="create_entities",
        default=True,
        action="store_false",
        help="Create Entities.",
    )
    parser.add_argument(
        "--skip_flows",
        dest="create_flows",
        default=True,
        action="store_false",
        help="Create Flows.",
    )
    parser.add_argument(
        "--skip_pages",
        dest="create_pages",
        default=True,
        action="store_false",
        help="Create Pages.",
    )

    parser.add_argument(
        "--intent_as_routes",
        dest="intents_as_routes",
        default=False,
        action="store_true",
        help="Create Pages.",
    )

    parser.add_argument(
        "--parentless_folders_as_flows",
        dest="parentless_folders_as_flows",
        default=False,
        action="store_true",
        help="Create Pages.",
    )
    
    
    args = parser.parse_args()
    main(
        args.input,
        args.output, 
        args.bot_name,
        args.project_id,
        args.creds_path,
        create_agent=args.create_agent,
        create_intents=args.create_intents,
        create_entities=args.create_entities,
        create_flows=args.create_flows,
        create_pages=args.create_pages,
        intents_as_routes=args.intents_as_routes,
        parentless_folders_as_flows=args.parentless_folders_as_flows)