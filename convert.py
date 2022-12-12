import argparse
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

import proto

from google.cloud.dialogflowcx_v3beta1 import types, FlowsClient, AgentsClient
from dfcx_scrapi.core.agents import Agents
from dfcx_scrapi.core.entity_types import EntityTypes
from dfcx_scrapi.core.intents import Intents
from dfcx_scrapi.core.flows import Flows
from dfcx_scrapi.core.pages import Pages

# Dialogflow Constants
QUOTA_ADMIN_REQUESTS_PER_MINUTE = 60
QUOTA_REFRESH_SECONDS_ADMIN_REQUESTS_PER_MINUTE = 60

# Help Constants
REQUESTS_LIMIT = QUOTA_ADMIN_REQUESTS_PER_MINUTE // 2
REQUESTS_REFRESH = QUOTA_REFRESH_SECONDS_ADMIN_REQUESTS_PER_MINUTE + 10
REQUESTS_BUFFER = QUOTA_REFRESH_SECONDS_ADMIN_REQUESTS_PER_MINUTE // 2

# Classes
class Flow():
    """A class to store a Watson workspace"""
    def __init__(self, input_dict: dict) -> None:
        self.workspace = input_dict.get("workspace", input_dict)
        self.answers = input_dict.get("answers", [])
        self.intents = self.workspace["intents"]
        self.entities = self.workspace["entities"]
        self.pages = self.workspace["dialog_nodes"]

class Agent():
    """A class to store multiple Watson workspaces"""
    def __init__(self, assuntos: List[Flow]) -> None:
        self.assuntos = assuntos

@dataclass
class Expression:
    sub_expressions: Dict[str, 'Expression']
    operations: Dict[str, str]
    intents: List[str]
    condition: List[str]
    events: List[str]

# Help Function
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
    """
    token_type = "unknown"
    converted = ""

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

    elif token[0] == "[":
        token_type = "string"
        converted = token[1:-1]

    elif token[0] == '#':
        token_type = "intent"
        converted = token[1:]
    
    elif token[0] == '@' or token[0] == '$':
        if ":" in token:
            token_type = "entity_value"
            tokens = token.split(":")
            converted = f"$session.params.{tokens[0]} == {tokens[1]}"
        else:
            token_type = "entity"
            converted = f"$session.params.{token[1:]} != null"
    
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
        converted = f"$sys.func.NOW() {operation} {time}"
    
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

def parse_expression(expression: str) -> Tuple[Expression, int, int]:
    
    first_open_parenthesis = expression.find("(")
    sub_expression_count = 0
    sub_expressions ={}
    intents = []
    events = []
    overall_start = len(expression)
    overall_end = 0

    # [TODO] Parse Parenthesis
    while first_open_parenthesis:
        first_close_parenthesis = expression.find(")")
        second_open_parenthesis = expression[first_open_parenthesis + 1:].find("(") + first_open_parenthesis + 1
        if second_open_parenthesis > first_close_parenthesis or second_open_parenthesis == -1:
            sub_expression, start, end = parse_expression(expression[first_open_parenthesis+1:first_close_parenthesis]) # type: ignore
        else:
            sub_expression, start, end = parse_expression(expression[second_open_parenthesis+1:]) # type: ignore
        sub_expressions[f"({sub_expression_count})"] = sub_expression
        intents += sub_expression.intents
        overall_start = min(overall_start, start)
        overall_end = max(overall_end, end)

    else:
        tokens = expression.split()
        converted_tokens = []
        for token in tokens:
            if token in ["||", "&&", ">", ">=", "<", "<="] or token.startswith("("):
                converted_tokens.append(token)
            else:
                token_type, converted = parse_condition_tokens(token)
                if token_type in ["no_match", "start"]:
                    events.append(token_type)
                elif token_type == "intent":
                    intents.append(converted)
                else:
                    converted_tokens.append(converted)

    
    expression_obj = Expression(
        sub_expressions=sub_expressions,
        operations={},
        intents=intents,
        events=events,
        condition=[expression])

    return expression_obj, overall_start, overall_end


def parse_conditions(conditions: str) -> Tuple[Expression, dict, int]:
    
    # Tokenize Literals
    conditions, token_map, token_count = tokenize_literals(conditions)

    expression, _, _ = parse_expression(conditions)

    return expression, token_map, token_count


# Creation Functions
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
    ):
    flows = []
    flows_functions = Flows(creds_path=creds_path)
    if create_flows:
        flows_client = FlowsClient(
            credentials=flows_functions.creds)
        for assunto in bot.assuntos:
            flow_obj = types.Flow(
                display_name=assunto.workspace["name"],
                description=assunto.workspace.get("description",""),
                transition_routes=[],
                event_handlers=[],
                transition_route_groups=[]
            )
            wait_for_time(requests_per_minute)
            flows.append(
                flows_client.create_flow(parent=str(agent.name),flow=flow_obj))
            print("Finished Flows")
    else:
        flows = flows_functions.list_flows(str(agent.name))
    
    start_flow_obj = flows_functions.get_flow(str(agent.start_flow))
    if intents_as_routes:
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

def create_or_get_pages(
    create_pages:bool,
    creds_path: str,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    flows: List[types.Flow],
    intents: List[types.Intent],
    ):
    pages = {}
    pages_functions = Pages(creds_path=creds_path)
    flows_functions = Flows(creds_path=creds_path)
    start_flow_obj = flows_functions.get_flow(str(agent.start_flow))
    if create_pages:
        transitions_flows = start_flow_obj.transition_routes
        for assunto in bot.assuntos:
            flow_id = ""
            flow_obj = types.Flow()
            flow_idx = 0
            flow_display_name = assunto.workspace["name"]
            for i, flow in enumerate(flows):
                if assunto.workspace["name"]==flow.display_name:
                    flow_id = str(flow.name)
                    flow_obj = flow
                    flow_idx = i
            pages[flow_display_name] = {
                "flow_id": flow_id,
                "pages": []
            }
            flow_transitions=[]
            for page in assunto.pages:
                if page.get("parent","") or len(pages[flow_display_name]["pages"]) > 200:
                    print("pulei 1")
                    continue
                intent_id = ""
                conditions = page.get("conditions","")
                if conditions == "":
                    continue
                expression, token_map, token_count = parse_conditions(conditions)
                if not (expression.condition or expression.intents):
                    continue
                # Validate conditions
                messages = [
                    types.ResponseMessage(
                        text=types.ResponseMessage.Text(
                            text=[answer["answer"]]
                        )
                    ) for answer in assunto.answers if (
                        answer["dialog_node"]==page["dialog_node"])]
                messages += [
                    types.ResponseMessage(
                        text=types.ResponseMessage.Text(text=[value])
                        ) for value in page["output"]["text"]["values"]]
                fullfillment = types.Fulfillment(messages=messages)
                page_obj=types.Page(
                    display_name=page["dialog_node"],
                    entry_fulfillment=fullfillment,
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
                pages[flow_display_name]["pages"].append(page_response)
                transition_route = types.TransitionRoute(
                    intent=intent_id,
                    target_page=page_response.name
                )
                transition_route_flow = types.TransitionRoute(
                    intent=intent_id,
                    target_flow=flow_id
                )
                flow_transitions.append(transition_route)
                transitions_flows.append(transition_route_flow) # type: ignore
                # [TODO] Implement Page Transitions      
            
            print("Finished flow pages")
            if flow_transitions:
                wait_for_time(requests_per_minute)
                transition_routes = proto.RepeatedField(
                    proto.MESSAGE,
                    number=4,
                    message=types.page.TransitionRoute,
                )
                transition_routes.extend(flow_transitions) # type: ignore
                flow_obj.transition_routes=transition_routes
                flows[flow_idx] = flow_obj
    else:
        for flow in flows:
            pages[str(flow.name)] = pages_functions.list_pages(flow_id=str(flow.name))
    
    # start_flow_obj.transition_routes=transitions_flows
    # flows_functions.update_flow(
    #     flow_id=agent.start_flow,
    #     obj=start_flow_obj
    # )
    
    return pages

# ============= Main =========================

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
    intents_as_routes: bool = False
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
        )
    
    # Create Pages
    create_or_get_pages(
        create_pages,
        creds_path,
        requests_per_minute,
        bot,
        agent,
        flows,
        intents
        )

    # [TODO] Write output json 

    with open(output_path, "wb") as f:
        f.write(exported_agent.agent_content) # type: ignore


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
        "--no_intent_as_routes",
        dest="intents_as_routes",
        default=True,
        action="store_false",
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
        intents_as_routes=args.intents_as_routes)