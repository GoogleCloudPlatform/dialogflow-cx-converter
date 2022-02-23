import argparse
import json
import time
from typing import List
from google.cloud.dialogflowcx_v3beta1 import types, FlowsClient, AgentsClient
from dfcx_scrapi.core.agents import Agents
from dfcx_scrapi.core.entity_types import EntityTypes
from dfcx_scrapi.core.intents import Intents
from dfcx_scrapi.core.flows import Flows
from dfcx_scrapi.core.pages import Pages

creds_path = "credentials.json"

class Assunto():
    def __init__(self, input_dict: dict) -> None:
        self.workspace = input_dict["workspace"]
        self.answers = input_dict["answers"]
        self.intents = self.workspace["intents"]
        self.entities = self.workspace["entities"]
        self.pages = self.workspace["dialog_nodes"]

class Bot():
    def __init__(self, assuntos: List[Assunto]) -> None:
        self.assuntos = assuntos

def wait_for_time(requests_per_minute: dict):
    time.sleep(1)
    now = time.time()
    if now - 30 > requests_per_minute["time"]:
        requests_per_minute["time"] = now
        requests_per_minute["request_count"] = 0
    elif requests_per_minute["request_count"] + 1 > 30:
        time.sleep(60-(now-requests_per_minute["time"]))
        requests_per_minute["time"] = time.time()
        requests_per_minute["request_count"] = 0
    else:
        requests_per_minute["request_count"] += 1

def main(
    input_paths: str,
    output_path: str,
    display_name: str,
    project_id: str,
    creds_path: str) -> None:
    """Main function"""
    requests_per_minute={
        "request_count": 0,
        "time": time.time()
    }
    assuntos = []
    entity_types = []
    entity_types_names = []
    intents = []
    flows = []
    pages = {}
    for input_path in input_paths:
        with open(input_path, "r") as f:
            assuntos.append(Assunto(json.loads(f.read())))
    #bot = Bot([])
    bot = Bot(assuntos)

    # Create Agent
    agents_functions = Agents(creds_path=creds_path)
    agents_client = AgentsClient(credentials=agents_functions.creds)
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
    # agent = agents_functions.get_agent_by_display_name(project_id, display_name)
    # Create Entity Types
    entity_types_functions = EntityTypes(creds_path=creds_path)
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
                    agent_id=agent.name,
                    obj=entity_type)
            )
            entity_types_names.append(entity["entity"])
    print("Finished Entities")

    # Create Intents
    intents_functions = Intents(creds_path=creds_path)
    intents = intents_functions.list_intents(agent.name, language_code="pt-br")
    for assunto in bot.assuntos:
        for intent in assunto.intents:
            training_phrases = []
            entity_ids=[]
            parameters=[]
            for example in intent["examples"]:
                parts=[]
                if "mentions" in example:
                    # Not found in any example to test
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

            intent_obj = types.Intent(
                display_name=intent["intent"],
                training_phrases=training_phrases,
                parameters=parameters,
                priority=500000, # Normal is 500000
                is_fallback=False,
                labels=[],
                description=intent["description"]
            )
            wait_for_time(requests_per_minute)
            intents.append(
                intents_functions.create_intent(
                    agent_id=agent.name,
                    obj=intent_obj)
            )

    print("Finished Intents")
    # Create Flows
    flows_functions = Flows(creds_path=creds_path)
    flows_client = FlowsClient(
        credentials=flows_functions.creds)
    flows = flows_functions.list_flows(agent.name)
    for assunto in bot.assuntos:
        flow_obj = types.Flow(
            display_name=assunto.workspace["name"],
            description=assunto.workspace["description"],
            transition_routes=[],
            event_handlers=[],
            transition_route_groups=[]
        )
        wait_for_time(requests_per_minute)
        flows.append(flows_client.create_flow(parent=agent.name,flow=flow_obj))
    
    print("Finished Flows")
    # Create Pages
    # bot = Bot(assuntos)
    pages_functions = Pages(creds_path=creds_path)
    # parents={}
    transitions_flows = []
    for assunto in assuntos:
        flow_id = ""
        flow_obj = None
        flow_idx = 0
        flow_display_name = assunto.workspace["name"]
        for i, flow in enumerate(flows):
            if assunto.workspace["name"]==flow.display_name:
                flow_id = flow.name
                flow_obj = flow
                flow_idx = i
        pages[flow_display_name] = {
            "flow_id": flow_id,
            "pages": []
        }
        flow_transitions=[]
        for page in assunto.pages:
            if page["parent"] or len(pages[flow_display_name]["pages"]) > 200:
                continue
            intent_id = ""
            c_symbol = page["conditions"][0]
            if c_symbol == "#":
                for intent in intents:
                    if intent.display_name == page["conditions"][1:]:
                        intent_id = intent.name
                        print("Found intent for page")
            if not intent_id:
                continue
            fullfillment = types.Fulfillment(
                messages=[
                    types.ResponseMessage(
                        text=types.ResponseMessage.Text(
                            text=[
                                answer["answer"],
                                page["output"]["text"]["values"][0]
                                ]
                        )
                    ) for answer in assunto.answers if (
                        answer["dialog_node"]==page["dialog_node"])
                ]
            )
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
            transitions_flows.append(transition_route_flow)
        
        print("Finished flow pages")
        if flow_transitions:
            wait_for_time(requests_per_minute)
            flow_obj.transition_routes=flow_transitions
            agent.start_flow
            flow_obj = flows_functions.update_flow(
                flow_id=flow_id,
                obj=flow_obj
            )
            flows[flow_idx] = flow_obj
    
    flow_obj = flows_functions.get_flow(agent.start_flow)
    flow_obj.transition_routes=transitions_flows
    flows_functions.update_flow(
        flow_id=agent.start_flow,
        obj=flow_obj
    )

    # [TODO] Write output json
    with open(output_path, "w") as f:
        f.write("")


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
    
    args = parser.parse_args()
    main(
        args.input,
        args.output, 
        args.bot_name,
        args.project_id,
        args.creds_path)