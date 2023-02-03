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

import time
from typing import List, Dict, Union

from src.convert.expression_translation import translate_conditions, translate_context
from src.convert.tree_creator import create_node_tree
from src.convert.custom_types import Node, Agent

from google.cloud.dialogflowcx_v3beta1 import types, AgentsClient
from google.cloud.dialogflowcx_v3beta1 import EntityTypesClient, FlowsClient 
from google.cloud.dialogflowcx_v3beta1 import IntentsClient, PagesClient
from google.api_core.exceptions import AlreadyExists, FailedPrecondition
from google.api_core.exceptions import InvalidArgument

#========= Dialogflow Constants ===============================================
QUOTA_REQUESTS = 60
QUOTA_REQUESTS_WINDOW_IN_SECONDS = 60


#========= Help Constants =====================================================
REQUESTS_LIMIT = QUOTA_REQUESTS // 2
REQUESTS_REFRESH = QUOTA_REQUESTS_WINDOW_IN_SECONDS * 2





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
   

def create_or_get_agent(
    create_agent: bool,
    requests_per_minute: dict,
    display_name: str,
    language_code: str, 
    time_zone: str, 
    project_id: str,
    location: str
    ) -> types.Agent:
    agents_client = AgentsClient()
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
    ) -> List[types.EntityType]:
    
    entity_types = []
    entity_types_names = []
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
                        parent=f"{str(agent.name)}",
                        entity_type=entity_type)
                )
                entity_types_names.append(entity["entity"])
        print("Finished Entities")
    else:
        wait_for_time(requests_per_minute)
        entity_types = list(entity_type_client.list_entity_types(
            parent=f"{str(agent.name)}"))
    
    return entity_types


def create_or_get_intents(
    create_intents: bool,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    entity_types: List[types.EntityType],
    add_description_as_training_phrase: bool
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
                        parent=f"{str(agent.name)}",
                        intent=intent_obj)
                )

        print("Finished Intents")
    else:
        intents = list(intents_client.list_intents(
            parent=f"{str(agent.name)}"))
    
    return intents


def create_or_get_flows(
    create_flows:bool,
    requests_per_minute: dict,
    bot: Agent,
    agent: types.Agent,
    parentless_folders_as_flows: bool
    ) -> Dict[str, types.Flow]:
    flows = {}
    flows_client = FlowsClient()
    if create_flows:
        if parentless_folders_as_flows:
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
            print("Finished Flows")
            
        else:
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
        wait_for_time(requests_per_minute)
        flows_list = list(flows_client.list_flows(parent=f"{str(agent.name)}"))
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
    agent: types.Agent):
    
    page_dict = page.page
    if page_dict.get("type", "standard") not in [
        "standard", "folder", "response_condition"]:
        print("not the right type")
        return

    if not page_dict.get("conditions", ""):
        print("no condition")
        return

    expression = translate_conditions(page_dict["conditions"])
    
    messages = [
        types.ResponseMessage(
            text=types.ResponseMessage.Text(text=[value])
            ) for value in page_dict.get(
                "output",{"text":{"values":[]}})
                .get("text", {"values":[]})
                .get("values",[])]
    
    parameter_actions = []

    for k, v in page_dict.get("context", {}).items():
        v = translate_context(str(v))
        parameter_actions.append(
            types.Fulfillment.SetParameterAction(
                parameter=k,
                value=v))

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
                parent=f"{flow_id}",
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
                parent=f"{flow_id}",
                page=child_page)
                page_target = True
            except AlreadyExists as e:
                print(e)
                wait_for_time(requests_per_minute)
                pages_map = {
                    page.display_name: page
                    for page in pages_client.list_pages(
                        parent=f"{flow_id}")
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
                    parent = flows_client.update_flow(flow=parent) # type: ignore
                else:
                    parent = pages_client.update_page(page=parent) # type: ignore
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
            agent=agent)


def create_jump_to_routes(
    name: str,
    page: Node,
    pages: Dict[str, dict],
    flows: Dict[str, types.Flow],
    requests_per_minute: dict,
    pages_client: PagesClient,
    flows_client: FlowsClient):

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
    target_page = None
    other_flow = False

    if dialog_node_id in flows:
        flow_id = str(flows[dialog_node_id].name)
    else:
        for key in pages:
            target_page = pages[key]["pages"].get(dialog_node_id, None)
            if target_page:
                page_id = target_page.name
                if key != page.flow_id:
                    other_flow = True
                    flow_id = key
                break
        if not page_id:   
            print("target page id not found")
            return

    parameter_actions = [] 
    messages =[] 
    
    if target_page:
        messages = [
            types.ResponseMessage(
                text=types.ResponseMessage.Text(text=[value])
                ) for value in page_dict.get(
                    "output",{"text":{"values":[]}})
                    .get("text", {"values":[]})
                    .get("values",[])]
        

        for k, v in page_dict.get("context", {}).items():
            
            v = translate_context(str(v))
            parameter_actions.append(
                types.Fulfillment.SetParameterAction(
                    parameter=k,
                    value=v))
    
    if other_flow:
        parameter_actions.append(
            types.Fulfillment.SetParameterAction(
                parameter="target_page",
                value=page_id))

    fullfillment = types.Fulfillment(
        messages=messages if messages else None,
        set_parameter_actions=parameter_actions)

    transition_route = types.TransitionRoute(
        target_page=page_id if page_id and not other_flow else None,
        target_flow=flow_id if flow_id else None,
        trigger_fulfillment=fullfillment,
        condition="true"
    )
    pages[page.flow_id]["pages"][name].transition_routes.append(transition_route) # type: ignore
    wait_for_time(requests_per_minute)
    pages[page.flow_id]["pages"][name] = pages_client.update_page(
        page=pages[page.flow_id]["pages"][name])
    
    if other_flow:
        transition_route = types.TransitionRoute(
            target_page=page_id,
            condition=f"$session.param.target_page={page_id}"
        )
        flow = flows[flow_id]
        flow.transition_routes.append(transition_route)
        flows[flow_id] = flows_client.update_flow(flow=flow)



def create_pages_folders_as_flows(
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    pages: Dict[str, dict],
    intents: List[types.Intent],
    flows_client: FlowsClient,
    pages_client: PagesClient,
    requests_per_minute: dict
    ) -> Dict[str, dict]:
    
    page_nodes, trees = create_node_tree(bot.assuntos[0])
    # Create intent and condition routes
    for name, tree in trees.items():
        flow = ""
        # Start flow
        if name == "start":
            wait_for_time(requests_per_minute)
            flow = flows_client.get_flow(name=str(agent.start_flow))
        elif name in flows:
            flow = flows[name]
        
        if flow:
            for child in tree.children:
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
                    agent=agent)
    
    # Create jump_to routes:
    for name, page in page_nodes.items():
        create_jump_to_routes(
            name=name,
            page=page,
            pages=pages,
            flows=flows,
            requests_per_minute=requests_per_minute,
            pages_client=pages_client,
            flows_client=flows_client)
    
    return pages


def create_pages_multiflow(
    bot: Agent,
    agent: types.Agent,
    flows: Dict[str, types.Flow],
    pages: Dict[str, dict],
    intents: List[types.Intent],
    flows_client: FlowsClient,
    pages_client: PagesClient,
    requests_per_minute: dict
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
                v = translate_context(str(v))
                parameter_actions.append(types.Fulfillment.SetParameterAction(parameter=k, value=v))

            fullfillment = types.Fulfillment(messages=messages, set_parameter_actions=parameter_actions)

            #[TODO] Manage next_steps like jump_to and skip

            # Validate conditions
            conditions = page.get("conditions","")
            if conditions == "":
                continue
            expression = translate_conditions(conditions)
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
                        parent=f"{flow_id}",
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
    parentless_folders_as_flows: bool
    ) -> Dict[str, dict]:
    """
    dialog_node - supported, used as display name
    conditions - partially supported, it does not work with time functions
    parent - supported as route using conditions
    previous_sibling - not supported
    output - supported for text values
    context - partially supported, it does not work with time functions
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
                requests_per_minute=requests_per_minute)
        else:
            return create_pages_multiflow(
                bot=bot,
                agent=agent,
                flows=flows,
                pages=pages, 
                intents=intents,
                flows_client=flows_client,
                pages_client=pages_client,
                requests_per_minute=requests_per_minute)

    else:
        for flow in flows:
            wait_for_time(requests_per_minute)
            pages[str(flows[flow].name)]["pages"] = pages_client.list_pages(
                parent=f"{str(flows[flow].name)}")

    
    return pages

