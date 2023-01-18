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
from typing import List

from src.convert.creation_functions import create_or_get_agent, create_or_get_entity_types, create_or_get_flows, create_or_get_intents, create_or_get_pages
from src.convert.custom_types import Flow, Agent

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
        agent=agent
        )
    
    # Create Intents
    intents = create_or_get_intents(
        create_intents=create_intents,
        requests_per_minute=requests_per_minute,
        bot=bot,
        agent=agent,
        entity_types=entity_types,
        add_description_as_training_phrase=add_description_as_training_phrase
        )

    # Create Flows
    flows = create_or_get_flows(
        create_flows=create_flows,
        requests_per_minute=requests_per_minute,
        bot=bot,
        agent=agent,
        parentless_folders_as_flows=parentless_folders_as_flows
        )
    
    # Create Pages
    create_or_get_pages(
        create_pages=create_pages,
        requests_per_minute=requests_per_minute,
        bot=bot,
        agent=agent,
        flows=flows,
        intents=intents,
        parentless_folders_as_flows=parentless_folders_as_flows
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