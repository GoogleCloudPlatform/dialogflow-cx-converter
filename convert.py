import argparse
from html import entities
import json
from typing import List
from google.cloud.dialogflowcx_v3beta1 import types
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

def main(
    input_paths: str,
    output_path: str,
    display_name: str,
    project_id: str,
    creds_path: str) -> None:
    """Main function"""
    assuntos = []
    for input_path in input_paths:
        with open(input_path, "r") as f:
            assuntos.append(Assunto(json.loads(f.read())))
    bot = Bot(assuntos)

    agents = Agents(creds_path=creds_path)
    agents.create_agent(project_id, display_name)

    entity_types = EntityTypes(creds_path=creds_path)
    for assunto in bot.assuntos:
        for entity in assunto.entities:
            entities = []
            for value in entity["values"]:
                entities.append(
                    types.EntityType.Entity(
                        value=value["value"],
                        synonyms=value["synonyms"]
                    )
                )
            entity_type = types.EntityType(
                display_name=entity["entity"],
                kind=types.EntityType.Kind.KIND_MAP,
                auto_explansion_mode=types.EntityType.AutoExpansionMode.AUTO_EXPANSION_MODE_UNSPECIFIED,
                entities=entities,
                excluded_phrases=[],
                enable_fuzzy_extraction=False,
                redact=False
            )
            entity_types.create_entity_type(entity_type)
    
    intents = Intents(creds_path=creds_path)
    # [TODO] Create Intents

    flows = Flows(creds_path=creds_path)
    # [TODO] Create Flows

    pages = Pages(creds_path=creds_path)
    # [TODO] Create Pages

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