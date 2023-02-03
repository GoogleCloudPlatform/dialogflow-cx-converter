# Dialogflow CX Converter

A tool to convert a Watson export to Dialogflow CX

## Usage: 

### Command

> python -m src.convert [-h] [--input INPUT [INPUT ...]] [--bot_display_name BOT_NAME] [--bot_language_code BOT_LANGUAGE_CODE] [--bot_time_zone BOT_TIME_ZONE] [--bot_location BOT_LOCATION] [--project_id PROJECT_ID] [--skip_agent_creation] [--skip_intents_creation][--skip_entities_creation] [--skip_flows_creation] [--skip_pages_creation] [--parentless_folders_as_flows]

### Arguments:
>  -h, --help 
- show the help message and exit

>  --input INPUT [INPUT ...]
- Input json to process.

>  --bot_display_name BOT_NAME
- Display name of the bot.

>  --bot_language_code BOT_LANGUAGE_CODE
- Language of the bot.

>  --bot_time_zone BOT_TIME_ZONE
- Time zone of the bot.

>  --bot_location BOT_LOCATION
- Location of the bot.

>  --project_id PROJECT_ID
- Google Cloud Project ID.

>  --skip_agent_creation
- Flag to skip agent creation. The script will just get the existing agent.

>  --skip_intents_creation
- Flag to skip intent creation. The script will just get the existing intents.

>  --skip_entities_creation
- Flag to skip entities creation. The script will just get the existing entities.

>  --skip_flows_creation
- Flag to skip flow creation. The script will just get the existing flows.

>  --skip_pages_creation
- Flag to skip page creation.

>  --parentless_folders_as_flows
- Flag to use parentless folders as flows inside Dialogflow CX.


## Instalation

> python -m pip install -e .

## Test
>  python -m pip install -e .[dev]

### Run unit tests
> pytest