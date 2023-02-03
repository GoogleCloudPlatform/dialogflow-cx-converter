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

from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class Expression:
    """A data class for recreating the Watson expression"""
    operations: Dict[str, str] = field(default_factory=dict)
    intents: List[str] = field(default_factory=list)
    condition: str = ""
    events: List[str] = field(default_factory=list)

def is_number(s: str) -> bool:
    """A help function to check if the string is a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def translate_expression(
    expression: str,
    subexpressions: Dict[str, Expression] ={},
    subexpression_count: int = 0) -> Tuple[Expression, dict, int]:
    """A function to translate expressions from Watson to Dialogflow format"""
    
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
            subexpression, subexpressions, subexpression_count = translate_expression(
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
                token_type, converted = translate_condition_tokens(token)
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

def ternary_walker(expression: str, last_type: str) -> list:
    return_sequence = []
    interrogation_idx = expression.find("?")
    colon_idx = expression.find(":")
    if interrogation_idx > 0 and interrogation_idx < colon_idx:
            path_type = "condition"
            condition, rest = expression.split("?", maxsplit=1)
            condition = translate_conditions(condition).condition.strip()
            return_sequence.append({"type": path_type, "value": condition})
            return_sequence += ternary_walker(
                rest, last_type=path_type)
    else:
        if expression.count(":") == 1:
            true, false = expression.split(":", maxsplit=1)
            true = translate_conditions(true).condition
            false = translate_conditions(false).condition
            return_sequence.append({"type": "true", "value": true})
            return_sequence.append({"type": "false", "value": false})


        else:
            if last_type in ["true", "false"]:
                path_type = "false"
            else:
                path_type = "true"
            statement, rest = expression.split(":",maxsplit=1)
            statement = translate_conditions(statement).condition
            return_sequence.append({"type": path_type, "value": statement})
            return_sequence += ternary_walker(rest, path_type)

    print(expression)
    print(return_sequence)
    return return_sequence


def translate_ternary(expression: str) -> str:
    def subcondition(sequence: List[dict], idx: int) -> Tuple[str, int]:
        condition = ""
        true = ""
        false = ""
        if len(sequence) - 1 < idx + 2:
            return sequence[idx]["value"], idx
        if sequence[idx]["type"] == "condition":
            condition = sequence[idx]["value"]
            if sequence[idx+1]["type"] == "true":
                true = sequence[idx+1]["value"]
                if sequence[idx+2]["type"] == "false":
                    false = sequence[idx+2]["value"]
                    idx += 2
                else:
                    false, idx = subcondition(sequence=sequence, idx=idx+2)
            else:
                true, idx = subcondition(sequence=sequence, idx=idx+1)
                false, idx = subcondition(sequence=sequence, idx=idx+1)

        
        return f'$sys.func.IF("{condition}","{true}","{false}")', idx


    sequence = ternary_walker(expression=expression, last_type="")
    print(sequence)

    parameter, _ = subcondition(sequence=sequence, idx=0)
    
    return parameter




def translate_context(context: str) -> str:
    if not "<?" in context or not "?>" in context:
        return context
    
    
    context = context[context.find("<?")+2:context.find("?>")]
    context, token_map, token_count = tokenize_literals(context)
    if "?" in context:
        context = translate_ternary(context)
    else:
        context = translate_conditions(context).condition
    
    
    # [TODO] Check for functions in the result


    if token_count:
        for k, v in token_map.items():
            context = context.replace(k,v)

    return context


def translate_conditions(conditions: str) -> Expression:
    """Function to translate Watson condition into a Dialogflow condition"""
    
    conditions = conditions.replace("((", "( (").replace("))", ") )")
    
    # Tokenize Literals
    conditions, token_map, token_count = tokenize_literals(conditions)

    expression, subexpressions, _ = translate_expression(conditions, {}, 0)

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
        
        # Probably a bug, it needs to be double quote
        while "$sys.func.TO_TEXT('" in condition:
            first = condition.find("$sys.func.TO_TEXT('") + 18
            second = condition[first + 1:].find("'") + first + 1
            condition_as_char_list = list(condition)
            condition_as_char_list[first] = '"'
            condition_as_char_list[second] = '"'
            condition = "".join(condition_as_char_list) # type: ignore    
    
    expression.condition = condition

    return expression


def translate_condition_tokens(token: str):
    """
    A help function to translate condition tokens
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
                item_list = tokens[0].replace("[","").replace("]","").split(",")

                
                converted_list = []

                for item in item_list:
                    converted_token = translate_condition_tokens(item)[1]

                    # Probably a bug, we need to use TO_TEXT when the string has
                    # a space char
                    converted_item = f'$sys.func.TO_TEXT({converted_token})'
                    converted_list.append(converted_item)
                
                _, converted_contain = translate_condition_tokens(tokens[1])
                converted = (
                    f"$sys.func.CONTAIN([{','.join(converted_list)}], "
                    f"{converted_contain})")
            else:
                tokens: List[str] = token[:-1].split(".contains(")
                if tokens != None and len(tokens) == 2:
                    _, token_string = translate_condition_tokens(tokens[0]) # type: ignore 
                    _, token_substring = translate_condition_tokens(tokens[1]) # type: ignore 
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
            token_type, converted = translate_condition_tokens(token[1:])
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
                converted = f'$session.params.{tokens[0][1:]} = "{tokens[1]}"'
            else:
                token_type = "entity"
                converted = f"$session.params.{token[1:]}"
        
        elif token.startswith("now()"):
            token_type = "checkTime"
            operation = ""
            converted = ""
            time = token.split("(")[-1][:-1]
            if "sameOrAfter" in token:
                operation = ">="
            elif "after" in token:
                operation = ">"
            elif "sameOrBefore" in token:
                operation = "<="
            elif "before" in token:
                operation = "<"
            else:
                converted = "$sys.func.NOW()"
                token_type = "now"
            
        
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


def tokenize_literals(expression: str) -> Tuple[str, dict, int]:
    """A function to tokenize watson literals with the format: &|number|"""
    token_count = 0
    token_map = {}
    while '"' in expression:
        first = expression.find('"')
        second = expression[first + 1:].find('"') + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = expression[first:second + 1].replace('"', "'")
        expression = expression[:first] + token_key + expression[second+1:]
    
    while "'" in expression:
        first = expression.find("'")
        second = expression[first + 1:].find("'") + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = expression[first:second + 1]
        expression = expression[:first] + token_key + expression[second+1:]
    
    while ':(' in expression:
        first = expression.find(':(')
        second = expression[first + 1:].find(')') + first + 1
        token_key = f"&|{token_count}"
        token_count += 1
        token_map[token_key] = expression[first+2:second]
        expression = expression[:first+1] + token_key + expression[second+1:]

    return expression, token_map, token_count

