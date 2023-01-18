import src.convert.expression_parser as expression_parser

def test_simple_false():
    result = expression_parser.parse_conditions("false")
    assert result.condition == "false"

def test_simple_true():
    result = expression_parser.parse_conditions("true")
    assert result.condition == "true"

def test_simple_intent():
    result = expression_parser.parse_conditions("#test")
    assert result.condition == ""
    assert result.intents[0] == "test"

def test_simple_entity():
    result = expression_parser.parse_conditions("$test")
    assert result.condition == "$session.params.test != null"

    result = expression_parser.parse_conditions("@test")
    assert result.condition == "$session.params.test != null"

def test_entity_equals_value():
    result = expression_parser.parse_conditions("$test:value")
    assert result.condition == "$session.params.test = 'value'"

    result = expression_parser.parse_conditions("@test:value")
    assert result.condition == "$session.params.test = 'value'"

def test_entity_has_value():
    result = expression_parser.parse_conditions("['has test'].contains($test_entity)")
    assert result.condition == "$session.params.test = 'value'"