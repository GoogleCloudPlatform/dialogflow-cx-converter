import src.convert.expression_translation as expression_translation

def test_simple_false():
    result = expression_translation.translate_conditions("false")
    assert result.condition == "false"

def test_simple_true():
    result = expression_translation.translate_conditions("true")
    assert result.condition == "true"

def test_simple_intent():
    result = expression_translation.translate_conditions("#test")
    assert result.condition == ""
    assert result.intents[0] == "test"

def test_simple_entity():
    result = expression_translation.translate_conditions("$test")
    assert result.condition == "$session.params.test != null"

    result = expression_translation.translate_conditions("@test")
    assert result.condition == "$session.params.test != null"

def test_entity_equals_value():
    result = expression_translation.translate_conditions("$test:value")
    assert result.condition == '$session.params.test = "value"'

    result = expression_translation.translate_conditions("@test:value")
    assert result.condition == '$session.params.test = "value"'

def test_entity_has_value():
    result = expression_translation.translate_conditions("['has test'].contains($test_entity)")
    assert result.condition == (
        '$sys.func.CONTAIN([$sys.func.TO_TEXT("has test")], '
        '$session.params.test_entity)')

def test_context_with_text():
    result = expression_translation.translate_context("test")
    assert result == 'test'

def test_context_with_now_function():
    result = expression_translation.translate_context("<? now() ?>")
    assert result == '$sys.func.NOW()'

def test_context_with_function_and_parenthesis():
    result = expression_translation.translate_context(
        "<? ($testData && $testData.test) ||"
        " ($test && $test != \"\") ? true : false ?>")
    print(result)
    assert result == (
        '$sys.func.IF("( $session.params.testData != null'
        ' AND $session.params.testData.test != null ) OR'
        ' ( $session.params.test != null AND '
        '$session.params.test != \'\' )","true","false")'
    )

def test_context_with_double_ternary():
    result = expression_translation.translate_context(
        "<? ($testData && $testData.test) ||"
        " ($test && $test != \"\") ? ($testData2 && $testData2.test) ||"
        " ($test2 && $test2 != \"\") ? true : false : false ?>")
    print(result)
    assert result == (
        '$sys.func.IF("( $session.params.testData != null'
        ' AND $session.params.testData.test != null ) OR'
        ' ( $session.params.test != null AND '
        '$session.params.test != \'\' )",'
        '"$sys.func.IF("( $session.params.testData2 != null'
        ' AND $session.params.testData2.test != null ) OR'
        ' ( $session.params.test2 != null AND '
        '$session.params.test2 != \'\' )","true","false")","false")'
    )

def test_context_with_text_and_function():
    result = expression_translation.translate_context(
        "test = <? $test.contains(\"has_test\") ? test : "
        "test.append('has_test') ?>")
    print(result)
    assert result == (
        '$sys.func.IF("$session.params.test : \'has_test\'", test,'
        ' $sys.func.APPEND($session.params.test, "has_test")')