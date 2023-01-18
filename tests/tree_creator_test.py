import src.convert.tree_creator as tree_creator

def test_tree_creation_empty():
    input_dict = {
        "intents": [],
        "entities": [],
        "dialog_nodes": []
    }
    flow = tree_creator.Flow(input_dict=input_dict)
    start_dict = {"start": tree_creator.Node()}
    pages, roots = tree_creator.create_node_tree(flow)
    assert pages == start_dict
    assert roots == start_dict

def test_tree_creation_flat():
    input_dict = {
        "intents": [],
        "entities": [],
        "dialog_nodes": [
            {
                "conditions": "false",
                "title": "no_title",
                "dialog_node": "node_test"
            },
            {
                "conditions": "false",
                "title": "no_title1",
                "dialog_node": "node_test1"
            },
            {
                "conditions": "false",
                "title": "no_title2",
                "dialog_node": "node_test2"
            }

        ]
    }
    flow = tree_creator.Flow(input_dict=input_dict)
    pages, roots = tree_creator.create_node_tree(flow)
    assert len(pages) == 4
    assert len(roots) == 4

def test_tree_creation_conversation_start():
    input_dict = {
        "intents": [],
        "entities": [],
        "dialog_nodes": [
            {
                "conditions": "conversation_start",
                "title": "no_title",
                "dialog_node": "node_test"
            },
            {
                "conditions": "false",
                "title": "no_title1",
                "dialog_node": "node_test1"
            },
            {
                "conditions": "false",
                "title": "no_title2",
                "dialog_node": "node_test2"
            }

        ]
    }
    flow = tree_creator.Flow(input_dict=input_dict)
    pages, roots = tree_creator.create_node_tree(flow)
    assert len(pages) == 4
    assert len(roots) == 3

def test_tree_creation_conversation_nested():
    input_dict = {
        "intents": [],
        "entities": [],
        "dialog_nodes": [
            {
                "conditions": "conversation_start",
                "title": "no_title",
                "dialog_node": "node_test"
            },
            {
                "conditions": "false",
                "title": "no_title1",
                "dialog_node": "node_test1"
            },
            {
                "conditions": "false",
                "title": "no_title2",
                "dialog_node": "node_test2",
                "parent": "node_test1"
            }

        ]
    }
    flow = tree_creator.Flow(input_dict=input_dict)
    pages, roots = tree_creator.create_node_tree(flow)
    assert len(pages) == 4
    assert len(roots) == 2