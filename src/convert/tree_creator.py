from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field
from src.convert.custom_types import Flow, Node

def create_node_tree(assunto: Flow) -> Tuple[Dict[str, Node], Dict[str, Node]]:
    """Function to create a node tree from Watson Dialog Nodes"""
    start_node = Node()

    # All pages
    pages: Dict[str, Node] = {"start": start_node}

    # Only roots
    roots: Dict[str, Node] = {"start": start_node}


    for page in assunto.pages:
        # Nodes for pages without a parent
        if not page.get("parent", ""):
            if not pages.get(page["dialog_node"], ""):
                parent = None 
                pages[page["dialog_node"]] = Node(
                    page=page,
                    flow_id=page["dialog_node"])
            # If the node already exists, just update the dict
            else:
                pages[page["dialog_node"]].page = page
                pages[page["dialog_node"]].flow_id = page["dialog_node"]
            
            # conversation_start indicates first node
            if "conversation_start" in page.get("conditions", ""):
                pages[page["dialog_node"]].parent = roots["start"]
                roots["start"].children.append(pages[page["dialog_node"]])
            else:
                # Add as root
                roots[page["dialog_node"]] = pages[page["dialog_node"]]
        # Nodes for pages with a parent
        else:
            # If parent node does not exist, creates a placeholder
            if not pages.get(page["parent"], ""):
                pages[page["parent"]] = Node()
            
            parent = pages[page["parent"]]
            flow_id = ""
            parents = []
            
            # If the flow id is defined, updates for all nodes in the tree
            while parent:
                if parent.flow_id:
                    flow_id = parent.flow_id
                    for parent_item in parents:
                        parent_item.flow_id = flow_id
                    break
                parents.append(parent)
                parent = parent.parent

            # Creates a page node or just update it
            if not pages.get(page["dialog_node"], ""):
                pages[page["dialog_node"]] = Node(
                    page=page,
                    flow_id=flow_id)
            else:
                if flow_id:
                    for child in pages[page["dialog_node"]].children:
                        child.flow_id = flow_id
                pages[page["dialog_node"]].page = page
                pages[page["dialog_node"]].flow_id = flow_id

            pages[page["parent"]].children.append(pages[page["dialog_node"]])
        
        if "next_step" in page:
            page_next_step = page["next_step"]
            if page_next_step.get("behavior","") == "jump_to":
                dialog_node_id = page_next_step.get("dialog_node", "")
                if dialog_node_id:
                    if not pages.get(dialog_node_id, ""):
                        pages[dialog_node_id] = Node(
                            jump_to=True)
                    else:
                        pages[page["dialog_node"]].jump_to = True
    
    return pages, roots
   