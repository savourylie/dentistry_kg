from typing import Dict, Tuple, List, Union, Callable, Optional
from dataclasses import dataclass
from neo4j import Transaction
import pandas as pd
import nltk
import random
from nltk.corpus import words
import ssl
from tqdm.auto import tqdm


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def get_random_word_of_length(length):
    # nltk.download('words')
    # Get the list of all words
    word_list = words.words()
    
    # Filter words by the desired length
    words_of_length = [word for word in word_list if len(word) == length]
    
    # Choose a random word from the filtered list
    if words_of_length:
        return random.choice(words_of_length)
    else:
        return None


@dataclass
class NodeModel:
    """
    This dataclass is a configuration to define what columns of a dataframe to be used to create nodes and with what columns to be used as properties in Neo4j.

    label (Tuple[str, str]): ('column_name_label', <column_name>) | ('hard_coded_label', <node_label>)
    id_prop (Tuple[str, str]):  (<column_name>, <node_property>)
    properties (Dict[str, str]): {
        "<column1>": "<property1>",
        "<column2>": "<property2>",
        "<column3>": "<property3>",
    }
    """
    label: Union[str, Tuple[str, str]]
    id_prop: Tuple[str, str]
    properties: Dict[str, Tuple[str, Optional[Callable]]]
    extra_labels: List[str] = None


@dataclass
class RelationshipModel:
    """
    This dataclass is a configuration to define what columns of a dataframe to be used to create relationships and with what columns to be used as properties in Neo4j.

    source_node (str): Source node label
    target_node (str): Target node label
    rel_label (Union[str, Tuple[str, str]]): Relationship label
    source_id (Tuple[str, str]): (<column_name>, <node_property>)
    target_id (Tuple[str, str]): (<column_name>, <node_property>)
    extra_labels (List[str]): List of extra labels to be added to the relationship
    """
    source_node: str
    target_node: str
    rel_label: Union[str, Tuple[str, str]]
    source_id: Tuple[str, str]
    target_id: Tuple[str, str]
    properties: Dict[str, str] = None
    extra_labels: List[str] = None


def create_nodes(tx: Transaction, df: pd.DataFrame, nodes: List[NodeModel]) -> None:
    """
    This function creates nodes in Neo4j from a dataframe using the configuration provided in the nodes list.

    Args:
    tx (Transaction): Neo4j transaction object
    df (pd.DataFrame): Dataframe to be used to create nodes
    nodes (List[Node]): List of Node dataclass objects

    Returns:
    None
    """
    for node in nodes:
        rows = df.dropna(subset=[node.id_prop[0]]).to_dict('records')

        if node.label[0] == 'column_name_label':            
            for row in tqdm(rows):
                label = row[node.label[1]]
                id_prop = row[node.id_prop[0]]

                clause_list = []

                for column_name, (property_name, func) in node.properties.items():
                    if not func:
                        clause_list.append(f"n.{property_name} = '{row[column_name]}'")
                    else:
                        clause_list.append(f"n.{property_name} = {func(row[column_name])}")

                on_create_props_string = ', '.join(clause_list)
                # on_create_props_string = ', '.join([f"n.{property_name} = '{row[column_name]}'" for column_name, (property_name, func) in node.properties.items()])

                if node.extra_labels:
                    extra_labels_string = ':' + ':'.join(node.extra_labels)
                else:
                    extra_labels_string = ''

                create_nodes_query = f"""
                MERGE (n:{label}{extra_labels_string} {{{node.id_prop[1]}: '{id_prop}'}})
                ON CREATE SET {on_create_props_string}
                """
                
                # print("Cypher query string: ", create_nodes_query)
                tx.run(create_nodes_query, id_value=row[node.id_prop[0]], **row)

        elif node.label[0] == 'hard_coded_label':
            label = node.label[1]

            for row in tqdm(rows):
                id_prop = row[node.id_prop[0]]

                clause_list = []

                for column_name, (property_name, func) in node.properties.items():
                    if not func:
                        clause_list.append(f"n.{property_name} = '{row[column_name]}'")
                    else:
                        clause_list.append(f"n.{property_name} = {func(row[column_name])}")

                on_create_props_string = ', '.join(clause_list)
                # on_create_props_string = ', '.join([f"n.{property_name} = '{row[column_name]}'" for column_name, (property_name, func) in node.properties.items()])

                if node.extra_labels:
                    extra_labels_string = ':' + ':'.join(node.extra_labels)
                else:
                    extra_labels_string = ''

                create_nodes_query = f"""
                MERGE (n:{label}{extra_labels_string} {{{node.id_prop[1]}: '{id_prop}'}})
                ON CREATE SET {on_create_props_string}
                """
                
                # print("Cypher query string: ", create_nodes_query)
                tx.run(create_nodes_query, id_value=row[node.id_prop[0]], **row)
            # on_create_props_string = ', '.join([f"n.{property_name} = row.{column_name}" for column_name, (property_name, func) in node.properties.items()])

            # label = node.label[1]

            # if node.extra_labels:
            #     extra_labels_string = ':' + ':'.join(node.extra_labels)
            # else:
            #     extra_labels_string = ''

            # create_nodes_query = f"""
            # UNWIND $rows AS row
            # MERGE (n:{label}{extra_labels_string} {{{node.id_prop[1]}: row.{node.id_prop[0]}}})
            # ON CREATE SET {on_create_props_string}
            # """
            
            # tx.run(create_nodes_query, rows=rows)
            
        else:
            raise ValueError("Invalid label type. It should be either 'column_name_label' or 'hard_coded_label'.")
        
        # print("Cypher query string: ", create_nodes_query)


def create_relationships(tx: Transaction, df: pd.DataFrame, relationships: List[RelationshipModel]) -> None:
    """
    This function creates relationships in Neo4j from a dataframe using the configuration provided in the relationships list.

    Args:
    tx (Transaction): Neo4j transaction object
    df (pd.DataFrame): Dataframe to be used to create relationships
    relationships (List[Relationship]): List of Relationship dataclass objects

    Returns:
    None
    """
    for rel in relationships:
        rows = df.dropna(subset=[rel.source_id[0], rel.target_id[0]]).to_dict('records')

        if rel.rel_label[0] == 'column_name_label':
            for row in tqdm(rows):

                rel_label = row[rel.rel_label[1]]
                source_id = row[rel.source_id[0]]
                target_id = row[rel.target_id[0]]

                if rel.extra_labels:
                    # extra_labels_string = ':' + ':'.join(rel.extra_labels)
                    merge_clause = f"MERGE (source)-[r:{rel_label}]->(target)"

                    for label in rel.extra_labels:
                        random_word = get_random_word_of_length(4) + '_r'
                        merge_clause += f"\nMERGE (source)-[{random_word}:{label}]->(target)"
                else:
                    merge_clause = f"MERGE (source)-[r:{rel_label}]->(target)"

                if rel.properties:
                    on_create_props_string = '\nON CREATE SET ' + ', '.join([f"r.{property_name} = '{row[column_name]}'" for column_name, property_name in rel.properties.items()])
                else:
                    on_create_props_string = ''

                create_relationships_query = f"""
                MATCH (source:{rel.source_node} {{{rel.source_id[1]}: '{source_id}'}}),
                    (target:{rel.target_node} {{{rel.target_id[1]}: '{target_id}'}})
                {merge_clause}
                {on_create_props_string}
                """

                # print("Cypher query string: ", create_relationships_query)
                tx.run(create_relationships_query, id_value=row[rel.source_id[0]], **row)

        elif rel.rel_label[0] == 'hard_coded_label':

            rel_label = rel.rel_label[1]

            if rel.extra_labels:
                merge_clause = f"MERGE (source)-[r:{rel_label}]->(target)"

                for label in rel.extra_labels:
                    random_word = get_random_word_of_length(4) + '_r'
                    merge_clause += f"\nMERGE (source)-[{random_word}:{label}]->(target)"
            else:
                merge_clause = f"MERGE (source)-[r:{rel_label}]->(target)"
                
            if rel.properties:
                on_create_props_string = '\nON CREATE SET ' + ', '.join([f"r.{property_name} = row.{column_name}" for column_name, property_name in rel.properties.items()])
            else:
                on_create_props_string = ''

            create_relationships_query = f"""
            UNWIND $rows AS row
            MATCH (source:{rel.source_node} {{{rel.source_id[1]}: row.{rel.source_id[0]}}}),
                (target:{rel.target_node} {{{rel.target_id[1]}: row.{rel.target_id[0]}}})
            {merge_clause}
            {on_create_props_string}
            """
            
            tx.run(create_relationships_query, rows=rows)
        
        # print("Cypher query string: ", create_relationships_query)
