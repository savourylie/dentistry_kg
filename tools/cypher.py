import streamlit as st
from llm import llm
from graph import graph
from langchain.prompts.prompt import PromptTemplate
# Create the Cypher QA chain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about medical subjects/objects and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Example Cypher Statements:

1. To find a particular subject:
MATCH (s:Subject {{name: "Subject Name"}})
RETURN s.name, s.type

2. To find topics around a subject:
MATCH (s:Subject {{name: "Subject Name"}})-[r:PREDICATE]->(o:Object)
RETURN s.name, r.type, o.name

3. To find more information around a subject:
MATCH (s:Subject)<-[:HAS_SUBJECT]-(k:Knowledge)
RETURN s.name, k.text
```

Schema:
{schema}

Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)


cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)