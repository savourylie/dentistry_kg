import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="knowledge",                 # (3)
    node_label="Knowledge",                      # (4)
    text_node_property="text",               # (5)
    embedding_node_property="text_embedding"
)

# neo4jvector = Neo4jVector.from_existing_index(
#     embeddings,                              # (1)
#     graph=graph,                             # (2)
#     index_name="knowledge",                 # (3)
#     node_label="Knowledge",                      # (4)
#     text_node_property="text",               # (5)
#     embedding_node_property="text_embedding", # (6)
#     retrieval_query="""
# RETURN
#     node.text AS text,
#     score,
#     {
#         subject: (node)-[:HAS_SUBJECT]->(subject) | subject.name,
#         predicate: (node)-[:HAS_SUBJECT]->(subject)-[p:PREDICATE]->(object) | p.name,
#         object: (node)-[:HAS_SUBJECT]->(subject)-[:PREDICATE]->(object) | object.name
#     } AS metadata
# """
# )

# Create the retriever
retriever = neo4jvector.as_retriever()

instructions = (
    "使用给定的上下文来回答问题。"
    "如果你不知道答案，就说你不知道。"
    "上下文: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
knowledge_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

# Create a function to call the chain
def get_medical_knowledge(input):
    return knowledge_retriever.invoke({"input": input})