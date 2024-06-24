import streamlit as st
from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv 

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# print(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)