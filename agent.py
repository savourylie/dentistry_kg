from llm import llm
from graph import graph
from tools.cypher import cypher_qa
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.prompts import PromptTemplate
from tools.vector import get_medical_knowledge
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id


chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个医学专家，乐于为大家提供各种医学知识。"),
        ("human", "{input}"),
    ]
)

medicine_chat = chat_prompt | llm | StrOutputParser()


tools = [
    Tool.from_function(
        name="General Chat",
        description="适用于其他工具未涵盖的一般医学话题聊天",
        func=medicine_chat.invoke,
    ),
    Tool.from_function(
        name="Medical Knowledge Search",  
        description="当您需要根据医学知识查找有关药物和症状的信息时",
        func=get_medical_knowledge, 
    ),
    Tool.from_function(
        name="Medical information",
        description="Provide information about medical questions using Cypher",
        func = cypher_qa
    )
]


def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a doctor providing information about medications and symptoms.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that are not related to movies, actors, or directors.
Do not answer any questions using your pre-trained knowledge, only use information provided in context.

Use the language the user prefers and provide information in a way that is easy to understand.                                        

Tools:
-----

You have access to the following tools:

{tools}

To use a tool, use the following format:

```
Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action: Input for the action
Observation: Result of the action
```

When you need to respond to a human, or when you do not need to use a tool, you must use the following format:

```
Thought: Do I need to use a tool? No
Final answer: [enter your response here]
```

Get started!

Previous chat history:

{chat_history}

New input: {input}
{agent_scratchpad}
""")

# agent_prompt = PromptTemplate.from_template("""
# 您是一名医生，提供有关药物和症状的信息。
# 尽可能提供帮助并返回尽可能多的信息。
# 不要回答任何与电影、演员或导演无关的问题。
# 不要使用您预先训练过的知识回答任何问题，仅使用上下文中提供的信息。

# 工具:
# ------

# 您可以访问以下工具：

# {tools}

# 要使用工具，请使用以下格式：

# ```
# Thought：我需要使用工具吗？是的
# Action：要采取的行动，应该是 [{tool_names}] 之一
# Action：行动的输入
# Observation：行动的结果
# ```

# 当您需要向人类做出回应时，或者您不需要使用工具时，您必须使用以下格式：

# ```
# 想法：我需要使用工具吗？不需要
# 最终答案：[此处输入您的回复]
# ```

# 开始！

# 先前的对话记录：
# {chat_history}

# 新输入： {input}
# {agent_scratchpad}
# """)

agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    early_stopping_method='generate'
)


# Create a handler to call the agent


# def generate_response(user_input):
#     """
#     Create a handler that calls the Conversational agent
#     and returns a response to be rendered in the UI
#     """

#     response = chat_agent.invoke(
#         {"input": user_input},
#         {"configurable": {"session_id": get_session_id()}},)

#     return response['output']

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": 'dummy_id'}},
    )

    return response['output']