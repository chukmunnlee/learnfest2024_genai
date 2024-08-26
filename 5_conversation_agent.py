from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits.load_tools import load_tools

from langchain_community.callbacks import get_openai_callback

from langchain.agents import AgentExecutor, create_react_agent

from load_vector_db import create_retriever_tool_from_vector_store

llm = ChatOpenAI(name="gpt4o", temperature=0.1, verbose=True, streaming=False)

tool_names = [ 'ddg-search', 'wikipedia', 'arxiv', 'stackexchange', 'llm-math' ]

# Create tools from retrievers
retriever_tools = create_retriever_tool_from_vector_store()
# Load tools
tools = load_tools(llm=llm, tool_names=tool_names, tools=retriever_tools)

prompt_template_text = '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
Previous conversation history: {chat_history}
'''

prompt = PromptTemplate.from_template(prompt_template_text)

# Configure agent's memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory,
      verbose=True, handle_parsing_errors=True)

while True:
   with get_openai_callback() as cb:
      print('============================\n')
      try:
         question = input('Question: ')
         if not bool(question):
            break
         result = executor.invoke({ 'input': question })
         print(result['output'])

      except ValueError as err:
         print(f'ERROR: {err}')

      print('\n\n---------------\n', cb)

