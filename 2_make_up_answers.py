from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

# set the environment variable key OPENAI_API_KEY

# create a LLM
# 0 <= temperature <= 2
llm = ChatOpenAI(name="gpt4o", temperature=1)

# create a prompt
prompt = PromptTemplate.from_template("""
   Answer the following question:

   {question}
""")

#prompt = PromptTemplate.from_template("""
#   Answer the following question:
#
#   {question}
#
#   If you do not know the answer, say I don't know.
#""")

question = input('\nQuestion: ')
if not bool(question):
   exit(0)

prompt_text = prompt.invoke({ 'question': question})

with get_openai_callback() as cb:
   # Invoke the LLM
   response = llm.invoke(prompt_text)

   # print the response
   print(response.content)

   # print the stats for the LLM invocation
   print('\n', cb)
