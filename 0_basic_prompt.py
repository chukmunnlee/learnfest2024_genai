from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

# set the environment variable key OPENAI_API_KEY

# create a LLM
llm = ChatOpenAI(name="gpt4o", temperature=.8)


# create a prompt
#prompt = PromptTemplate.from_template("""
#   Tell me {count} jokes on {topic}. 
#   If the topic is offensive, crude or tasteless, then refrain from telling the joke.
#
#   Rate the jokes between 1 and 10
#""")

# create a prompt
prompt = PromptTemplate.from_template("""
   Tell me {count} jokes on {topic}. 

   Rate the jokes between 1 and 10
""")

count = int(input('\n\nNumber of jokes: '))
topic = input('\nTopic: ')

prompt_text = prompt.invoke({ 'count': count, 'topic': topic})

with get_openai_callback() as cb:
   # Invoke the LLM
   response = llm.invoke(prompt_text)

   # print the response
   print(response.content)

   # print the stats for the LLM invocation
   print('\n', cb)
