
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# class to capture the ouput from the LLM
class DiscussionQuestions(BaseModel):
   questions: list[str] = Field(description="List of discussion question")
   answers: list[str] = Field(description="List of answers corresponding the the above question")

# load the book
loader = UnstructuredEPubLoader('assets/the_selfish_giant.epub')
story = loader.load()

# LLM
llm = ChatOpenAI(name="gpt4o", temperature=.5)

# Pass the class to the LLM
llm_structured = llm.with_structured_output(DiscussionQuestions)

# create the prompt
prompt = ChatPromptTemplate.from_messages([
   ('system', """
      You are an able assistant. Your goal is to understant the story and 
      generate a set of discussion questions for literiture class.
   """),
   ('human', """
      Generate {count} questions from the following story:
    
      {story}
   """)
])
prompt_text = prompt.invoke({ 
   'count': 5,
   'story': story
})

# invoke the llm
with get_openai_callback() as cb:
   # result is DiscussionQuestions class
   result = llm_structured.invoke(prompt_text)

   for i in range(len(result.questions)):
      print(f'{i + 1}: {result.questions[i]}')
      print(f'   {result.answers[i]}')

   print('\n', cb)
