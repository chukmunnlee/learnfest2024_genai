from os import path

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import DirectoryLoader, UnstructuredEPubLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Utility to load Chroma
def load(db_path = './.chroma_db', load_dir = 'assets') -> VectorStore:
   embed_model = OpenAIEmbeddings()
   # if db_path does not exists then create the database
   if path.exists(db_path):
      print('Loading database...')
      return Chroma(persist_directory=db_path, embedding_function=embed_model)

   # create the database
   print('Create database...')
   loader = DirectoryLoader(path=load_dir, glob='./*.epub', loader_cls=UnstructuredEPubLoader)
   content = loader.load()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
   splits = text_splitter.split_documents(content)
   return Chroma.from_documents(persist_directory=db_path, documents=splits, embedding=embed_model)

def create_retriever_tool_from_vector_store(db_path = './.chroma_db', load_dir = 'assets'):
   retriever = load(db_path=db_path, load_dir=load_dir).as_retriever()
   prompt_search_query = ChatPromptTemplate.from_messages([
      ('system', """
            You are an able assistant. You are here to help the humans with their questions.
            If you don't know the answer, say 'I do not know the answer to that question'"""
      ),
      ('user', """
            Given the following question
               {input}
            look up information relevant to the query"""
      ),
      MessagesPlaceholder(variable_name="chat_history")
   ])
   return [ create_retriever_tool( retriever=retriever, name="Search ebooks",
         description="Search the Selfish Giant story. Always use this tool if the question pertains to the Selfish Giant",
         document_prompt=prompt_search_query) ]
