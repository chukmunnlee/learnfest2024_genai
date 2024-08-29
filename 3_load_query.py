from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load the the book
content = UnstructuredEPubLoader('assets/monkeys_paw.epub').load()

# split the document
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size=1500, chunk_overlap=50, separators=['.', ';', ' ']
)
splits = text_splitter.split_documents(content)

# Load into vector database, Chroma
# create an embedding model
embed_model = OpenAIEmbeddings()

# load the splits into the database
vector_store = Chroma.from_documents(documents=splits, embedding=embed_model)


query = input('Ask Monkey\'s Paw questions ')
if not bool(query):
   exit(0)

# perform a similarity search
results = vector_store.similarity_search_with_score(
   query=query, k=3)

i = 0
for doc, score in results:
   i += 1
   print(f"{i} {doc.page_content}, SCORE={score}")
   print("\n=======================\n")

