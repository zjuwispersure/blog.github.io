

# Memory
* ConversationBufferMemory 
* ConversationBufferWindowMemory 
* ConversationTokenBufferMemory 
* ConversationSummaryMemory 

# LangChain: Q&A over Documents
方法1:
```
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.indexes import VectorstoreIndexCreator

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
response = index.query(query)
display(Markdown(response))


loader = CSVLoader(file_path=file)
docs = loader.load()
embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)

retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0)
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.")

display(Markdown(response))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)
display(Markdown(response))
```


长文本
* Stuff
* Map_reduce
* Refine
* Map_rerank


# LangChain: Evaluation
Outline:
Example generation
Manual evaluation (and debuging)
LLM-assisted evaluation
LangChain evaluation platform
