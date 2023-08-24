# LangChain: Models, Prompts and Output Parsers
Outline
* Direct API calls to OpenAI
* API calls through LangChain:
* Prompts
* Models
* Output parsers

## Chat API : OpenAI
```
#!pip install python-dotenv
#!pip install openai

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]
get_completion("What is 1+1?")

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""
style = """American English \
in a calm and respectful tone
"""
prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

response = get_completion(prompt)


```
## Chat API : LangChain
```
#!pip install --upgrade langchain
from langchain.chat_models import ChatOpenAI
# To control the randomness and creativity of the generated
# text by an LLM, use temperature = 0.0
chat = ChatOpenAI(temperature=0.0, model=llm_model)
chat

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

```



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
* Example generation
* Manual evaluation (and debuging)
* LLM-assisted evaluation
* LangChain evaluation platform

  ```
  # account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"
    
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

from langchain.evaluation.qa import QAGenerateChain
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
examples += new_examples
qa.run(examples[0]["query"])

import langchain
langchain.debug = True

qa.run(examples[0]["query"])
langchain.debug = False

predictions = qa.apply(examples)
from langchain.evaluation.qa import QAEvalChain
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()

  ```

