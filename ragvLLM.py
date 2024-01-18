
import os
import torch
from torch import cuda, bfloat16
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[3]:


model_id = "maximuslee07/llama-2-13b-rockwellautomation"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device_map = "cuda:0"


# In[4]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)
hf_auth = 'HUGGINGFACEHUB_API_TOKEN'
model_config = AutoConfig.from_pretrained(model_id, cache_dir="./cache", quantization_config=bnb_config, use_auth_token=hf_auth)
model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config, cache_dir="./cache", quantization_config=bnb_config, use_auth_token=hf_auth, device_map=device_map)


# In[5]:


model.eval()

print(f'Model is running on: {device_map}')


# In[6]:


tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./cache", use_auth_token=hf_auth, device_map=device_map)
#tokenizer.pad_token = tokenizer.eos_token


# In[7]:


model_stop_tokens = ['### Human:', '### Assistant:']

model_stop_tokens_ids = [tokenizer(x)['input_ids'] for x in model_stop_tokens]

model_stop_tokens_ids


# In[8]:


model_stop_tokens_ids = [torch.LongTensor(x) for x in model_stop_tokens_ids]

model_stop_tokens_ids


# In[9]:


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in model_stop_tokens_ids:
            if torch.eq(input_ids[0].to(device_map)[-len(stop_ids):], stop_ids.to(device_map)).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])


# In[10]:


pipe = pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation', # we pass model parameters here too
    device_map="auto",
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=1024,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)


# In[11]:


res = pipe("Is my PowerFlex 40 AC Drive still available or do I have to convert or Migrate to a replacement product?")
print(res[0]["generated_text"])


# In[12]:


from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipe)

llm(prompt="Is my PowerFlex 40 AC Drive still available or do I have to convert or Migrate to a replacement product?")


# In[13]:


import os

current_directory = os.getcwd()
print(current_directory)


# In[14]:


from langchain.document_loaders import TextLoader
import json

# Define a path to your .jsonl file
file_path = 'rag-vllm-finetune/raqna_train_100.jsonl'

# Function to process each JSON object
def process_json_object(json_obj):
    text = json_obj.get('text', '')
    if ' ### Assistant: ' in text:
        human_text, assistant_text = text.split(' ### Assistant: ')
        human_text = human_text.replace('### Human: ', '').strip()
        assistant_text = assistant_text.replace('### Assistant: ', '').strip()
    else:
        # Handle the case where the delimiter is not found
        human_text = text.replace('### Human: ', '').strip()
        assistant_text = "No response found"

    return {"question": human_text, "answer": assistant_text}


# Load and process the documents
processed_data = []
with open(file_path, 'r') as file:
    for line_number, line in enumerate(file, start=1):
        try:
            start_index = line.index('"text": "') + 9  # Start index of the 'text' field value
            end_index = line.rindex('"', start_index)  # End index of the 'text' field value
            before_text = line[:start_index]  # Content before the 'text' field value
            text_value = line[start_index:end_index]  # 'text' field value
            after_text = line[end_index:]  # Content after the 'text' field value
            corrected_text_value = text_value.replace('\\', '').replace('\\', '').replace('\"', '').replace('"', '').replace("\/", "").replace('\b', '').replace('\f', '').replace('\r', '').replace('\t','') # Remove all json escape characters except for \n

        # Reassemble the line
            corrected_line = before_text + corrected_text_value + after_text
            json_obj = json.loads(corrected_line)
            processed_data.append(process_json_object(json_obj))
        except json.JSONDecodeError as e:
            print(f"Error processing line {line_number}: {e}")
            # Optional: handle the error, skip the line, or take other actions

text_list = [processed_data[i]['question'] + ' ' + processed_data[i]['answer'] for i in range(len(processed_data))]
print(text_list[1])
# get_text_chunks_langchain(' '.join(text_list))
# 'processed_data' now is a list of dictionaries with question and answer


# In[15]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
docs = [Document(page_content=x) for x in text_splitter.split_text(' '.join(text_list))]
print(len(docs))


# In[16]:


# os.environ["max_split_size_mb"] = "128"


# In[17]:


import torch
import gc

# Clear GPU memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    global result
    try: 
        del model, tokenizer, pipeline, docs, text_splitter
    except NameError:
        pass
        
    try:
        del result
    except NameError:
        pass
    
clear_memory()


# In[19]:


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "maximuslee07/llama-2-13b-rockwellautomation"
model_kwargs = {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Store embeddings in a FAISS index
vector_store = FAISS.from_documents(docs, embeddings)


# In[20]:


vector_store.save_local('faq_vector_store')


# In[21]:


faq_vector_store = FAISS.load_local('faq_vector_store', embeddings)


# In[22]:


from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, faq_vector_store.as_retriever())


# In[28]:


chat_history = []

query = "Is my PowerFlex 40 AC Drive still available or do I have to convert or Migrate to a replacement product?"
result = chain({"question": query, "chat_history": chat_history})
print(result['answer'])


# In[24]:


chat_history = [(query, result['answer'])]

query = "CompactLogix 5380 question. When I type the IP address of the PLC in Chrome a web server appears.  Is there a way to see tag values here?"
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])


# In[60]:


from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import PromptTemplate

template = """
  {Your_Prompt}
  
  CONTEXT:
  {context}
  
  QUESTION: 
  {query}

  CHAT HISTORY: 
  {chat_history}
  
  ANSWER:
  
  """

prompt = PromptTemplate(input_variables=["chat_history", "query", "context"], template=template)
query = "Is my PowerFlex 40 AC Drive still available or do I have to convert or Migrate to a replacement product?"
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")

chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt, memory=memory)
print(chain({"query": query, "context": "Rockwell Automation", "chat_history": chat_history, 'input_documents':"", 'Your_Prompt': ""}))

