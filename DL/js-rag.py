# venv is chromenv
"""Это не тот раг, который в нашем проекте, но с теми же данными. Тот, что в проекте - работа Кати"""
import torch.nn.functional as F
import chromadb
import pandas as pd
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import pipeline
from chonkie import RecursiveChunker
from openai import OpenAI
from dotenv import load_dotenv
import os

# открываем файл
df = pd.read_csv('dataset_js.csv')
texts = []
for one in df["text"]:
    texts.append(one)

#делаем датасет
dataset = []
for i, text in enumerate(texts):
    doc = {
        "id": i,
        "text": text
    }
    dataset.append(doc)


def chunk_dataset(dataset):
    '''Делит на чанки'''
    # Initialize the chunker
    chunker = RecursiveChunker()
    chunks = []
    text_chunks = []

    for doc in dataset:
        doc_chunks = chunker(doc['text'])
    
        # Добавляем все чанки документа в общий список text_chunks
        text_chunks.extend(doc_chunks)

        for i, chunk in enumerate(doc_chunks):
            chunk_data = {
                    'id': f'chunk_{i}',
                    'original_doc_id': doc['id'],
                    'text': chunk.text,
                    'size_tokens': chunk.token_count,
                    }
            chunks.append(chunk_data)

#   print(len(text_chunks))

    return chunks

final_chunks = chunk_dataset(dataset)
#print(final_chunks[555])
#print(len(final_chunks))


# усредняющий пулинг 
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

input_texts = []

for chunk in final_chunks:
    input_texts.append("passage:" + chunk['text'])
print(len(input_texts))
input_texts = input_texts[0:120]


tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:3] @ embeddings[3:].T) * 100
#print(scores.tolist())


'''Векторная бд'''
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection( # default is all-MiniLM-L6-v2
    name="test_collection",
    configuration={
        "hnsw": {
            "space": "cosine",
            "batch_size": 48
        }
    }
    )

nembeddings = embeddings.detach().numpy()
documents=[chunk['text'] for chunk in final_chunks]
documents=documents[0:120]
# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=documents, # we handle tokenization, embedding, and indexing automatically. I skip that and add my own embeddings 
    embeddings=nembeddings, # filter on these!
    ids=[str(i) for i in range(120)] 
    )

query = "cycle"

def vectorize_query(query):
    input_texts = [f'query: {query}']
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    query_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    return query_embeddings.detach().numpy()

query_embeddings = vectorize_query(query=query)

# Query/search 3 most similar results. You can also .get by id
results = collection.query(
    # query_texts=["This is a query document"],
    query_embeddings=query_embeddings,
    include=["documents", "metadatas", "embeddings"],
    n_results=3,
)

print(results)



context = results['documents']

user_prompt = "How do I write a cycle in Javascript?"


def generate_answer(context, user_prompt):
    '''Генерирует ответ на вопрос на основе полученных чанков'''
    llm = pipeline(
        "text-generation",
        model="bigscience/bloom-1b7",
        max_new_tokens=512,
        model_kwargs={
            #"quantization_config": quantization_config,
            "device_map": "auto",
            "dtype": torch.float16
        }
        )
   
    prompt = f"You are an coding expert. You must answer the user's question: {user_prompt}. Create a relevant, concise answer with the help of the given context: {context}. DO NOT repeat the question or the contexts, create a new, relevant answer. "
    response = llm(prompt)[0]['generated_text']
    return response



def generate_answer(context, user_prompt):
    '''Или с гроком '''
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )
    response = client.responses.create(
    input=f"You are an coding expert. You must answer the user's question: {user_prompt}. Create a relevant, concise answer with the help of the given context: {context}. DO NOT repeat the question or the contexts, create a new, relevant answer.",
    model="openai/gpt-oss-20b"
    )
    return response.output_text


response = generate_answer(context, user_prompt)
print(response)