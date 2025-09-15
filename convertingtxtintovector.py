import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

doc= TextLoader('content.txt', encoding='utf8')
document= doc.load()

splitter= RecursiveCharacterTextSplitter(
    separators= ["\n\n", "\n", " ", ""],
    chunk_size= 80,
    chunk_overlap= 20
)

chunks= splitter.split_text(document[0].page_content)

"""for chunk in chunks:
    print(chunk + "\nx---x\n")
    print(len(chunk)) """

embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vector_store= FAISS.from_texts(chunks, embeddings)
if not os.path.exists('vectorstore'):
    os.makedirs('vectorstore')
vector_store.save_local('vectorstore/db_faiss')
print("Vector store created and saved.")




