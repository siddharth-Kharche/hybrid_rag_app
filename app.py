import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
import pandas as pd
from athina.keys import AthinaApiKey, OpenAiApiKey
from athina.loaders import Loader
from athina.evals import RagasContextRelevancy

# Setup OpenAI and Athina API keys
try:
     os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
     os.environ['ATHINA_API_KEY'] = st.secrets['ATHINA_API_KEY']
     OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
     AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))
except KeyError:
    st.error("Please set the OPENAI_API_KEY and ATHINA_API_KEY in streamlit secrets.")

# Set the title of the app
st.title("Hybrid RAG Application")

@st.cache_resource()  # Cache this to avoid reloading every time
def initialize_rag():
     # Load embedding model
     embeddings = OpenAIEmbeddings()

     # Load data
     loader = CSVLoader("./context.csv")
     documents = loader.load()

     # Split documents
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
     documents = text_splitter.split_documents(documents)

     # Create vectorstore
     vectorstore = Chroma.from_documents(documents, embeddings)

     # Create retriever
     retriever = vectorstore.as_retriever()

     # Create keyword retriever
     keyword_retriever = BM25Retriever.from_documents(documents)
     keyword_retriever.k = 3

     # Create ensemble retriever
     ensemble_retriever = EnsembleRetriever(retrievers=[retriever, keyword_retriever], weights=[0.5, 0.5])

     # Create llm
     llm = ChatOpenAI()

      # Create document chain
     template = """
     You are a helpful assistant that answers questions based on the following context.
     If you don't find the answer in the context, just say that you don't know.
     Context: {context}

     Question: {input}

     Answer:

     """
     prompt = ChatPromptTemplate.from_template(template)

     # Setup RAG pipeline
     rag_chain = (
         {"context": ensemble_retriever,  "input": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
     )

     return rag_chain, ensemble_retriever

rag_chain, ensemble_retriever = initialize_rag()

query = st.text_input("Enter your question:")

if query:
     with st.spinner("Generating response..."):
        response = rag_chain.invoke(query)
        st.write("Response:")
        st.write(response)

     # Context retrieval and display
     with st.expander("Retrieved Context", expanded=False):
          contexts = [docs.page_content for docs in ensemble_retriever.get_relevant_documents(query)]
          st.write(contexts)

     # Evaluation
     with st.spinner("Evaluating with Athina..."):
         # Prepare data
        data = {
             "query": [query],
             "response": [response],
            "context": [contexts]
          }
        dataset = Dataset.from_dict(data)
        df = pd.DataFrame(dataset)
        df_dict = df.to_dict(orient='records')
         # Convert context to list
        for record in df_dict:
           if not isinstance(record.get('context'), list):
              if record.get('context') is None:
                 record['context'] = []
              else:
                  record['context'] = [record['context']]
        dataset = Loader().load_dict(df_dict)
        eval_df = RagasContextRelevancy(model="gpt-4o").run_batch(data=dataset).to_df()
        st.write("Evaluation Results")
        st.dataframe(eval_df)