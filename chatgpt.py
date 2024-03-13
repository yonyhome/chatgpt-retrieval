import os
import sys
from typing import List
 
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import constants
os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False



def initialize_chatbot() -> ConversationalRetrievalChain:
    custom_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer based on context
        {context}
        Eres una experta programadora y tutura para la clase de algoritmia y programacion.
        tu nombre es BerticaBot y debes responder las dudas o inquietudes de los estudiantes usando la informacion sumnistrada.
        Siempre debes ser amigable y animar a los estudiantes a aprender. cuando inicies una conversacion debes presentarte y saludar.
        Original question: {question}"""
    )

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2),
        combine_docs_chain_kwargs={"prompt": custom_template },
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True,
        verbose=True
    )

    return chain


def chat_with_prompt(prompt: str, chat_history: List[tuple] = []) -> str:
    chain = initialize_chatbot()

    result = chain({"question": prompt, "chat_history": chat_history})
    response = result['answer']

    chat_history.append((prompt, response))

    return response

if __name__ == "__main__":
    chat_history = []

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ['quit', 'q', 'exit']:
            sys.exit()
        response = chat_with_prompt(prompt, chat_history)
        print(response)


