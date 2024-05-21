import streamlit as st

import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_resource
def get_data():
    #read txt file
    with open("href_list.txt", "r") as file:
        text = file.read()

    #split text into list of urls
    web_urls = text.split("\n")

    pdf_urls = []
    for url in web_urls:
        if url == "":
            web_urls.remove(url)

        #if url is a pdf, add it to the pdf_urls list
        if url.endswith(".pdf") > 0:
            pdf_urls.append(url)
            web_urls.remove(url)
    #remove empty strings
    web_urls = list(filter(None, web_urls))

    return web_urls, pdf_urls


@st.cache_resource
def init():
    if not os.path.exists("./chroma_db"):
        # os.makedirs("./chroma_db")
        web_urls, pdf_urls = get_data()
        pages = []
        for url in pdf_urls:
            loader = PyPDFLoader(url)
            pages += loader.load()

        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=(web_urls),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("entry-header", "entry-content")
                )
            ),
        )
        docs = loader.load()
        docs += pages

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings(), persist_directory="./chroma_db")
    else:
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=GPT4AllEmbeddings())

    return vectorstore

@st.cache_resource
def get_retriever(_vector):
    # retriever = _vector.as_retriever(
    #     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    # )

    # if retriever != 0 :
    #     return retriever

    # retriever = _vector.as_retriever( k = 1)

    # return retriever
    return _vector.as_retriever(k=5)


@st.cache_resource
def create_rag_chain(_retriever):
    ### Contextualize question ###
    contextualize_q_system_prompt = """Compte tenu de l'historique des discussions et de la dernière question de l'utilisateur \
    qui pourrait faire référence au contexte dans l'historique des discussions, formuler une question autonome \
    qui peut être compris sans l'historique des discussions. Ne répondez PAS à la question, \
    reformulez-le simplement si nécessaire et sinon renvoyez-le tel quel."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, _retriever, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt = """Vous êtes assistant pour les tâches de réponses aux questions. \
    Utilisez les éléments de contexte récupérés suivants pour répondre à la question. \
    Si vous ne connaissez pas la réponse, dites simplement que vous ne la savez pas. \
    Utilisez trois phrases maximum et gardez la réponse concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def chat_bot(question):
    result=conversational_rag_chain.invoke(
    {"input": question},
    config={"configurable": {"session_id": "abc123"}},)

    # Extract answer from result
    answer = result["answer"]

    # Extract sources from result
    # sources = "\n - ".join(document.metadata["source"] for document in result["context"])
    first_source = result["context"][0].metadata["source"] if result["context"] else "No source available"

    return answer + "\n\nSource : \n - " + first_source #sources


def launch_streamlit():
    st.title("💬 Chatbot")
    st.caption("🚀 Chatbot de l'UQAC")


    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Comment est ce que je pourrais vous aider?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


    if prompt := st.chat_input(placeholder="Votre message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg = chat_bot(prompt)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


if __name__ == "__main__":
    llm = Ollama(
        model="llama3"
    )

    vectorstore = init()

    retriever = get_retriever(vectorstore)

    rag_chain = create_rag_chain(retriever)

    store = {}

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    launch_streamlit()
