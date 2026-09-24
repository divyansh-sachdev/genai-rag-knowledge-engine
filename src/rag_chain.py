from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

from .config import settings
from .ingest import load_index

PROMPT_TEMPLATE = """You are an internal knowledge assistant. Answer only from the
provided context. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


def build_chain() -> ConversationalRetrievalChain:
    store = load_index()
    retriever = store.as_retriever(search_kwargs={"k": settings.top_k})

    llm = ChatOllama(model=settings.llm_model, temperature=0.1, streaming=True)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
