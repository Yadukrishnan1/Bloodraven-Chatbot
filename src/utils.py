def generate_response(question, retriever, llm_chain):
    """Generates a response using the RAG (retrieval-augmented generation) chain."""
    rag_chain = {"context": retriever, "question": question}
    response = llm_chain.invoke(rag_chain)
    return response.split('cutresponsefromhereplease')[-1]
