def generate_response(question, rag_chain):
    """Generates a response using the RAG (retrieval-augmented generation) chain."""
    response = rag_chain.invoke(question)
    return response.split('cutresponsefromhereplease')[-1]
