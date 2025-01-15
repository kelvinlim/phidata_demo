from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase, PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

# Create a knowledge base from a PDF

# get list of articles from the pdf directory
# import os
# pdf_dir = 'pdfs'
# pdf_files = os.listdir(pdf_dir)
# pdf_urls = [f'{pdf_dir}/{pdf_file}' for pdf_file in pdf_files]

knowledge_base = PDFKnowledgeBase(
    path= 'pdfs',
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="research_articles",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=SentenceTransformerEmbedder(model="all-mpnet-base-v2"),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load(recreate=True)

agent = Agent(
    model=Ollama(id="llama3.3", api_base="http://localhost:11434"),  # Replace with your Ollama server address
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
    add_datetime_to_instructions=True,
)
# Replace with your question about the research articles
agent.print_response(
    "What are the main findings of the research_articles?", stream=True
)