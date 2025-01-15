from phi.agent import Agent
#from phi.model.openai import OpenAIChat
from phi.model.ollama import Ollama
#from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder

from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=SentenceTransformerEmbedder(model="all-mpnet-base-v2"),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load(recreate=False)

agent = Agent(
     model=Ollama(id="llama3.3", api_base="http://localhost:11434"),  # Replace with your Ollama server address

    # model=OpenAIChat(id="gpt-4o"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("How do I make chicken and galangal in coconut milk soup", stream=True)
