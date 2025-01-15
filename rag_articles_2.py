from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

# 1. Create a list of PDF file paths
pdf_files = [
    "pdfs/Chen et al. 2024 - Neuroimaging studies of resting-state functional magnetic resonance imaging in eating disorders.pdf",
    "pdfs/Laczkovics et al. 2022 - White matter integrity is disrupted in adolescents with acute anorexia nervosa - A diffusion tensor imaging study.pdf",
    "pdfs/Muratore et al. 2024 - Reduced dorsal fronto-striatal connectivity at rest in anorexia nervosa.pdf",
    "pdfs/Sudo et al. 2024 - Comprehensive elucidation of resting-state functional connectivity in anorexia nervosa by a multicenter cross-sectional study.pdf",
]

# 2. Initialize the embedder
embedder = SentenceTransformerEmbedder(model_name="all-mpnet-base-v2") 

# 3. Initialize LanceDB
# vector_db = LanceDB(
#     embedder=embedder, 
#     connection_string="lancedb://path/to/your/lancedb"  # Specify the path to your LanceDB directory
# )
vector_db=LanceDb(
        table_name="research_articles",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=SentenceTransformerEmbedder(model="all-mpnet-base-v2"),
    )
# 4. Create a PDF knowledge base
knowledge_base = PDFKnowledgeBase(
    path='pdfs',
    vector_db=vector_db,
)

# 5. Load the knowledge base (this will process and store the PDF data)
knowledge_base.load()

# 6. Initialize the LLaMaIndex model
llm = Ollama(id="llama3.3", api_base="http://localhost:11434") 

# 7. Create an agent
agent = Agent(
    model=llm,
    knowledge=knowledge_base,
)

# 8. Ask for a summary
response = agent.print_response("Please summarize the key points from the PDFs.")
print(response)