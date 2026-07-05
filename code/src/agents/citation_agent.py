from src.orchestrator.context import AgentContext
from src.core.citation import CitationEngine

class CitationAgent:
    def __init__(self):
        pass

    def run(self, context: AgentContext) -> AgentContext:
        chunks_only = [item[0] for item in context.retrieved_chunks]
        
        # Verify and link citations using clean CitationEngine implementation
        clean_answer, citations = CitationEngine.extract_citations(context.final_answer, chunks_only)
        
        context.final_answer = clean_answer
        context.citations = citations
        
        context.add_log(
            "CitationAgent",
            f"Citation linking complete. Mapped {len(citations)} citations.",
            {"citations_count": len(citations)}
        )
        return context
