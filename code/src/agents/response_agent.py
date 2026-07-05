from src.orchestrator.context import AgentContext
from src.agents.response_generator import ResponseGenerator
from src.core.services.summarization_service import SummarizationService

class ResponseGenerationAgent:
    def __init__(self, call_llm_fn):
        self.generator = ResponseGenerator(call_llm_fn)
        self.summarizer = SummarizationService(call_llm_fn)

    def run(self, context: AgentContext) -> AgentContext:
        intent_norm = context.intent.lower().replace(" ", "_")
        
        # Hierarchical summarization redirect
        if intent_norm in ["summarization", "document_overview", "document overview"]:
            # Aggregate all available chunks to summarize the whole document text
            text = "\n".join([c[0].get("text", "") for c in context.retrieved_chunks])
            doc_name = context.retrieved_chunks[0][0].get("source", "document") if context.retrieved_chunks else "document"
            
            if not text:
                context.final_answer = "No document text available to summarize."
                return context
                
            res = self.summarizer.summarize_document(text, doc_name, "short")
            context.final_answer = res.get("summary", "")
            context.confidence_metrics = {
                "retrieval_confidence": 1.0,
                "evidence_coverage": 1.0,
                "answer_confidence": 1.0
            }
            context.add_log("ResponseGenerationAgent", "Hierarchical Map-Reduce summary completed successfully.")
            return context

        # Generate the response using context chunks
        answer, scores = self.generator.generate(
            context.rewritten_query,
            context.retrieved_chunks,
            context.intent
        )
        
        context.final_answer = answer
        context.confidence_metrics = scores
        
        context.add_log(
            "ResponseGenerationAgent",
            "Synthesized response text completed.",
            {"confidence_metrics": scores}
        )
        return context
