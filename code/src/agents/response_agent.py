from src.orchestrator.context import AgentContext
from src.agents.response_generator import ResponseGenerator

class ResponseGenerationAgent:
    def __init__(self, call_llm_fn):
        self.generator = ResponseGenerator(call_llm_fn)

    def run(self, context: AgentContext) -> AgentContext:
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
