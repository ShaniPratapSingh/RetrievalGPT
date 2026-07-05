from src.orchestrator.context import AgentContext
from src.agents.intent_classifier import QueryIntentClassifier
from src.agents.query_rewriter import QueryRewriter

class QueryUnderstandingAgent:
    def __init__(self, call_llm_fn):
        self.classifier = QueryIntentClassifier(call_llm_fn)
        self.rewriter = QueryRewriter(call_llm_fn)

    def run(self, context: AgentContext) -> AgentContext:
        # Step 1: Detect intent
        hist_str = "\n".join([f"{r}: {c}" for r, c in context.history])
        res = self.classifier.classify(context.query, hist_str)
        context.intent = res["intent"]
        
        # Step 2: Rewrite query
        context.rewritten_query = self.rewriter.rewrite(context.query, hist_str)
        
        # Step 3: Simple language & complexity mapping
        context.language = "english"
        context.complexity = "high" if len(context.query.split()) > 10 else "medium"
        
        context.add_log(
            "QueryUnderstandingAgent",
            "Completed query intent classification and pronoun resolution.",
            {"intent": context.intent, "rewritten_query": context.rewritten_query}
        )
        return context
