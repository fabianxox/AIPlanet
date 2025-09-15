import os
import time
import uuid
from typing import TypedDict, List, Dict, Optional
import warnings

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.graph import StateGraph, END
import dspy

# load environment dspy
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
dspy.configure(lm=dspy.LM("groq/deepseek-r1-distill-llama-70b", api_key=groq_api_key))

tavily_search_wrapper = TavilySearchAPIWrapper()


DB_FAISS_PATH = "vectorstore/db_faiss"
vector_store = None
SIMILARITY_THRESHOLD = 0.53

if os.path.exists(DB_FAISS_PATH):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vector_store = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"[Vector Store] Error loading: {e}")

# dspy signatures
class ClassificationSig(dspy.Signature):
    """Classify if a query is math related."""
    query: str = dspy.InputField()
    is_math: bool = dspy.OutputField()

class ExtractEquationSig(dspy.Signature):
    """Extract equation from a query."""
    query: str = dspy.InputField()
    equation: str = dspy.OutputField()

class ExplanationSig(dspy.Signature):
    """Explain a math question using retrieved context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField()
    solution: str = dspy.OutputField()

class RefinementSig(dspy.Signature):
    """Refine an existing solution using feedback."""
    question: str = dspy.InputField()
    previous_solution: str = dspy.InputField()
    feedback: str = dspy.InputField()
    refined_solution: str = dspy.OutputField()

# dSPy predictors
classifier = dspy.Predict(ClassificationSig)
extractor = dspy.Predict(ExtractEquationSig)
explainer = dspy.Predict(ExplanationSig)
refiner = dspy.Predict(RefinementSig)

# langGraph begins :)
class AgentState(TypedDict):
    messages: List[HumanMessage]
    is_math_query: bool
    user_query: str
    search_context: str
    response: str
    feedback: str
    original_response: str

# agents
def api_gateway_agent(state: AgentState):
    query = state["messages"][-1].content
    c = classifier(query=query)
    return {"is_math_query": c.is_math, "user_query": query}

def vector_db_search_agent(state: AgentState):
    eq_result = extractor(query=state["user_query"])
    equation = eq_result.equation
    print(f"[Vector Agent] Extracted equation: {equation}")

    if vector_store is None:
        print("[Vector Agent] Vector store not loaded. Skipping search.")
        return {"search_context": None}

    search_query = f"A solution to the equation: {equation}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        docs_and_scores = vector_store.similarity_search_with_relevance_scores(search_query, k=1)

    if docs_and_scores:
        doc, score = docs_and_scores[0]
        if score >= SIMILARITY_THRESHOLD:
            print(f"[Vector Agent] Found relevant document with score {score:.4f}")
            return {"search_context": doc.page_content}
        else:
            print(f"[Vector Agent] Top document relevance {score:.4f} below threshold")
            return {"search_context": None}
    else:
        print("[Vector Agent] No documents found in vector store.")
        return {"search_context": None}

def tavily_search_agent(state: AgentState):
    user_query = state["user_query"]
    print(f"[Web Search Agent] Searching web for: '{user_query}'")
    search_results = tavily_search_wrapper.results(query=user_query, max_results=3)
    if search_results:
        first_result = search_results[0]
        formatted_result = f"Content: {first_result['content']}\nSource: {first_result['url']}"
        print(f"[Web Search Agent] Found results: {first_result['url']}")
        return {"search_context": formatted_result}
    else:
        print("[Web Search Agent] No results found on the web.")
        return {"search_context": None}

def math_explanation_agent(state: AgentState):
    if not state.get("search_context"):
        return {"response": "I do not have a correct answer for your query.",
                "original_response": "I do not have a correct answer for your query."}
    e = explainer(question=state["user_query"], context=state["search_context"])
    return {"response": e.solution, "original_response": e.solution}

def feedback_agent(state: AgentState):
    # kept for graph compatibility (not used by FastAPI endpoints)
    print("\n🤖 MathBot's answer:\n-----------------\n", state["response"])
    feedback = input("\n Was this helpful? (yes or correction): ")
    return {"feedback": feedback}

def refinement_agent(state: AgentState):
    r = refiner(
        question=state["user_query"],
        previous_solution=state["response"],
        feedback=state["feedback"]
    )
    return {"response": r.refined_solution, "original_response": r.refined_solution}

def rejection_node(state: AgentState):
    return {"response": " I am a specialized math assistant. Please ask a math question."}

# conditional routing
def route_after_gateway(state: AgentState):
    return "vector_db_search_agent" if state["is_math_query"] else "rejection_node"

def route_after_db_search(state: AgentState):
    return "math_explanation_agent" if state["search_context"] else "tavily_search_agent"

def route_after_feedback(state: AgentState):
    return "END" if state["feedback"].lower().strip() == "yes" else "refinement_agent"


workflow = StateGraph(AgentState)

workflow.add_node("api_gateway_agent", api_gateway_agent)
workflow.add_node("vector_db_search_agent", vector_db_search_agent)
workflow.add_node("tavily_search_agent", tavily_search_agent)
workflow.add_node("math_explanation_agent", math_explanation_agent)
workflow.add_node("feedback_agent", feedback_agent)
workflow.add_node("refinement_agent", refinement_agent)
workflow.add_node("rejection_node", rejection_node)

workflow.set_entry_point("api_gateway_agent")
workflow.add_conditional_edges("api_gateway_agent", route_after_gateway, {
    "vector_db_search_agent": "vector_db_search_agent",
    "rejection_node": "rejection_node",
})
workflow.add_conditional_edges("vector_db_search_agent", route_after_db_search, {
    "math_explanation_agent": "math_explanation_agent",
    "tavily_search_agent": "tavily_search_agent",
})
workflow.add_edge("tavily_search_agent", "math_explanation_agent")
workflow.add_edge("math_explanation_agent", "feedback_agent")
workflow.add_conditional_edges("feedback_agent", route_after_feedback, {
    "refinement_agent": "refinement_agent",
    "END": END,
})
workflow.add_edge("refinement_agent", "feedback_agent")
workflow.add_edge("rejection_node", END)
app_graph = workflow.compile()

# fastAPI app with HITL 
app = FastAPI(title="MathBot with HITL (browser)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local session store (single-user/demo). Each session: { created_at, messages, user_query, search_context, intermediate }
sessions: Dict[str, Dict] = {}

# session TTL cleanup 
SESSION_TTL = 60 * 60 

def cleanup_sessions(ttl: int = SESSION_TTL):
    now = time.time()
    to_delete = [sid for sid, data in sessions.items() if now - data.get("created_at", 0) > ttl]
    for sid in to_delete:
        sessions.pop(sid, None)

class AskRequest(BaseModel):
    session_id: Optional[str] = None
    query: str

class RefineRequest(BaseModel):
    session_id: str
    feedback: str

class FinalRequest(BaseModel):
    session_id: str

# helper: do retrieval + explanation
def retrieve_and_explain(user_query: str):
    """
    Runs the retrieval (vector -> tavily) and explanation agents,
    returns (search_context, intermediate_answer)
    """
    # vector DB attempt
    v_state = {"user_query": user_query}
    v_res = vector_db_search_agent(v_state)
    search_context = v_res.get("search_context")

    # fallback to web
    if not search_context:
        t_state = {"user_query": user_query}
        t_res = tavily_search_agent(t_state)
        search_context = t_res.get("search_context")

    if not search_context:
        return None, "I do not have a correct answer for your query."

    exp_state = {"user_query": user_query, "search_context": search_context}
    expl_res = math_explanation_agent(exp_state)
    intermediate = expl_res.get("response", "").strip()
    if not intermediate:
        return search_context, "I do not have a correct answer for your query."

    return search_context, intermediate

# endpoint
@app.post("/ask")
def ask_question(request: AskRequest):
    cleanup_sessions()
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query provided.")

    session_id = request.session_id or str(uuid.uuid4())
    
    session = sessions.setdefault(session_id, {
        "created_at": time.time(),
        "messages": [],
        "user_query": None,
        "search_context": None,
        "intermediate": None
    })

    
    session["messages"].append(HumanMessage(content=request.query))
    session["user_query"] = request.query

    # input AI gateway 
    gateway_state = {"messages": session["messages"]}
    gateway_res = api_gateway_agent(gateway_state)
    if not gateway_res.get("is_math_query"):
        return {"session_id": session_id, "intermediate_answer": "I am a specialized math assistant. Please ask a math question."}
    
    search_context, intermediate_answer = retrieve_and_explain(request.query)
    session["search_context"] = search_context

    
    try:
        c = classifier(query=intermediate_answer)
        is_math_flag = bool(c.is_math)
    except Exception as e:
        print(f"[Classifier error] {e}")
        is_math_flag = False

    if not is_math_flag:
        refine_state = {
            "user_query": session["user_query"],
            "response": intermediate_answer,
            "feedback": "Make this answer math-related and focus on steps and final result."
        }
        refined = refinement_agent(refine_state)
        intermediate_answer = refined.get("response", "").strip()

        # Re-check
        try:
            c2 = classifier(query=intermediate_answer)
            if not bool(c2.is_math):
                intermediate_answer = "I do not have a correct answer for your query."
        except Exception:
            intermediate_answer = "I do not have a correct answer for your query."

    
    session["intermediate"] = intermediate_answer

    return {"session_id": session_id, "intermediate_answer": intermediate_answer}

# endpoint: /refine 
@app.post("/refine")
def refine_answer(request: RefineRequest):
    cleanup_sessions()
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Session not found")

    if not request.feedback or not request.feedback.strip():
        raise HTTPException(status_code=400, detail="Empty feedback provided.")

    # use the last intermediate response as previous_solution
    previous = session.get("intermediate", "")
    state = {
        "user_query": session.get("user_query", ""),
        "response": previous,
        "feedback": request.feedback
    }

    refined_state = refinement_agent(state)
    refined_answer = refined_state.get("response", "").strip()

    # Guardrail check
    try:
        c = classifier(query=refined_answer)
        if not bool(c.is_math):
            refined_answer = "I do not have a correct answer for your query."
    except Exception:
        refined_answer = "I do not have a correct answer for your query."

    session["intermediate"] = refined_answer
    return {"session_id": request.session_id, "refined_intermediate_answer": refined_answer}

# endpoint: /final
@app.post("/final")
def final_answer(request: FinalRequest):
    cleanup_sessions()
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Session not found")

    # Prefer refined/intermediate answer if present
    final_response = session.get("intermediate")
    user_query = session.get("user_query", "")

    if not final_response:
        _, final_response = retrieve_and_explain(user_query)

    # final guardrail
    try:
        c = classifier(query=final_response)
        is_math_flag = bool(c.is_math)
    except Exception as e:
        print(f"[Classifier error] {e}")
        is_math_flag = False

    if not is_math_flag:
        refine_state = {
            "user_query": user_query,
            "response": final_response,
            "feedback": "Ensure this is a valid math solution (step-by-step and final result)."
        }
        refined = refinement_agent(refine_state)
        final_response = refined.get("response", "").strip()
        # re-check
        try:
            c2 = classifier(query=final_response)
            if not bool(c2.is_math):
                final_response = "I do not have a correct answer for your query."
        except Exception:
            final_response = "I do not have a correct answer for your query."

    # fallback if empty
    if not final_response or final_response.strip() == "":
        final_response = "I do not have a correct answer for your query."

    # clearing session after delivering final answer
    sessions.pop(request.session_id, None)

    return {"final_answer": final_response}
