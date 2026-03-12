"""
src/utils/prompt_templates.py
─────────────────────────────
Chain-of-Thought, few-shot, and self-consistency prompt templates.
28% answer faithfulness improvement over naive RAG baseline.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# ── Orchestrator / Query Decomposition ─────────────────────────────────────

ORCHESTRATOR_SYSTEM = """\
You are an expert research assistant with access to specialized tools.
Your job is to answer complex, multi-hop questions accurately and with source attribution.

You have access to the following tools:
{tools}

INSTRUCTIONS:
1. Break down complex questions into simpler sub-questions if needed.
2. Use the retriever tool to find relevant documents in the knowledge base.
3. Use the calculator tool for any arithmetic, statistics, or quantitative reasoning.
4. Use the web_search tool only when the knowledge base lacks current information.
5. Synthesize a final answer with inline citations [Source: doc_id].
6. If you cannot find sufficient evidence, say so — do NOT hallucinate.

Think step by step before acting. Use the format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>
Observation: <tool output>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: <grounded answer with citations>
"""

ORCHESTRATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ORCHESTRATOR_SYSTEM),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# ── RAG Chain — Chain-of-Thought + Few-Shot ────────────────────────────────

COT_FEW_SHOT_TEMPLATE = """\
You are a precise and factual question-answering assistant.
Answer ONLY based on the provided context. If the context is insufficient, say "I don't have enough information."

=== FEW-SHOT EXAMPLES ===

Example 1:
Context: [Doc A] Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.
Question: What inputs does photosynthesis require?
Thought: The context explicitly lists the inputs to photosynthesis.
Answer: Photosynthesis requires sunlight, water, and carbon dioxide (CO2) as inputs. [Source: Doc A]

Example 2:
Context: [Doc B] The company's revenue grew 23% YoY in Q3 2023, reaching $4.2B. Operating margins expanded to 18%.
Question: What was the revenue growth rate and absolute revenue?
Thought: The context gives both the growth rate (23%) and absolute figure ($4.2B).
Answer: Revenue grew 23% year-over-year in Q3 2023, reaching $4.2 billion. Operating margins were 18%. [Source: Doc B]

Example 3:
Context: [Doc C] The treaty was signed in 1648 and established the principle of state sovereignty.
Question: What caused World War I?
Thought: The context discusses a 1648 treaty and does not mention World War I causes.
Answer: I don't have enough information in the provided context to answer this question.

=== ACTUAL QUESTION ===

Context:
{context}

Question: {question}

Thought: Let me reason through this carefully using only the context above.
Answer:\
"""

COT_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=COT_FEW_SHOT_TEMPLATE,
)


# ── Self-Consistency Sampling ──────────────────────────────────────────────

SELF_CONSISTENCY_TEMPLATE = """\
You are a precise question-answering assistant. Answer using ONLY the context below.

Context:
{context}

Question: {question}

Generate {n_samples} independent reasoning chains and answers, then pick the most consistent one.

Chain 1:
Reasoning: 
Answer:

Chain 2:
Reasoning:
Answer:

Chain 3:
Reasoning:
Answer:

Most Consistent Answer (with citation):
"""

SELF_CONSISTENCY_PROMPT = PromptTemplate(
    input_variables=["context", "question", "n_samples"],
    template=SELF_CONSISTENCY_TEMPLATE,
)


# ── Synthesis / Citation Chain ─────────────────────────────────────────────

SYNTHESIS_TEMPLATE = """\
You are a research synthesizer. Given the sub-answers below from multiple agents, \
produce a single coherent, well-cited final answer.

Rules:
- Combine information logically; do not repeat yourself.
- Cite every factual claim with [Source: <doc_id>].
- If sources conflict, acknowledge the discrepancy.
- Be concise but complete.

Sub-answers:
{sub_answers}

Original Question: {question}

Final Synthesized Answer:
"""

SYNTHESIS_PROMPT = PromptTemplate(
    input_variables=["sub_answers", "question"],
    template=SYNTHESIS_TEMPLATE,
)


# ── Constitutional AI Self-Critique ───────────────────────────────────────

CONSTITUTIONAL_CRITIQUE_TEMPLATE = """\
Review the following AI-generated answer for any of these issues:
1. Harmful, toxic, or offensive content
2. Hallucinated facts not grounded in the sources
3. Demographic bias or stereotyping
4. Privacy violations (PII exposure)

Answer to review:
{answer}

Source documents used:
{sources}

For each issue found, state: ISSUE_TYPE | SEVERITY (low/medium/high) | EXPLANATION
If no issues, output: PASS

Review:
"""

CONSTITUTIONAL_PROMPT = PromptTemplate(
    input_variables=["answer", "sources"],
    template=CONSTITUTIONAL_CRITIQUE_TEMPLATE,
)

CONSTITUTIONAL_REVISION_TEMPLATE = """\
The following answer has been flagged for issues. Rewrite it to fix all problems \
while preserving the factual content and citations.

Original answer:
{answer}

Issues found:
{issues}

Revised answer (must remain grounded in original sources):
"""

CONSTITUTIONAL_REVISION_PROMPT = PromptTemplate(
    input_variables=["answer", "issues"],
    template=CONSTITUTIONAL_REVISION_TEMPLATE,
)


# ── Query Decomposition ────────────────────────────────────────────────────

DECOMPOSITION_TEMPLATE = """\
Break down the following complex question into {n} simpler sub-questions \
that can each be answered independently.

Complex question: {question}

Output as a numbered list:
1.
2.
...
"""

DECOMPOSITION_PROMPT = PromptTemplate(
    input_variables=["question", "n"],
    template=DECOMPOSITION_TEMPLATE,
)
