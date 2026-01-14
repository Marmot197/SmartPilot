# 🤖 SmartPilot Agentic-KG Challenge

**SmartPilot** is a custom, compact, and neurosymbolic AI model for intelligent manufacturing. This competition focuses on the **InfoGuide** agent—a question-answering chatbot that leverages knowledge graphs and manufacturing documentation to answer domain-specific queries.

**Demo**: [Watch the demo](https://www.youtube.com/watch?v=fh11PULNrTM)

---

# 🏆 Competition Participation Guide

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Step 1: Register](#step-1-register)
3. [Step 2: Download the Starter Kit](#step-2-download-the-starter-kit)
4. [Step 3: Develop Your Agent](#step-3-develop-your-agent)
5. [Step 4: Local Testing](#step-4-local-testing)
6. [Step 5: Submit](#step-5-submit)
7. [Step 6: Iterate](#step-6-iterate)
8. [How to Cite](#how-to-cite)

---

## 1. Prerequisites

Before starting, ensure you have the following skills and tools:

### Required Skills

- **Python** (3.8+)

- **Basic Graph Concepts**:
  - Nodes, edges, and relations
  - Graph traversal and querying

- **Prompt Engineering**:
  - Large Language Models (LLMs) and their APIs
  - Designing effective prompts
  - Understanding model outputs and limitations

- **Reading Technical Documentation**

### Required Tools

- **Git**
- **Python 3.8+**: Check with `python --version`
- **pip**
- **Text Editor/IDE**
- **API Access**

---

## Step 1: Register

### Registration Process

1. **Sign up on the challenge platform** : [Codabench - Competition Link](https://www.codabench.org/competitions/12608/)

2. **Team or Individual Registration**

3. **Access Verification**
   - Log in to the competition portal
   - Verify you can access:
     - Competition dashboard
     - Submission interface
     - Leaderboard (if public)
     - Documentation and resources

---

## Step 2: Download the Starter Kit

1. **SmartPilot Application**
   - Full SmartPilot codebase with hook functions
   - Baseline implementation of `run_participant_doc_agent`
   - Infrastructure for running and testing your agent

2. **Baseline Agent Implementation**
   - File: `Agent 3: InfoGuide/src/run_updated.py`
   - Function: `run_participant_doc_agent()`
   - This is your starting point—modify this function to improve performance

3. **Sample Questions**
   - Development set for local testing available at `Challenge_utils/SmartPilot Agentic-KG Challenge.xlsx`
   - Examples of expected question formats
   - Use these to validate your approach

4. **Knowledge Graph**
   - File: `Analog24HrRunKG_Demo.ttl`
   - Format: RDF Turtle (.ttl)
   - Contains structured manufacturing knowledge
   - Query this graph to enhance your answers

5. **Documentation**
   - Sample manual file: `Manual_sample.pdf`


### Download Instructions

**Option 1: Git Clone (Recommended)**

```bash
# Clone the repository
git clone https://github.com/ChathurangiShyalika/SmartPilot.git
cd SmartPilot

# Switch to the competition branch (IMPORTANT!)
git checkout KG_Challenge

# Verify you're on the correct branch
git branch

# Install dependencies
pip install -r requirements.txt
```

**Option 2: Download ZIP**

1. Navigate to: https://github.com/ChathurangiShyalika/SmartPilot
2. Switch to the `KG_Challenge` branch using the branch dropdown
3. Click "Code" → "Download ZIP"
4. Extract the ZIP file
5. Install dependencies: `pip install -r requirements.txt`

### Verify Installation

Run these commands to verify everything is set up correctly:

```bash
# Check Python version (should be 3.8+)
python --version

# Verify dependencies installed
pip list | grep -E "streamlit|rdflib|sparql"

# Test import (if applicable)
python -c "import streamlit; print('Streamlit installed')"
```

### File Structure Overview

```
SmartPilot/
├── Agent 3: InfoGuide/
│   └── src/
│       ├── run_updated.py          # Main file with run_participant_doc_agent
│       └── ...
├── Challenge_utils/
|   └──Analog24HrRunKG_Demo.ttl        # Knowledge Graph file
|   └──sample_questions.txt            # Test questions                 
|   └──Manual_sample.pdf            # Sample manufacturing manual
|   requirements.txt        # Python dependencies
└── README.md                       # Documentation
```

---

## Step 3: Develop Your Agent

Your agent must answer manufacturing operator questions like:
- "What sensors are connected to Robot 3?"
- "What is the safe range for Gripper_Load during cycle startup?"
- "Which components are most likely involved if temperature spikes at Robot 4?"
- "How should I calibrate the Gripper_Load sensor?"

Answers must be grounded in the Knowledge Graph (KG), process ontology, and documentation.

### Function Signature

Modify `run_participant_doc_agent()` in `run_updated.py`:

```python
def run_participant_doc_agent(user_input, rdf_graph, process_qa, session_state) -> str:
```

**Inputs:**
- `user_input`: Natural language question from operator
- `rdf_graph`: RDF Knowledge Graph (rdflib graph object)
- `process_qa`: Helper object with process ontology JSON access
- `session_state`: Dict with conversation history, context, cached info

**Available Functions:**
- KG helpers: `get_full_feature_semantic_info()`, `get_full_entity_semantic_info()`
- ProcessOntologyQA methods
- Documentation retrieval utilities
- `LLM()` wrapper for your prompting logic

### Required Output Format

Return a JSON string with these fields:

```python
{
    "answer": "Short operator-facing answer in natural language",
    "explanation": "Concise explanation grounded in process/sensors/states",
    "kg_entities": ["Robot_3", "Gripper_Load_Sensor_1", "Safety_Door_Sensor_2"],
    "kg_relations": ["robotSensingPart", "rdf:type"]
}
```

- `answer`: Displayed to operator in UI
- `explanation`: Shows reasoning (optional display)
- `kg_entities`: List of KG node IDs/URIs used (required for evaluation)
- `kg_relations`: List of KG predicates/relations used (required for evaluation)

### Available Resources

**Codebase:**
- KG access utilities (rdflib graph)
- Documentation loading/retrieval (AssetLoader, Knowledge_Representation, Retr)
- Memory management helpers
- Causal graph info (optional)

**Knowledge Resources:**
- RDF Knowledge Graph file (manufacturing assets, sensors, robots, anomalies)
- Process ontology JSON (`Agent 3: InfoGuide/src/assets/d3_graph.json`)
- Sensor range metadata (tolerances, units, states)
- Domain documentation (setup guides, safety procedures, calibration, troubleshooting)

### Baseline System

The baseline implementation:
- Classifies query topic
- Retrieves documentation chunks
- Appends KG/ontology descriptions as plain text
- Calls LLM once to generate answer
- Heuristically guesses `kg_entities` from text

**Baseline weaknesses:**
- Weak grounding metrics (no structural KG reasoning)
- Hallucinates when information is missing
- Over-generalizes on complex queries

### Improvement Strategies

**Focus Areas:**
- **Structural KG Querying**: Use SPARQL or graph traversal to extract relevant entities/relations
- **Multi-Step Reasoning**: Decompose complex questions, query KG for each part, synthesize answers
- **Explicit Grounding**: Track which KG entities/relations you use (don't guess)
- **Better Prompting**: Use KG structure in prompts, implement chain-of-thought reasoning
- **Documentation Integration**: Combine KG facts with retrieved documentation chunks

**What You Can Do:**
- Call `LLM()` multiple times for multi-step reasoning
- Query RDF graph structurally (SPARQL, graph traversal)
- Use or replace existing RAG pipeline
- Create helper functions for entity extraction, KG querying, answer synthesis
- Add preprocessing/postprocessing steps

---

## Step 4: Local Testing

### Testing Your Agent

**Run the SmartPilot UI:**
```bash
cd "Agent 3: InfoGuide/src"
streamlit run run_updated.py
```

**Test on sample questions:**
- Use provided development set questions
- Verify JSON output format is correct
- Check that `kg_entities` and `kg_relations` are populated
- Ensure answers are grounded in KG/documentation

**Run evaluation script (if provided):**
```bash
python evaluate.py --questions sample_questions.txt --output results.json
```

### Evaluation Metrics

Your agent will be scored on:

1. **Answer Correctness (40%)**
   - Compare your `answer` field to reference answers
   - Must be correct, relevant, consistent with KG/docs

2. **KG Grounding Quality (30%)**
   - `kg_entities` and `kg_relations` compared to gold-standard subgraphs
   - Measured by recall (coverage) and precision (accuracy)
   - Penalties for invented entities/relations not in KG

3. **Explanation Quality (20%)**
   - Evaluated for clarity and usefulness
   - Must align with claimed KG entities/relations
   - No contradictions

4. **Robustness & Abstention (10%)**
   - Rewards clear statements when answer cannot be produced
   - Penalizes hallucinated sensors/states/relationships

### Pre-Submission Checklist

- [ ] Agent runs without errors
- [ ] JSON output format is correct (answer, explanation, kg_entities, kg_relations)
- [ ] `kg_entities` and `kg_relations` contain actual KG elements (not invented)
- [ ] Answers are grounded in KG/documentation
- [ ] Handles ambiguous queries gracefully (abstention when needed)

---

## Step 5: Submit

**Package your code:**
- Create a ZIP file containing `run_participant_doc_agent.py` and `requirements.txt`
- Include any helper modules your code imports

**Submit via competition portal:**
- Upload your ZIP file
- Platform runs your code in a controlled environment on test questions
- Scores are computed automatically and displayed on the leaderboard


## Step 6: Iterate

Use feedback from the leaderboard to continuously improve your agent.

## How to Cite

If you use SmartPilot or any of its components in your research, publications, or systems, please cite:

```
@inproceedings{shyalika2025smartpilot,
  title={SmartPilot: Agent-Based CoPilot for Intelligent Manufacturing},
  author={Shyalika, Chathurangi and Prasad, Renjith and Al Ghazo, Alaa and Eswaramoorthi, Darssan L and Shree Muthuselvam, Sara and Sheth, Amit},
  booktitle={Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems},
  pages={3053--3055},
  year={2025}
}

@article{shyalika2025smartpilot,
  title={SmartPilot: A Multiagent CoPilot for Adaptive and Intelligent Manufacturing},
  author={Shyalika, Chathurangi and Prasad, Renjith and Ghazo, Alaa Al and Eswaramoorthi, Darssan and Kaur, Harleen and Muthuselvam, Sara Shree and Sheth, Amit},
  journal={arXiv preprint arXiv:2505.06492},
  year={2025}
}
```

---

Good luck with the competition! 🚀
