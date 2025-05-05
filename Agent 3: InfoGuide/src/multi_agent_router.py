from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.router.multi_prompt import MultiPromptChain
from langchain_groq import ChatGroq
from copilots.Agents import LLM
# Define LLM and memory
# llm = LLM()
llm = ChatGroq(
    groq_api_key='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',
    model_name="llama3-70b-8192"
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define prompt templates for each agent
qa_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a helpful and factual assistant answering user questions clearly and accurately.
Question: {input}
Answer:"""
)

doc_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a documentation specialist. Answer only using the product or system documentation.
Documentation Query: {input}
Documentation Answer:"""
)

causal_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a friendly assistant which answers user questions about causal discovery algorithms.
User says: {input}
Casual Reply:"""
)

forecast_prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a forecasting expert. Use trends, patterns, or previous data to predict possible outcomes.
Forecasting question: {input}
Forecast Response:"""
)

# Map destination names to chains
qa_chain = LLMChain(llm=llm, prompt=qa_prompt, memory=memory)
doc_chain = LLMChain(llm=llm, prompt=doc_prompt, memory=memory)
causal_chain = LLMChain(llm=llm, prompt=causal_prompt, memory=memory)
forecast_chain = LLMChain(llm=llm, prompt=forecast_prompt, memory=memory)

# Router prompt
router_template = """
Given a user question, classify it into one of the following categories:
- qa: factual Q&A
- doc: documentation lookup
- causal: causal discovery
- forecast: forecasting and predictions

User input: {input}
Which category best fits?
Answer with one of: qa, doc, casual, forecast
"""

router_prompt = PromptTemplate(input_variables=["input"], template=router_template)

prompt_infos = [
    {
        "name": "qa",
        "description": "factual Q&A",
        "prompt_template": qa_prompt.template.strip()
    },
        {
        "name": "doc",
        "description": "documentation lookup",
        "prompt_template": doc_prompt.template.strip()
    },
    {
        "name": "causal",
        "description": "causal discovery",
        "prompt_template": causal_prompt.template.strip()
    },
    {
        "name": "forecast",
        "description": "forecasting and predictions",
        "prompt_template": forecast_prompt.template.strip()
    },
]
# Define the router chain
router_chain = MultiPromptChain.from_prompts(
    default_chain=qa_chain,
    # router_prompt=router_template,
    llm=llm,
    prompt_infos = prompt_infos,
    verbose =True
)

def run_conversation():
    print("\n--- AI Assistant (Multi-Agent) ---")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = router_chain.run(user_input)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    run_conversation()
