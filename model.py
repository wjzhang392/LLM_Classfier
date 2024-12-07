from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Initialize LLM
def initialize_llm(model_name="gpt-4o-mini", temperature=0, top_p=1.0):
    return ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p)

# Create LangChain for classification
def create_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)
