from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool
from langchain.prompts import PromptTemplate
load_dotenv()

class CoachResponse(BaseModel):
    situation: str
    decision: str
    reasoning: str
    summary: str
    elixir_remaining: int

llm = ChatOpenAI(model = "gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object = CoachResponse)
prompt_template = """
            You are a Clash Royale strategy coach modeled after Ryley, a top-tier player known for:

            - Predictive and calculative gameplay
            - Anticipating opponent moves
            - Precision and strategic foresight
            - Maximizing value from spell baiting decks

            Your job is to analyze the current game state and give concise recommendations.

            You may be provided with:
            - Natural language descriptions of the situation
            - OR structured details like troop positions and elixir count

            If details are missing, make reasonable assumptions and still give the best recommendation. 

            Current Situation:
            {topic}

            Your task:
            - Evaluate threats and opportunities
            - Do not make up situations. Only use what has been provided in the "topics"
            - Only make decisions based on what information is currently/has been provided.
            - Suggest the optimal play (e.g., defend with a specific troop, counter-push, cycle, or wait)
            - Explain briefly *why* this is the best option
            - Output must follow the schema defined in `CoachResponse` 

            Keep your reasoning practical, Ryley-style: predictive, calculative, and precise. 
            {format_instructions}
"""
prompt = PromptTemplate(
    input_variables = ["topic", "format_instructions"],
    template = prompt_template,
)


def runModel(positions, time, elixir, hand):
    chain = prompt | llm
    result = chain.invoke({
        "topic": "I have hog rider, ice golem, log, and ice spirit in my starting hand. What should I do? My opponent just played a megaknight in the back.",
        "format_instructions": parser.get_format_instructions()
        }).content

    print(result)



'''
def runModel(user_input):
    agent_executor = AgentExecutor(agent = agent, tools = [], verbose = True)
    raw_response = agent_executor.invoke({"query": input("User: ")})
    try:
        structured_response = parser.parse(raw_response.get("output")[0]["text"])
    except Exception as e:
        print("error parsing response", e, "raw Response - ", raw_response)
'''