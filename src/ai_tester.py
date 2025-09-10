from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import save_tool
load_dotenv()

class CoachResponse(BaseModel):
    positions: str
    summary: str
    elixir_remaining: int

llm = ChatOpenAI(model = "gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object = CoachResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
                    You are a Clash Royale strategy coach modeled after Ryley, a top-tier Log Bait player known for:

            - Predictive and calculative gameplay
            - Anticipating opponent moves
            - Precision and strategic foresight
            - Maximizing value from spell baiting decks

            Your job is to analyze the current game state and give concise recommendations.

            You will be provided with:
            - Positions of enemy troops
            - Positions of friendly troops
            - Current elixir count for the player

            Your task:
            - Evaluate threats and opportunities
            - Suggest the optimal play (e.g., defend with a specific troop, counter-push, cycle, or wait)
            - Explain briefly *why* this is the best option
            - Output must follow the schema defined in `CoachResponse` 

            Keep your reasoning practical, Ryley-style: predictive, calculative, and precise. 
            {format_instructions}
        """

        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions = parser.get_format_instructions())

tools = [save_tool]

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = []
)

agent_executor = AgentExecutor(agent = agent, tools = [], verbose = True)
raw_response = agent_executor.invoke({"query": "What is a good starting play if my hand is hogrider, ice golem, log, ice spirit?"})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
except Exception as e:
    print("error parsing response", e, "raw Response - ", raw_response)
print(structured_response)