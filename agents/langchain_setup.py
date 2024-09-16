# langchain_setup.py
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import ChatOllama
from langchain.tools import tool
import logging
import pandas as pd
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Ollama LLM
llm = ChatOllama(model="phi3.5")

@tool
def tool_get_all_rates(input: str) -> str:
    """
    Retrieves exchange rates for a given date and optional currency code.
    Input format: 'date' or 'date, currency_code'.
    """
    from analysis_functions import get_all_rates
    parts = input.split(',')
    date_str = parts[0].strip()
    currency_code = parts[1].strip() if len(parts) > 1 else None
    result = get_all_rates(date_str, currency_code)
    if isinstance(result, pd.DataFrame):
        result = result.to_dict(orient='records')
    print(f'{parts}')
    return str(result)

@tool
def tool_plot_currency_trends(input: str) -> str:
    """
    Plots the exchange rate trends for a specific currency between two dates.
    Input format: 'currency_code, start_date, end_date'
    """
    from analysis_functions import plot_currency_trends
    parts = input.split(',')
    if len(parts) != 3:
        return "Invalid input format. Please provide 'currency_code, start_date, end_date'."
    currency_code = parts[0].strip()
    start_date_str = parts[1].strip()
    end_date_str = parts[2].strip()
    result = plot_currency_trends(currency_code, start_date_str, end_date_str)
    if result.startswith("No exchange rate data found") or result.startswith("Invalid date format"):
        return result
    else:
        # Return the image with the appropriate prefix
        return f"data:image/png;base64,{result}"

tools = [
    Tool(
        name="Get All Rates",
        func=tool_get_all_rates,
        description="Retrieves exchange rates for a given date and optional currency code. Input format: 'date' or 'date, currency_code'."
    ),
    Tool(
        name="Plot Currency Trends",
        func=tool_plot_currency_trends,
        description="Plots the exchange rate trends for a specific currency between two dates. Input format: 'currency_code, start_date, end_date'. Returns a base64-encoded image."
    )
]

agent_kwargs = {
"format_instructions": """Use the following format:

Question: the input question you must answer

Thought: think about what action to take

Action: the action to take, should be one of [Get All Rates, Plot Currency Trends]

Action Input: the input to the action (if needed)

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

**Important Guidelines**:
- **The 'Action' line should ONLY contain the action name, and nothing else.**
- **The 'Action Input' should ONLY contain the input, and nothing else.**
- **Do not include additional explanations or text in these lines.**
- **If the Observation indicates that no data is available, politely inform the user in the Final Answer and do not attempt further actions.**
- **If the Observation returns an image (Base64), directly return it in the final answer without further analysis.**
- **The final answer should use MYR**
- **If the Observation returns an image (Base64), directly return it in the final answer without further analysis.**


**Examples**:

1. **Data Available**:

Question: Show me the trends for USD from 2022-01-01 to 2022-12-31

Thought: I should use the Plot Currency Trends tool.

Action: Plot Currency Trends

Action Input: 'USD, 2022-01-01, 2022-12-31'

Observation: data:image/png;base64,<base64-encoded image>

Thought: I now know the final answer.

Final Answer: Here is the exchange rate trend for USD between 2022-01-01 and 2022-12-31:
data:image/png;base64,<base64-encoded image>

2. **No Data Available**:

Question: Show me the trends for USD from 2022-06-01 to 2022-12-31

Thought: I should use the Plot Currency Trends tool.

Action: Plot Currency Trends

Action Input: 'USD, 2022-06-01, 2022-12-31'

Observation: No exchange rate data found for USD between 2022-06-01 and 2022-12-31.

Thought: The data is not available. I should inform the user.

Final Answer: I'm sorry, but there is no exchange rate data available for USD between 2022-06-01 and 2022-12-31.
"""
}

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs=agent_kwargs
)

# def run_agent(query: str) -> str:
#     try:
#         response = agent.run(query)
#         logger.info(f"Agent response: {response}")
#         return response
#     except Exception as e:
#         logger.error(f"Error running agent: {e}")
#         return f"Sorry, I encountered an error while processing your request: {str(e)}"

def run_agent(query: str) -> str:
    try:
        response = agent.run(query)
        logger.info(f"Agent response: {response}")
        if response.startswith("data:image/png;base64,"):
            return response  # Return the image data directly
        return response
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"