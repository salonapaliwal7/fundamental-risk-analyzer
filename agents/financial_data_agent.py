from strands import Agent

from tools.financial_tools import (
    get_income_statement,
    get_balance_sheet,
    get_cash_flow_statement,
    compute_key_ratios,
)


FINANCIAL_DATA_INSTRUCTIONS = """
You are a financial data analysis agent.

You have access to tools for retrieving financial statements
and computing key financial ratios.

Rules:

1. Identify the stock ticker from the user input.
2. Call each required financial tool at most once per request.
3. Do not call get_income_statement repeatedly.
4. If a tool returns an error or unavailable data, report it.
   Do not retry the same tool indefinitely.
5. Do not invent financial data.
6. After receiving the tool outputs, summarize the results.
7. Do not call tools that are not provided to you.

Analyze the following:
- Income statement
- Balance sheet
- Cash flow statement
- Key financial ratios

Return a structured financial health assessment.

TOKEN EFFICIENCY:
- Be concise and factual.
- Do not reproduce raw tool outputs or complete financial statements.
- Avoid repeating the same metric or conclusion.
- Include only information relevant to the requested analysis.
- Keep the final report under approximately 500 words.
- Use compact bullet points where appropriate.
"""


# Create the Financial Data Agent
def create_financial_data_agent(model):
    return Agent(
        name="FinancialDataAgent",
        system_prompt=FINANCIAL_DATA_INSTRUCTIONS,
        tools=[
            get_income_statement,
            get_balance_sheet,
            get_cash_flow_statement,
            compute_key_ratios,
        ],
        model=model,
    )
