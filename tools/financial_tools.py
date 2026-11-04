import math

import yfinance as yf
from strands import Agent, tool
from strands_tools.tavily import tavily_search
import json



class _Caching:
    # to avoid uplicate tool calls
    # key = tool_name
    # value = ticker
    def __init__(self):
        self.stored_data = {}
    def get(self, key):
        return self.stored_data.get(key)
    def store(self,key, value):
        self.stored_data[key] = value

    
_cache = _Caching()


def _float(n):
    # convert to float if possible, else return None
    try:
        f=float(n)
        return None if math.isnan(f) else f
    except(TypeError, ValueError):
        return None


def _million(n):
    # convert to million if possible, else return None
    try:
        f=float(n)
        return None if math.isnan(f) else round(f/1_000_000, 2)
    except(TypeError, ValueError):
        return None


@tool
def get_income_statement(ticker: str) -> str:
    """
    Retrieve the income statement for a publicly traded company.

    Fetches reported financial data for the specified stock ticker,
    including revenue, cost of revenue, gross profit, EBIDTA, operating income,
    net income, earnings per share (EPS), and other available metrics.
    All values are returned in millions USD.

    Args:
        ticker (str): The ticker symbol of the company.

    Returns:
        A dictionary containing the company's income statement data,
        including financial metrics and reporting periods. Returns an
        error message if the data cannot be retrieved.
    """

    _key = ("get_income_statement", ticker.upper())
    _hit = _cache.get(_key)
    if _hit is not None:
        return _hit
    try:
        stock = yf.Ticker(ticker.upper())
        income_statement = stock.financials
        if income_statement.empty or income_statement is None:
            return {"error": f"No income statement data found for ticker '{ticker}'."}

        curr = stock.info.get("currency", "USD")

        # restricting the format that will be returned
        result: dict = {
            "ticker"    : ticker.upper(),
            "company_name": stock.info.get("longName", ticker.upper()),
            "currency": curr,
            "unit": f"millions {curr}",
            "annual_data": {},
        }
        for col in income_statement.columns[:4]:          # newest 4 years of data
            result["annual_data"][col.year] = {
                "totalRevenue": _million(income_statement.at["Total Revenue", col]),
                "costOfRevenue": _million(income_statement.at["Cost Of Revenue", col]),
                "grossProfit": _million(income_statement.at["Gross Profit", col]),
                "ebitda": _million(income_statement.at.get("EBITDA", {}).get(col)),
                "operatingIncome": _million(income_statement.at["Operating Income", col]),
                "netIncome": _million(income_statement.at["Net Income", col]),
                "eps": _float(income_statement.at.get("Earnings Per Share", {}).get(col)),
                "researchAndDevelopment": _million(income_statement.at.get("Research Development", {}).get(col)),
            }

        info = stock.info
        result["eps"] = _float(info.get("trailingEps"))
        result["forwardEPS"] = _float(info.get("forwardEps"))
        result["totalRevenue"] = _million(info.get("totalRevenue"))
        result["grossMargins"] = _float(info.get("grossMargins"))
        result["operatingMargins"] = _float(info.get("operatingMargins"))
        result["profitMargins"] = _float(info.get("profitMargins"))


        output = json.dumps(result, indent=2, default=str)
        _cache.store(_key, output)
        return output
    except Exception as e:
        return {"error": str(e), "ticker": ticker.upper()}




@tool
def get_balance_sheet(ticker: str) -> str:
    """
    Retrieve the annual balance sheet for a publicly traded company.

    Fetches reported financial data for the specified stock ticker,
    including total assets, current assets, cash and cash equivalents,
    accounts receivable, inventory, property and equipment, total
    liabilities, current liabilities, short-term debt, long-term debt,
    total debt, shareholders' equity, retained earnings, and working
    capital.

    Returns the latest four available annual reporting periods.
    All monetary values are returned in millions of the reporting
    currency. Financial ratios retain their original units.

    Args:
        ticker (str): The ticker symbol of the company, such as "AAPL".

    Returns:
        A JSON string containing company information, annual balance
        sheet data, and selected financial ratios. Returns an error
        dictionary if the data cannot be retrieved.
    """

    ticker = ticker.upper().strip()
    _key = ("get_balance_sheet", ticker)

    _hit = _cache.get(_key)
    if _hit is not None:
        return _hit

    try:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet

        if balance_sheet is None or balance_sheet.empty:
            return {
                "error": f"No balance sheet data found for ticker '{ticker}'."
            }

        info = stock.info or {}
        curr = info.get("currency", "USD")

        # Safely retrieve metrics that may be missing for some companies.
        def get_value(row_names, col):
            for row_name in row_names:
                if row_name in balance_sheet.index:
                    return balance_sheet.at[row_name, col]
            return None

        result: dict = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "currency": curr,
            "unit": f"millions {curr}",
            "annual_data": {},
        }

        # Latest four annual reporting periods.
        for col in balance_sheet.columns[:4]:
            result["annual_data"][col.year] = {
                # Assets
                "totalAssets": _million(
                    get_value(["Total Assets"], col)
                ),
                "currentAssets": _million(
                    get_value(["Current Assets"], col)
                ),
                "cashAndCashEquivalents": _million(
                    get_value([
                        "Cash And Cash Equivalents",
                        "Cash Cash Equivalents And Short Term Investments",
                    ], col)
                ),
                "shortTermInvestments": _million(
                    get_value(["Other Short Term Investments"], col)
                ),
                "accountsReceivable": _million(
                    get_value(["Accounts Receivable", "Receivables"], col)
                ),
                "inventory": _million(
                    get_value(["Inventory"], col)
                ),
                "otherCurrentAssets": _million(
                    get_value(["Other Current Assets"], col)
                ),
                "netPPE": _million(
                    get_value(["Net PPE"], col)
                ),
                "goodwill": _million(
                    get_value(["Goodwill"], col)
                ),
                "intangibleAssets": _million(
                    get_value([
                        "Other Intangible Assets",
                        "Intangible Assets",
                    ], col)
                ),
                "totalNonCurrentAssets": _million(
                    get_value(["Total Non Current Assets"], col)
                ),

                # Liabilities
                "totalLiabilities": _million(
                    get_value([
                        "Total Liabilities Net Minority Interest",
                        "Total Liabilities",
                    ], col)
                ),
                "currentLiabilities": _million(
                    get_value(["Current Liabilities"], col)
                ),
                "accountsPayable": _million(
                    get_value(["Accounts Payable"], col)
                ),
                "currentDebt": _million(
                    get_value([
                        "Current Debt",
                        "Current Debt And Capital Lease Obligation",
                    ], col)
                ),
                "longTermDebt": _million(
                    get_value([
                        "Long Term Debt",
                        "Long Term Debt And Capital Lease Obligation",
                    ], col)
                ),
                "totalDebt": _million(
                    get_value(["Total Debt"], col)
                ),
                "otherCurrentLiabilities": _million(
                    get_value(["Other Current Liabilities"], col)
                ),
                "totalNonCurrentLiabilities": _million(
                    get_value(["Total Non Current Liabilities"], col)
                ),

                # Shareholders' equity
                "stockholdersEquity": _million(
                    get_value(["Stockholders Equity"], col)
                ),
                "totalEquity": _million(
                    get_value([
                        "Total Equity Gross Minority Interest",
                        "Stockholders Equity",
                    ], col)
                ),
                "retainedEarnings": _million(
                    get_value(["Retained Earnings"], col)
                ),
                "commonStock": _million(
                    get_value(["Common Stock"], col)
                ),

                # Working capital and net debt
                "workingCapital": _million(
                    get_value(["Working Capital"], col)
                ),
                "netDebt": _million(
                    get_value(["Net Debt"], col)
                ),
            }

        # Selected current metrics from the company profile.
        result["current_metrics"] = {
            "totalCash": _million(info.get("totalCash")),
            "totalDebt": _million(info.get("totalDebt")),
            "currentRatio": _float(info.get("currentRatio")),
            "quickRatio": _float(info.get("quickRatio")),
            "debtToEquity": _float(info.get("debtToEquity")),
            "bookValuePerShare": _float(info.get("bookValue")),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "marketCap": _million(info.get("marketCap")),
        }

        output = json.dumps(result, indent=2, default=str)
        _cache.store(_key, output)

        return output

    except Exception as e:
        return {
            "error": str(e),
            "ticker": ticker,
        }





@tool
def get_cash_flow_statement(ticker: str) -> str:
    """
    Retrieve the annual cash flow statement for a publicly traded company.

    Fetches reported financial data for the specified stock ticker,
    including operating cash flow, capital expenditure, free cash flow,
    depreciation and amortization, stock-based compensation, changes
    in working capital, investing cash flow, financing cash flow,
    dividends paid, share repurchases, debt issued, and debt repayment.

    Returns the latest four available annual reporting periods.

    All monetary values are returned in millions of the reporting
    currency. Cash flow values retain their reported signs: cash
    outflows are generally negative.

    Args:
        ticker (str): The ticker symbol of the company, such as "AAPL"
            for Apple or "MSFT" for Microsoft.

    Returns:
        A JSON string containing company information, annual cash flow
        data, and selected cash flow metrics. Returns an error dictionary
        if the data cannot be retrieved.
    """

    ticker = ticker.upper().strip()
    _key = ("get_cash_flow_statement", ticker)

    _hit = _cache.get(_key)
    if _hit is not None:
        return _hit

    try:
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow

        if cash_flow is None or cash_flow.empty:
            return {
                "error": (
                    f"No cash flow statement data found for ticker '{ticker}'."
                )
            }

        info = stock.info or {}
        curr = info.get("currency", "USD")

        # Safely retrieve metrics that may not be available for every company.
        def get_value(row_names, col):
            for row_name in row_names:
                if row_name in cash_flow.index:
                    return cash_flow.at[row_name, col]
            return None

        result: dict = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "currency": curr,
            "unit": f"millions {curr}",
            "annual_data": {},
        }

        # Latest four available annual reporting periods.
        for col in cash_flow.columns[:4]:
            result["annual_data"][col.year] = {
                # Operating activities
                "operatingCashFlow": _million(
                    get_value(["Operating Cash Flow"], col)
                ),
                "netIncome": _million(
                    get_value(["Net Income From Continuing Operations"], col)
                ),
                "depreciationAndAmortization": _million(
                    get_value([
                        "Depreciation And Amortization",
                        "Depreciation Amortization Depletion",
                    ], col)
                ),
                "stockBasedCompensation": _million(
                    get_value(["Stock Based Compensation"], col)
                ),
                "changeInWorkingCapital": _million(
                    get_value(["Change In Working Capital"], col)
                ),
                "changeInReceivables": _million(
                    get_value(["Changes In Accounts Receivable"], col)
                ),
                "changeInInventory": _million(
                    get_value(["Change In Inventory"], col)
                ),
                "changeInPayables": _million(
                    get_value(["Change In Account Payable"], col)
                ),

                # Investing activities
                "investingCashFlow": _million(
                    get_value(["Investing Cash Flow"], col)
                ),
                "capitalExpenditure": _million(
                    get_value([
                        "Capital Expenditure",
                        "Capital Expenditures",
                    ], col)
                ),
                "investmentsPurchased": _million(
                    get_value(["Purchase Of Investment"], col)
                ),
                "investmentsSold": _million(
                    get_value(["Sale Of Investment"], col)
                ),
                "acquisitions": _million(
                    get_value(["Net Business Purchase And Sale"], col)
                ),

                # Financing activities
                "financingCashFlow": _million(
                    get_value(["Financing Cash Flow"], col)
                ),
                "dividendsPaid": _million(
                    get_value(["Cash Dividends Paid"], col)
                ),
                "shareRepurchases": _million(
                    get_value(["Repurchase Of Capital Stock"], col)
                ),
                "debtIssued": _million(
                    get_value(["Long Term Debt Issuance"], col)
                ),
                "debtRepaid": _million(
                    get_value(["Long Term Debt Payments"], col)
                ),
                "netShortTermDebtIssuance": _million(
                    get_value(["Net Short Term Debt Issuance"], col)
                ),

                # Overall cash movement
                "freeCashFlow": _million(
                    get_value(["Free Cash Flow"], col)
                ),
                "beginningCashPosition": _million(
                    get_value(["Beginning Cash Position"], col)
                ),
                "endingCashPosition": _million(
                    get_value(["End Cash Position"], col)
                ),
                "changeInCash": _million(
                    get_value(["Changes In Cash"], col)
                ),
                "effectOfExchangeRateChanges": _million(
                    get_value(["Effect Of Exchange Rate Changes"], col)
                ),
            }

        # Selected current metrics from the company profile.
        result["current_metrics"] = {
            "operatingCashFlow": _million(
                info.get("operatingCashflow")
            ),
            "freeCashFlow": _million(
                info.get("freeCashflow")
            ),
            "capitalExpenditure": _million(
                info.get("capitalExpenditures")
            ),
        }

        output = json.dumps(result, indent=2, default=str)
        _cache.store(_key, output)

        return output

    except Exception as e:
        return {
            "error": str(e),
            "ticker": ticker,
        }




@tool
def compute_key_ratios(ticker: str) -> str:
    """
    Compute key financial ratios for a publicly traded company.

    Calculates valuation, profitability, liquidity, leverage, and
    cash-flow ratios using available financial statements and market data.

    Metrics include:
    - Valuation: trailing P/E, price-to-book, price-to-sales.
    - Profitability: gross margin, operating margin, net profit margin,
      return on equity, and return on assets.
    - Liquidity: current ratio and quick ratio.
    - Leverage: debt-to-equity and debt-to-assets.
    - Cash flow: operating cash flow to total debt and free cash flow margin.

    Args:
        ticker (str): The company's stock ticker symbol, such as "AAPL".

    Returns:
        A JSON string containing the company name, reporting currency,
        ratio values, and reporting period. Returns an error dictionary
        if the data cannot be retrieved.

    Note:
        Ratios are calculated only when the required inputs are available
        and meaningful. Margins and returns are expressed as decimals,
        e.g. 0.15 means 15%. Monetary amounts are expressed in millions
        of the reporting currency.
    """

    ticker = ticker.upper().strip()
    _key = ("compute_key_ratios", ticker)

    _hit = _cache.get(_key)
    if _hit is not None:
        return _hit

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        income = stock.financials
        balance = stock.balance_sheet
        cash_flow = stock.cashflow

        if (
            income is None or income.empty
            or balance is None or balance.empty
        ):
            return {
                "error": f"Insufficient financial statements for '{ticker}'.",
                "ticker": ticker,
            }

        currency = info.get("currency", "USD")

        # Safely retrieve a financial statement value.
        def get_value(statement, row_names, col):
            for row in row_names:
                if row in statement.index:
                    value = statement.at[row, col]
                    if value is not None:
                        try:
                            value = float(value)
                            if math.isfinite(value):
                                return value
                        except (TypeError, ValueError):
                            continue
            return None

        # Safely divide two numbers.
        def divide(numerator, denominator):
            if numerator is None or denominator is None:
                return None
            if denominator == 0:
                return None
            return _float(numerator / denominator)

        # Most recent annual reporting period.
        income_col = income.columns[0]
        balance_col = balance.columns[0]

        revenue = get_value(income, ["Total Revenue"], income_col)
        gross_profit = get_value(income, ["Gross Profit"], income_col)
        operating_income = get_value(
            income, ["Operating Income"], income_col
        )
        net_income = get_value(income, ["Net Income"], income_col)

        total_assets = get_value(balance, ["Total Assets"], balance_col)
        current_assets = get_value(
            balance, ["Current Assets"], balance_col
        )
        current_liabilities = get_value(
            balance, ["Current Liabilities"], balance_col
        )
        inventory = get_value(balance, ["Inventory"], balance_col)
        equity = get_value(
            balance,
            [
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
            ],
            balance_col,
        )
        total_liabilities = get_value(
            balance,
            [
                "Total Liabilities Net Minority Interest",
                "Total Liabilities",
            ],
            balance_col,
        )
        total_debt = get_value(balance, ["Total Debt"], balance_col)

        operating_cash_flow = None
        free_cash_flow = None

        if cash_flow is not None and not cash_flow.empty:
            cash_col = cash_flow.columns[0]
            operating_cash_flow = get_value(
                cash_flow, ["Operating Cash Flow"], cash_col
            )
            free_cash_flow = get_value(
                cash_flow, ["Free Cash Flow"], cash_col
            )

            # Calculate FCF if it is not directly reported.
            if free_cash_flow is None and operating_cash_flow is not None:
                capex = get_value(
                    cash_flow,
                    ["Capital Expenditure", "Capital Expenditures"],
                    cash_col,
                )
                if capex is not None:
                    free_cash_flow = operating_cash_flow + capex

        # Current market price and market capitalization.
        price = _float(
            info.get("currentPrice", info.get("regularMarketPrice"))
        )
        market_cap = info.get("marketCap")

        eps = _float(info.get("trailingEps"))
        book_value_per_share = _float(info.get("bookValue"))
        trailing_revenue = _float(info.get("totalRevenue"))

        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "currency": currency,
            "reporting_period": income_col.strftime("%Y-%m-%d"),
            "unit": f"millions {currency} for monetary statement values",

            "valuation_ratios": {
                "trailingPE": divide(price, eps),
                "priceToBook": divide(price, book_value_per_share),
                "priceToSales": divide(market_cap, trailing_revenue),
            },

            "profitability_ratios": {
                "grossMargin": divide(gross_profit, revenue),
                "operatingMargin": divide(operating_income, revenue),
                "netProfitMargin": divide(net_income, revenue),
                "returnOnEquity": divide(net_income, equity),
                "returnOnAssets": divide(net_income, total_assets),
            },

            "liquidity_ratios": {
                "currentRatio": divide(
                    current_assets, current_liabilities
                ),
                "quickRatio": divide(
                    None if current_assets is None or inventory is None
                    else current_assets - inventory,
                    current_liabilities,
                ),
            },

            "leverage_ratios": {
                "debtToEquity": divide(total_debt, equity),
                "debtToAssets": divide(total_liabilities, total_assets),
            },

            "cash_flow_ratios": {
                "operatingCashFlowToDebt": divide(
                    operating_cash_flow, total_debt
                ),
                "freeCashFlowMargin": divide(
                    free_cash_flow, revenue
                ),
            },
        }

        output = json.dumps(result, indent=2, default=str)
        _cache.store(_key, output)

        return output

    except Exception as e:
        return {
            "error": str(e),
            "ticker": ticker,
        }
