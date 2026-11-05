import math

import yfinance as yf
from strands import Agent, tool
from strands_tools.tavily import tavily_search
import json
from tools.financial_tools import _cache, _float, _million



@tool
def run_dcf_model(
    ticker: str,
    growth_rate: float = 0.08,
    discount_rate: float = 0.09,
    terminal_growth_rate: float = 0.03,
) -> str:
    """
    Estimate intrinsic value using a 5-year Discounted Cash Flow model.

    Projects free cash flow for five years, discounts the projected
    cash flows, calculates terminal value, and estimates intrinsic
    value per share.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
        growth_rate: Annual FCF growth rate as a decimal, e.g. 0.10.
        discount_rate: Discount rate as a decimal, e.g. 0.09.
        terminal_growth_rate: Perpetual growth rate as a decimal,
            e.g. 0.03.

    Returns:
        JSON string containing intrinsic_value, current_price,
        upside_pct, and verdict.
    """

    ticker = ticker.upper().strip()
    _key = (
        "run_dcf_model",
        ticker,
        growth_rate,
        discount_rate,
        terminal_growth_rate,
    )

    _hit = _cache.get(_key)
    if _hit is not None:
        return _hit

    try:
        if not all(
            math.isfinite(rate)
            for rate in (growth_rate, discount_rate, terminal_growth_rate)
        ):
            raise ValueError("All rates must be finite numbers.")

        if growth_rate <= -1:
            raise ValueError("Growth rate must be greater than -100%.")

        if discount_rate <= terminal_growth_rate:
            raise ValueError(
                "Discount rate must exceed terminal growth rate."
            )

        if terminal_growth_rate <= -1 or discount_rate <= -1:
            raise ValueError(
                "Discount and terminal growth rates must exceed -100%."
            )

        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow
        balance_sheet = stock.balance_sheet
        info = stock.info or {}

        if cash_flow is None or cash_flow.empty:
            raise ValueError("No cash flow statement data found.")

        # Get the latest annual free cash flow.
        latest_period = cash_flow.columns[0]

        base_fcf = None
        if "Free Cash Flow" in cash_flow.index:
            base_fcf = _float(
                cash_flow.at["Free Cash Flow", latest_period]
            )

        # Calculate FCF if it is not directly available.
        if base_fcf is None:
            def get_value(statement, rows, period):
                for row in rows:
                    if row in statement.index:
                        value = _float(statement.at[row, period])
                        if value is not None:
                            return value
                return None

            operating_cash_flow = get_value(
                cash_flow, ["Operating Cash Flow"], latest_period
            )
            capex = get_value(
                cash_flow,
                ["Capital Expenditure", "Capital Expenditures"],
                latest_period,
            )

            if operating_cash_flow is not None and capex is not None:
                base_fcf = operating_cash_flow + capex

        if base_fcf is None:
            raise ValueError("Unable to determine free cash flow.")

        if base_fcf <= 0:
            raise ValueError(
                "Standard DCF is unsuitable for zero or negative "
                "historical free cash flow."
            )

        # Project and discount FCF for five years.
        projected_fcf = base_fcf
        pv_fcf = 0.0

        for year in range(1, 6):
            projected_fcf *= 1 + growth_rate
            pv_fcf += projected_fcf / ((1 + discount_rate) ** year)

        # Terminal value at the end of Year 5.
        terminal_fcf = projected_fcf * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (
            discount_rate - terminal_growth_rate
        )
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** 5)

        # Enterprise value.
        enterprise_value = pv_fcf + pv_terminal_value

        # Get cash, debt, and shares.
        cash = _float(info.get("totalCash"))
        debt = _float(info.get("totalDebt"))
        shares = _float(info.get("sharesOutstanding"))

        if balance_sheet is not None and not balance_sheet.empty:
            period = balance_sheet.columns[0]

            if cash is None:
                for row in [
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                ]:
                    if row in balance_sheet.index:
                        cash = _float(balance_sheet.at[row, period])
                        if cash is not None:
                            break

            if debt is None and "Total Debt" in balance_sheet.index:
                debt = _float(balance_sheet.at["Total Debt", period])

        if cash is None or debt is None:
            raise ValueError(
                "Unable to retrieve cash and total debt for equity valuation."
            )

        if shares is None or shares <= 0:
            raise ValueError("Valid shares outstanding are unavailable.")

        # Convert enterprise value to equity value, then per-share value.
        equity_value = enterprise_value + cash - debt
        intrinsic_value = equity_value / shares

        price = _float(
            info.get("currentPrice", info.get("regularMarketPrice"))
        )

        if price is None or price <= 0:
            raise ValueError("Current market price is unavailable.")

        upside = (intrinsic_value / price - 1) * 100

        result = {
            "intrinsic_value": round(intrinsic_value, 2),
            "current_price": round(price, 2),
            "upside_pct": round(upside, 1),
            "verdict": _dcf_verdict(upside),
        }

        output = json.dumps(result, indent=2)
        _cache.store(_key, output)
        return output

    except Exception as e:
        return json.dumps({
            "ticker": ticker,
            "error": str(e),
        })


def _dcf_verdict(upside: float) -> str:
    if upside >= 30:
        return "Significantly Undervalued"
    elif upside >= 10:
        return "Slightly Undervalued"
    elif upside >= -10:
        return "Fairly Valued"
    elif upside > -30:
        return "Slightly Overvalued"
    else:
        return "Significantly Overvalued"



@tool
def get_peer_multiples(ticker: str) -> str:
    """
    Compare a company's valuation multiples with comparable companies.

    Identifies potential peers using Yahoo Finance industry information,
    retrieves their trailing P/E, price-to-book, price-to-sales, and
    enterprise-value-to-EBITDA multiples, and calculates peer-group
    medians.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        JSON string containing the target company's multiples, peer
        multiples, peer medians, relative valuation verdict, and the
        list of companies included in the comparison.

    Note:
        Peer selection and available metrics depend on Yahoo Finance.
        Relative valuation does not establish intrinsic value.
    """
    ticker = ticker.upper().strip()
    _key = ("get_peer_multiples", ticker)

    _hit = _cache.get(_key)
    if _hit is not None:
        return _hit

    try:
        import statistics

        stock = yf.Ticker(ticker)
        info = stock.info or {}

        if not info:
            return json.dumps({
                "ticker": ticker,
                "error": "Company information is unavailable."
            })

        industry_key = info.get("industryKey")
        sector = info.get("sector")
        industry = info.get("industry")

        # Multiples used consistently for target and peers.
        metric_keys = {
            "PE": "trailingPE",
            "PB": "priceToBook",
            "PS": "priceToSalesTrailing12Months",
            "EV_EBITDA": "enterpriseToEbitda",
        }

        def extract_multiples(company_info):
            return {
                metric: _float(company_info.get(info_key))
                for metric, info_key in metric_keys.items()
            }

        target_multiples = extract_multiples(info)

        # Find candidate companies in the same industry.
        peer_symbols = []

        if industry_key:
            industry_data = yf.Industry(industry_key).top_companies

            if industry_data is not None and not industry_data.empty:
                # yfinance's returned column names may vary.
                symbol_column = next(
                    (
                        col for col in industry_data.columns
                        if str(col).lower() in
                        ("symbol", "ticker", "stock symbol")
                    ),
                    None,
                )

                if symbol_column is not None:
                    peer_symbols = [
                        str(symbol).strip()
                        for symbol in industry_data[symbol_column].tolist()
                        if symbol and str(symbol).strip().upper() != ticker
                    ]

        # Avoid excessive requests and duplicate symbols.
        peer_symbols = list(dict.fromkeys(peer_symbols))[:10]

        peers = []

        for peer_symbol in peer_symbols:
            try:
                peer_info = yf.Ticker(peer_symbol).info or {}

                if not peer_info:
                    continue

                peers.append({
                    "ticker": peer_symbol,
                    "company_name": peer_info.get(
                        "longName", peer_symbol
                    ),
                    "multiples": extract_multiples(peer_info),
                })

            except Exception:
                # Skip unavailable peers without failing the entire tool.
                continue

        # Compute medians from valid, positive multiples only.
        peer_medians = {}

        for metric in metric_keys:
            values = [
                peer["multiples"][metric]
                for peer in peers
                if peer["multiples"][metric] is not None
                and peer["multiples"][metric] > 0
            ]

            peer_medians[metric] = (
                round(statistics.median(values), 2)
                if values else None
            )

        verdict = peer_verdict(target_multiples, peer_medians)

        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": sector,
            "industry": industry,
            "target_multiples": target_multiples,
            "peer_medians": peer_medians,
            "peer_count": len(peers),
            "peers": peers,
            "peer_verdict": verdict,
        }

        output = json.dumps(result, indent=2, default=str)
        _cache.store(_key, output)
        return output

    except Exception as e:
        return json.dumps({
            "ticker": ticker,
            "error": str(e),
        })

def peer_verdict(target_multiples, peer_medians):
    WEIGHTS = {
    "pe_ratio": 0.35,
    "pb_ratio": 0.20,
    "ps_ratio": 0.20,
    "ev_ebitda": 0.25,
}

WEIGHTS = {
    "pe_ratio": 0.35,
    "pb_ratio": 0.20,
    "ps_ratio": 0.20,
    "ev_ebitda": 0.25,
}

def _peer_verdict(subject, peer_avg):
    total, weight_used = 0.0, 0.0

    for key, weight in WEIGHTS.items():
        s, p = subject.get(key), peer_avg.get(key)

        # Skip multiples that are missing or invalid
        if s is None or p is None or s <= 0 or p <= 0:
            continue

        # Percentage premium/discount relative to peers
        total += weight * (s / p - 1) * 100
        weight_used += weight

    if weight_used == 0:
        return "N/A", None

    # Re-scale so skipped multiples don't affect the score
    weighted_avg = round(total / weight_used, 1)

    if weighted_avg > 30:
        verdict = "SIGNIFICANTLY OVERVALUED"
    elif weighted_avg > 15:
        verdict = "SLIGHTLY OVERVALUED"
    elif weighted_avg >= -15:
        verdict = "FAIRLY VALUED"
    elif weighted_avg >= -30:
        verdict = "SLIGHTLY UNDERVALUED"
    else:
        verdict = "SIGNIFICANTLY UNDERVALUED"

    return verdict, weighted_avg