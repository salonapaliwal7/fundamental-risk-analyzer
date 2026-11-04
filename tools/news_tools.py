import math

import yfinance as yf
from strands import Agent, tool
from strands_tools.tavily import tavily_search
import json

import os

# disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from tools.financial_tools import _cache
from transformers import pipeline



_finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    device=-1
)



def _score_headline(text: str):
    if not text or not text.strip():
        return "neutral", 0.0

    predictions = _finbert(text, top_k=None)[0]

    scores = {
        item["label"].lower(): item["score"]
        for item in predictions
    }

    positive = scores.get("positive", 0.0)
    negative = scores.get("negative", 0.0)
    neutral = scores.get("neutral", 0.0)

    # Sentiment score always falls between -1 and +1
    net = positive - negative

    if net > 0.15:
        sentiment = "bullish"
    elif net < -0.15:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return sentiment, round(net, 2)


@tool
def search_recent_news(ticker: str, max_results: int = 5) -> str:
    """
    Search for recent news articles about a stock ticker.

    Args:
        ticker: Stock ticker symbol, such as AAPL or TSLA.
        max_results: Maximum number of news articles to retrieve.

    Returns:
        A JSON string containing the retrieved news articles.
    """

    try:
        search = yf.Search(
            ticker,
            news_count=max_results,
        )

        news_items = search.news or []

        articles = []

        for item in news_items[:max_results]:
            content = item.get("content", item)

            title = content.get("title", "")
            publisher = content.get("publisher", "")

            summary = content.get("summary", "")

            url = content.get("canonicalUrl", {})
            if isinstance(url, dict):
                url = url.get("url", "")

            date = content.get("pubDate", "")

            if not date:
                date = content.get("providerPublishTime", "")

            articles.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "date": date,
                    "url": url,
                    "summary": summary,
                }
            )

        result = {
            "ticker": ticker,
            "max_results": max_results,
            "results_count": len(articles),
            "news": articles,
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {
                "ticker": ticker,
                "error": str(e),
                "results_count": 0,
                "news": [],
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------
# Sentiment analysis function
# ---------------------------------------------------------

def analyze_sentiment(news_json: str) -> str:
    """
    Analyze a collection of news articles using FinBERT.

    Args:
        news_json: JSON string returned by search_recent_news.

    Returns:
        JSON string containing article-level and overall sentiment.
    """

    try:
        data = json.loads(news_json)

    except json.JSONDecodeError:
        return json.dumps(
            {
                "error": "Invalid JSON input",
                "articles": [],
            }
        )

    articles = data.get("news", [])

    if not articles:
        return json.dumps(
            {
                "ticker": data.get("ticker", ""),
                "results_count": 0,
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "articles": [],
                "summary": "No news articles were found.",
            },
            ensure_ascii=False,
        )

    analyzed_articles = []

    positive_scores = []
    negative_scores = []
    neutral_scores = []

    for article in articles:
        title = article.get("title", "")
        summary = article.get("summary", "")

        # Combine headline and summary for more context.
        text = f"{title}. {summary}".strip()

        if not text:
            continue

        try:
            prediction = _finbert(
                text,
                truncation=True,
                max_length=512,
            )[0]

            model_label = prediction["label"].lower()
            confidence = float(prediction["score"])

            # FinBERT labels are typically positive, negative, neutral.
            if model_label == "positive":
                sentiment = "positive"
                score = confidence
                positive_scores.append(score)

            elif model_label == "negative":
                sentiment = "negative"
                score = -confidence
                negative_scores.append(score)

            else:
                sentiment = "neutral"
                score = 0.0
                neutral_scores.append(score)

        except Exception as e:
            sentiment = "neutral"
            score = 0.0

        analyzed_articles.append(
            {
                "title": title,
                "publisher": article.get("publisher", ""),
                "date": article.get("date", ""),
                "url": article.get("url", ""),
                "sentiment": sentiment,
                "score": round(score, 4),
                "explanation": (
                    f"FinBERT classified this article as {sentiment}."
                ),
            }
        )

    # -----------------------------------------------------
    # Aggregate sentiment
    # -----------------------------------------------------

    total = len(analyzed_articles)

    positive_count = sum(
        1 for article in analyzed_articles
        if article["sentiment"] == "positive"
    )

    negative_count = sum(
        1 for article in analyzed_articles
        if article["sentiment"] == "negative"
    )

    neutral_count = sum(
        1 for article in analyzed_articles
        if article["sentiment"] == "neutral"
    )

    scores = [
        article["score"]
        for article in analyzed_articles
    ]

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    # Overall verdict based on the average sentiment score.
    if average_score > 0.15:
        overall_sentiment = "positive"

    elif average_score < -0.15:
        overall_sentiment = "negative"

    else:
        overall_sentiment = "neutral"

    result = {
        "ticker": data.get("ticker", ""),
        "results_count": total,
        "overall_sentiment": overall_sentiment,
        "average_score": round(average_score, 4),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "articles": analyzed_articles,
        "summary": (
            f"News sentiment for {data.get('ticker', 'the stock')} "
            f"is {overall_sentiment}. "
            f"Analyzed {total} articles: "
            f"{positive_count} positive, "
            f"{negative_count} negative, and "
            f"{neutral_count} neutral."
        ),
    }

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------
# Combined tool exposed to the news sentiment agent
# ---------------------------------------------------------

@tool
def get_news_with_sentiment(
    ticker: str,
    max_results: int = 5,
) -> str:
    """
    Fetch recent news for a stock ticker and analyze its sentiment.

    Args:
        ticker: Stock ticker symbol, such as AAPL or TSLA.
        max_results: Maximum number of news articles to analyze.

    Returns:
        JSON string containing news articles and sentiment analysis.
    """

    news_json = search_recent_news(
        ticker=ticker,
        max_results=max_results,
    )

    sentiment_json = analyze_sentiment(news_json)

    return sentiment_json
