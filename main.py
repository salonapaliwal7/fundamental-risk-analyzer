import os

from dotenv import load_dotenv
from strands.models.openai import OpenAIModel
from agents.orchestrator import create_orchestrator
from agents.valuation_agent import create_valuation_agent

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

MODEL_ID = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

model = OpenAIModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        ),
        "max_retries": 5,
    },
    model_id=MODEL_ID,
    params={
        "extra_body": {
            "include_reasoning": False,
        },
    },
)
print(f"Using {MODEL_ID} on Groq")


orchestrator = create_orchestrator(model)

result = orchestrator("AAPL")
print(result)


