import os

from dotenv import load_dotenv
from strands.models.openai import OpenAIModel
from agents.orchestrator import create_orchestrator

load_dotenv()  # reads GROQ_API_KEY from a .env file, if present

MODEL_ID = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

model = OpenAIModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),  # or a local OpenAI-compatible server
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by default; hide it so only the answer prints.
    params={"extra_body": {"include_reasoning": False}},
)
print(f"Using {MODEL_ID} on Groq")


orchestrator = create_orchestrator(model)

result = orchestrator("AAPL")
print(result)


