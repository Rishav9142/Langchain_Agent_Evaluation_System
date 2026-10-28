from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0,
)

response = model.invoke("Tell me something about the Sun in one sentence.")
print(response.content)