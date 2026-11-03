import asyncio

from dotenv import load_dotenv
from groq import Groq
from deepeval.models import DeepEvalBaseLLM

load_dotenv()


class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return Groq()

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


eval_model = GroqEvalModel()
