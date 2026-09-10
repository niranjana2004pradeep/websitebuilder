import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


class WebsiteOutput(BaseModel):
    html: str = Field(description="The HTML code for the website")
    css: str = Field(description="The CSS code for styling the HTML")
    js: str = Field(description="The JavaScript code for website behavior")


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert web developer.

        generate a simple website based on user request
        Return:
        HTML
        CSS
        JavaScript
        Do not include any additional text or markdown fences."""
    ),
    (
        "human",
        "{description}"
    ),
])


structured_llm = llm.with_structured_output(WebsiteOutput)


chain = prompt | structured_llm


result = chain.invoke({
    "description": "Create a simple landing page for a coffee shop"
})

with open("index.html", "w", encoding="utf-8") as f:
    f.write(result.html)

with open("style.css", "w", encoding="utf-8") as f:
    f.write(result.css)

with open("script.js", "w", encoding="utf-8") as f:
    f.write(result.js)