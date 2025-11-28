from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core import Settings

from tools.swift_expert import SwiftExpert

import asyncio

"""
what is an agent?
A central brain that picks which tool to use depending on your question.

User → Agent → (Agent decides to call tool) → You execute tool → Return result back → Agent finishes answer

"""

import requests
def sendRequest(message: str):
    url = "http://raspberrypi.local:8000/print/text"

    response = requests.post(url, data={"content": message})

Settings.llm = Ollama(model="qwen2.5-coder:3b", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

def pocoyoanswer(question: str):
    print("experto")

def create_agent():
    llm = Ollama(model="qwen2.5-coder:3b", request_timeout=60)

    swift_tool_instance = SwiftExpert()

    swift_tool = FunctionTool.from_defaults(
        name="swift_expert",
        description="Answer iOS development and Swift questions using local project knowledge.",
        fn=swift_tool_instance.answer
    )

    pocoyo_tool = FunctionTool.from_defaults(
        name="pocoyo_expert",
        description="Answer questions related to TV show pocoyo",
        fn=pocoyoanswer
    )

    agent = FunctionAgent(
        tools=[swift_tool, pocoyo_tool],
        llm=llm,
        system_prompt="You are an expert iOS engineer that can analyze swift code. Also likes Pocoyo TV show.",
    )

    return agent


async def run_agent(agent, query):
    # Step 1: LLM produces either a tool call OR a final answer
    result = await agent.run(query)

    # If it's a dict, then it's a tool call (FunctionAgent returns dicts)
    if isinstance(result, dict) and "name" in result:
        tool_name = result["name"]
        args = result["arguments"]

        # Find the tool
        for tool in agent.tools:
            if tool.metadata.name == tool_name:
                tool_asyncresponse = tool.call(**args)

                # Feed tool output back to agent
                final = await agent.run(tool_response)
                return final

    # Otherwise it's a normal LLM answer
    return result

async def main():
    # Run the agent
    agent = create_agent()

    while True:
        q = input("\nAsk the Swift Agent: ")
        final_answer = await run_agent(agent, q)
        print("\n🧠 Agent Answer:\n", final_answer)
        sendRequest(str(final_answer))

if __name__ == "__main__":
    print("📱 Swift Dev Agent ready!")
    asyncio.run(main())