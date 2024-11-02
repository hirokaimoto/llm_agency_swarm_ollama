
import openai
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import datetime
from tools import WebSearchTool, ImageDescriptionGeneratorTool, ImageModificationTool, CameraFaceCaptureTool


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

client = OpenAI()

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list

def run_full_turn(agent, messages):

    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [openai.pydantic_function_tool(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)
    

        if message.content:  # print agent response
            print("[Decided a final answer]")
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"Transfered to {current_agent.name}. Adopt persona immediately."
                )

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])


def execute_tool_call(tool_call, tools, agent_name):
    print(f"Tool called: {tool_call} | tools: {tools} | Agent: {agent_name}")
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)


    return tools[name](**args).run()  # call corresponding function with provided arguments



class TransferToAiImageAgent(BaseModel):
    """User for anything related to image generation, image modification and ocr."""

    def run(self):
        """User for anything related to image generation, image modification and ocr."""
        return image_ai_agent
    
class TransferToAiWebAgent(BaseModel):
    """User for anything related web search and real time data from web."""

    def run(self):
        """User for anything related to image generation, image modification and ocr."""
        return web_agent


class TransferBackToTriageAgent(BaseModel):
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""

    def run(self):
        """User for anything related to image generation, image modification and ocr."""
        return triage_agent


current_data = datetime.datetime.now().strftime("%d/%m/%Y")  # Format date as DD/MM/YYYY


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a Triage AI assistant. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right Agent. "
        "But make your questions subtle and natural."
        f"Current date: {current_data}"
    ),
    tools=[TransferToAiImageAgent, TransferToAiWebAgent],
)



image_ai_agent = Agent(
    name="Image AI Agent",
    instructions=(
        "You are an AI Image Agent."
        "Use your available tools to fulfill the user requests."
        f"Current date: {current_data}"
    ),
    tools=[CameraFaceCaptureTool, ImageDescriptionGeneratorTool, ImageModificationTool, TransferBackToTriageAgent],
)



web_agent = Agent(
    name="Web search Agent",
    instructions=(
        "You are a Web Search AI agent"
        "Always answer in a sentence or less."
        "Help the user accomplish what he needs, using your available tools."
        f"Current date: {current_data}"
    ),
    tools=[WebSearchTool, TransferBackToTriageAgent],
)

agent = triage_agent
messages = []

while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)

