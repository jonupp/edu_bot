
from datetime import datetime
import json
import os
from contextlib import AsyncExitStack
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langfuse import get_client
from langfuse.langchain import CallbackHandler



CURRENT_DATE = datetime.now(
    ZoneInfo("Europe/Zurich")
).strftime("%B %d, %Y")

SYSTEM_PROMPT = f"""
You are a helpful, accurate, and concise assistant named Edu Bot. Always answer in the user's language.

## Current date
Today is {CURRENT_DATE}. Interpret all temporal expressions relative to this date, regardless of your knowledge cutoff. Do not treat past or current events as future events merely because they occurred after your training cutoff.

## Knowledge and retrieval
- Use built in knowledge for stable, well established facts.
- For mathematical calcuations and unit conversions, use the appropriate tools instead of calculating mentally.
- If you do not know an answer confidently, or the question concerns recent, changing, or specific factual information, search available tools for reliable sources.
- For factual questions about topics covered by Wikipedia, prefer searching Wikipedia and fetching the relevant article to inspect its contents before answering. Do not rely on search snippets alone when the full article is available.
- Treat retrieved evidence as more authoritative than uncertain recollection, while evaluating source reliability and consistency.
- Never assume an event did not happen simply because it is absent from your built in knowledge.
- Never invent facts, dates, events, results, or sources. If information cannot be verified, state what remains uncertain.

## Tool usage
Use tools when they materially improve accuracy or are required to complete an action. Do not claim to have searched, retrieved, or completed anything unless the tool confirms it.

## Response style
Answer directly and concisely. Distinguish verified facts from estimates and speculation. Use absolute dates when relative dates could be ambiguous.
""".strip()


class AgentService:
    def __init__(self):
        load_dotenv("agent.env", override=False)

        self.mcp_url = os.getenv(
            "MCP_URL", "http://127.0.0.1:9999/mcp"
        )
        self.checkpoint_path = os.getenv(
            "CHECKPOINT_PATH", "checkpoints.sqlite"
        )

        self.model = ChatOpenAI(
            model=os.getenv(
                "MODEL_NAME",
                "Qwen/Qwen3-4B-Instruct-2507-FP8",
            ),
            base_url=os.getenv(
                "INFERENCE_API_URL",
                "http://192.168.1.13:8000/v1",
            ),
            api_key=os.getenv("INFERENCE_API_KEY", "SECRET"),
        )

        self.stack = AsyncExitStack()
        self.agent = None
        self.tools_by_name = {}
        self.langfuse_handler = None
        self._started = False

    async def start(self):
        if self._started:
            return

        try:
            # Keep MCP and SQLite resources alive for the service lifetime.
            adapter = await self.stack.enter_async_context(
                MCPAdapter(self.mcp_url)
            )

            tools = await adapter.list_tools()
            self.tools_by_name = {
                tool.name: tool for tool in tools
            }

            model_with_tools = self.model.bind_tools(tools)

            checkpointer = await self.stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(
                    self.checkpoint_path
                )
            )
            self.checkpointer = checkpointer

            # Explicit setup initializes checkpoint storage when required
            # by the installed LangGraph checkpoint-sqlite version.
            await checkpointer.setup()

            get_client()
            self.langfuse_handler = CallbackHandler()

            builder = StateGraph(MessagesState)
            builder.add_node(
                "assistant",
                self._make_assistant_node(model_with_tools),
            )
            builder.add_node("tools", self._run_tools)

            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                self._route,
                {
                    "tools": "tools",
                    "end": END,
                },
            )
            builder.add_edge("tools", "assistant")

            self.agent = builder.compile(
                checkpointer=checkpointer
            )
            self._started = True

            print(
                f"Agent started. Loaded {len(tools)} MCP tools."
            )

        except Exception:
            await self.stack.aclose()
            raise

    def _make_assistant_node(self, model_with_tools):
        async def assistant_node(state: MessagesState):
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]

            response = await model_with_tools.ainvoke(messages)
            return {"messages": [response]}

        return assistant_node

    @staticmethod
    def _route(state: MessagesState):
        last_message = state["messages"][-1]

        if isinstance(last_message, AIMessage):
            if last_message.tool_calls:
                return "tools"

        return "end"

    async def _run_tools(self, state: MessagesState):
        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            return {"messages": []}

        results = []

        for call in last_message.tool_calls:
            name = call["name"]
            args = call["args"]
            call_id = call["id"]

            tool = self.tools_by_name.get(name)

            if tool is None:
                result = f"Error: unknown tool '{name}'."
            else:
                try:
                    result = await tool.ainvoke(args)
                except Exception as exc:
                    # Return a tool error to the model so it can recover.
                    result = {
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }

            results.append(
                ToolMessage(
                    content=self._to_text(result),
                    name=name,
                    tool_call_id=call_id,
                )
            )

        return {"messages": results}

    @staticmethod
    def _to_text(value: Any) -> str:
        if isinstance(value, str):
            return value

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "callbacks": [self.langfuse_handler],
            "metadata": {
                "langfuse_session_id": thread_id,
            },
        }

    async def chat(self, thread_id: str, message: str) -> str:
        if not self._started or self.agent is None:
            raise RuntimeError("Agent service is not started.")

        result = await self.agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=message)
                ]
            },
            config=self._config(thread_id),
        )

        # Return the last final assistant response, not intermediate
        # messages requesting tool calls.
        for item in reversed(result["messages"]):
            if (
                isinstance(item, AIMessage)
                and not item.tool_calls
                and item.content
            ):
                return self._to_text(item.content)

        return "The agent did not produce a final response."

    async def load_messages(self, thread_id: str) -> list[dict[str, str]]:
        if not self._started or self.agent is None:
            raise RuntimeError("Agent service is not started.")

        snapshot = await self.agent.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )

        history = []

        for message in snapshot.values.get("messages", []):
            if isinstance(message, HumanMessage):
                history.append({
                    "role": "user",
                    "content": self._to_text(message.content),
                })

            elif (
                isinstance(message, AIMessage)
                and not message.tool_calls
                and message.content
            ):
                history.append({
                    "role": "assistant",
                    "content": self._to_text(message.content),
                })

        return history

    async def close(self):
        try:
            await self.stack.aclose()
        finally:
            self._started = False
            self.agent = None


    async def get_conversations(self) -> list[dict[str, str]]:
        """Return unique conversations stored in the checkpointer."""
        if self.checkpointer is None:
            raise RuntimeError("Agent service is not started.")

        conversations = {}

        async for checkpoint_tuple in self.checkpointer.alist(None):
            configurable = (
                checkpoint_tuple.config.get("configurable", {})
            )
            thread_id = configurable.get("thread_id")

            if not thread_id:
                continue

            # Checkpoints are returned newest first. Keep the first
            # checkpoint encountered for each conversation.
            if thread_id in conversations:
                continue

            checkpoint = checkpoint_tuple.checkpoint or {}

            conversations[thread_id] = {
                "thread_id": thread_id,
                "updated_at": checkpoint.get("ts", ""),
            }

        return list(conversations.values())
