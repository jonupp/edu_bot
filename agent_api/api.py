
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_service import AgentService

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(".env")

    service = AgentService()
    await service.start()
    app.state.agent_service = service

    try:
        yield
    finally:
        await service.close()


app = FastAPI(
    title="LangGraph Agent API",
    lifespan=lifespan,
)


class CreateConversationResponse(BaseModel):
    conversation_id: str


class SendMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    base64_image: str | None = Field(default=None, description="Base64-encoded jpeg, optional")


class SendMessageResponse(BaseModel):
    conversation_id: str
    reply: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/conversations",
    response_model=CreateConversationResponse,
)
async def create_conversation():
    return {"conversation_id": str(uuid4())}


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    service = app.state.agent_service
    messages = await service.load_messages(conversation_id)
    return {"messages": messages}


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=SendMessageResponse,
)
async def send_message(
    conversation_id: str,
    body: SendMessageRequest,
):
    service = app.state.agent_service
    
    try:
        reply = await service.chat(
            conversation_id,
            body.message,
            body.base64_image,
        )
        return {
            "conversation_id": conversation_id,
            "reply": reply,
        }
    except Exception as exc:
        # Log the exception server-side in production.
        print(f"Agent request failed: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Agent request failed",
        ) from exc


@app.get("/conversations")
async def get_conversations():
    service: AgentService = app.state.agent_service

    try:
        conversations = await service.get_conversations()
        return {"conversations": conversations}
    except Exception as exc:
        print(f"Could not list conversations: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve conversations.",
        ) from exc
