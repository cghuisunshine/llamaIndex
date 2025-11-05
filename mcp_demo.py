#!/usr/bin/env python3
import logging
import os
import random
import requests
import uvicorn
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, Mount
from mcp.server.fastmcp import FastMCP

# --- Configuration & logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("demo-mcp-server")

PORT = int(os.environ.get("PORT", 8001))

# --- MCP server (old FastMCP API: no logger, no port params) ---
mcp = FastMCP("demo-mcp-server", stateless_http=True)

# --- Tools ---
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    logger.info(f"Tool called: add({a}, {b})")
    return a + b

@mcp.tool()
def get_secret_word() -> str:
    """Get a random secret word"""
    logger.info("Tool called: get_secret_word()")
    return random.choice(["apple", "banana", "cherry"])

@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather for a city"""
    logger.info(f"Tool called: get_current_weather({city})")
    try:
        resp = requests.get(f"https://wttr.in/{city}", timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return f"Error fetching weather data: {e}"

# --- Build the HTTP app (for /mcp) ---
mcp_http = mcp.streamable_http_app()

# Lifespan: older FastMCP needs the session manager started
@asynccontextmanager
async def lifespan(_app):
    async with mcp.session_manager.run():
        yield

# --- Top-level Starlette app ---
app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/_health", lambda req: PlainTextResponse("ok")),
        Mount("/", app=mcp_http),
    ],
)

if __name__ == "__main__":
    print(f"🚀 MCP at http://127.0.0.1:{PORT}/mcp  (health: /_health)")
    uvicorn.run(app, host="127.0.0.1", port=PORT)

