from fastapi import APIRouter
from fastapi.websockets import WebSocket

router = APIRouter()
ws_clients = set()

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    print("[WS] Client connected")

    try:
        while True:
            await ws.receive_text()
    except:
        ws_clients.remove(ws)
        print("[WS] Client disconnected")


async def broadcast(event_type: str, data: dict):
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_json({"type": event_type, "data": data})
        except:
            dead.append(ws)

    for ws in dead:
        ws_clients.remove(ws)
