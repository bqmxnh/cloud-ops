from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except:
        pass
    finally:
        ws_clients.discard(ws)


async def broadcast(event_type: str, data: dict):
    """Gửi sự kiện real-time cho toàn bộ client"""
    dead_clients = []
    for ws in ws_clients:
        try:
            await ws.send_json({"type": event_type, "data": data})
        except:
            dead_clients.append(ws)

    for ws in dead_clients:
        ws_clients.discard(ws)
