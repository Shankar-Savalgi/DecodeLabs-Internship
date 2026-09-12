import os
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from chatbot_engine import StatefulChatSession

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global active session instance
session = StatefulChatSession()

class ChatbotHTTPRequestHandler(SimpleHTTPRequestHandler):
    """
    HTTP Server handler providing REST API endpoints and static file web server.
    """
    
    def __init__(self, *args, **kwargs):
        # Serve static files from 'public' directory
        static_dir = os.path.join(os.path.dirname(__file__), "public")
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/api/session":
            self._send_json({
                "status": "success",
                "active_persona": session.active_persona,
                "persona_name": session.PERSONAS.get(session.active_persona, {}).get("name", "Custom"),
                "system_prompt": session.system_prompt,
                "history": session.history,
                "history_length": len(session.history),
                "estimated_tokens": session.total_tokens_estimated,
                "max_tokens": session.max_context_tokens,
                "created_at": session.created_at
            })
            return

        if parsed.path == "/api/export":
            query = parse_qs(parsed.query)
            fmt = query.get("format", ["markdown"])[0]
            if fmt == "json":
                content = session.export_json()
                content_type = "application/json"
                filename = "chat_session.json"
            else:
                content = session.export_markdown()
                content_type = "text/markdown"
                filename = "chat_session.md"

            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
            return

        if parsed.path == "/api/personas":
            self._send_json({
                "status": "success",
                "personas": session.PERSONAS,
                "active": session.active_persona
            })
            return

        # Serve static files for root or html requests
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        
        try:
            body = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if parsed.path == "/api/chat":
            user_input = body.get("message", "").strip()
            api_key = body.get("api_key") or os.environ.get("GEMINI_API_KEY")

            if not user_input:
                self._send_json({"error": "Message field is required"}, status=400)
                return

            try:
                res = session.send_message(user_input, api_key=api_key)
                self._send_json({
                    "status": "success",
                    "response": res["response"],
                    "model": res["model"],
                    "history_length": res["history_length"],
                    "estimated_tokens": res["estimated_tokens"],
                    "pruned_messages": res["pruned_messages"],
                    "timestamp": res["timestamp"],
                    "history": session.history
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        if parsed.path == "/api/clear":
            session.clear_history()
            self._send_json({
                "status": "success",
                "message": "Session history cleared.",
                "history_length": 0,
                "estimated_tokens": 0
            })
            return

        if parsed.path == "/api/persona":
            persona_key = body.get("persona")
            custom_prompt = body.get("system_prompt")
            
            if custom_prompt:
                session.system_prompt = custom_prompt
                session.active_persona = "custom"
                msg = "Custom system prompt set."
            elif persona_key:
                msg = session.set_persona(persona_key)
            else:
                self._send_json({"error": "Persona key or system_prompt required"}, status=400)
                return

            self._send_json({
                "status": "success",
                "message": msg,
                "active_persona": session.active_persona,
                "system_prompt": session.system_prompt
            })
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChatbotHTTPRequestHandler)
    print("=" * 65)
    print(f" 🚀 DECODELABS GENERATIVE AI PROJECT 1 - SERVER STARTED")
    print(f"    URL: http://localhost:{port}")
    print("=" * 65)
    print(" Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
