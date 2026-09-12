import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

class StatefulChatSession:
    """
    DecodeLabs Generative AI Project 1: Stateful Chatbot Engine with Memory.
    Maintains an active in-memory array to store conversation history,
    dynamically appending interactions to payload to preserve multi-turn context.
    """
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are DecodeLabs AI, a helpful, intelligent, and context-aware AI assistant. "
        "You remember previous context accurately and provide concise, professional, and well-structured answers."
    )

    PERSONAS = {
        "decodelabs": {
            "name": "DecodeLabs AI Tutor",
            "prompt": "You are DecodeLabs AI Tutor. You explain concepts step-by-step with practical examples, code snippets, and encouraging feedback."
        },
        "code_assistant": {
            "name": "Expert Senior Developer",
            "prompt": "You are a Senior Full-Stack Engineer and Architect. Provide production-ready, clean, well-commented code solutions and design advice."
        },
        "technical_writer": {
            "name": "Technical Documentation Specialist",
            "prompt": "You are a Technical Documentation Writer. Format output in clear Markdown with bullet points, code blocks, and structured headings."
        },
        "creative_partner": {
            "name": "Creative Brainstorming Partner",
            "prompt": "You are an imaginative and innovative AI creative consultant. Provide novel ideas, engaging storytelling, and out-of-the-box suggestions."
        }
    }

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_history_messages: int = 20, max_context_tokens: int = 4000):
        self.system_prompt: str = system_prompt
        self.max_history_messages: int = max_history_messages
        self.max_context_tokens: int = max_context_tokens
        self.history: List[Dict[str, Any]] = [] # Active in-memory array
        self.active_persona: str = "decodelabs"
        self.created_at: str = datetime.now().isoformat()
        self.total_tokens_estimated: int = 0

    def set_persona(self, persona_key: str) -> str:
        """Switch assistant persona and update system prompt."""
        if persona_key in self.PERSONAS:
            self.active_persona = persona_key
            self.system_prompt = self.PERSONAS[persona_key]["prompt"]
            return f"Persona updated to {self.PERSONAS[persona_key]['name']}"
        else:
            self.system_prompt = persona_key
            self.active_persona = "custom"
            return "Custom system prompt set."

    def clear_history(self) -> None:
        """Reset the in-memory conversation state."""
        self.history = []
        self.total_tokens_estimated = 0

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count (1 token ~= 4 chars)."""
        return max(1, len(text) // 4)

    def prune_history(self) -> int:
        """
        Logic loop to prune older messages when the history exceeds
        max history count or token limit, preserving recent context.
        """
        pruned_count = 0
        
        # 1. Enforce max message count (keep system prompt context)
        while len(self.history) > self.max_history_messages:
            self.history.pop(0)
            pruned_count += 1

        # 2. Enforce token limits
        current_tokens = sum(self._estimate_tokens(msg["content"]) for msg in self.history)
        while current_tokens > self.max_context_tokens and len(self.history) > 2:
            removed = self.history.pop(0)
            current_tokens -= self._estimate_tokens(removed["content"])
            pruned_count += 1

        self.total_tokens_estimated = current_tokens
        return pruned_count

    def get_payload(self) -> List[Dict[str, str]]:
        """
        Dynamically build the context payload by combining system instruction
        and active in-memory chat history.
        """
        payload = [{"role": "system", "content": self.system_prompt}]
        for msg in self.history:
            payload.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return payload

    def send_message(self, user_input: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Appends user input to history, queries LLM (or stateful fallback engine),
        appends model response to history array, and returns response metadata.
        """
        clean_input = user_input.strip()
        if not clean_input:
            raise ValueError("User message cannot be empty.")

        # 1. Append user input to active in-memory array
        user_msg = {
            "role": "user",
            "content": clean_input,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "tokens": self._estimate_tokens(clean_input)
        }
        self.history.append(user_msg)

        # 2. Auto-prune if history exceeds threshold
        pruned = self.prune_history()

        # 3. Generate response via Gemini API or Stateful Mock Engine
        api_key_to_use = api_key or os.environ.get("GEMINI_API_KEY")
        response_text = ""
        model_name = "Gemini 2.5 Flash" if api_key_to_use else "DecodeLabs Stateful Engine (Offline Fallback)"

        if api_key_to_use:
            response_text = self._call_gemini_api(api_key_to_use)
        else:
            response_text = self._generate_contextual_mock_response(clean_input)

        # 4. Append model response to active in-memory array
        model_msg = {
            "role": "model",
            "content": response_text,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "tokens": self._estimate_tokens(response_text)
        }
        self.history.append(model_msg)
        self.prune_history()

        return {
            "response": response_text,
            "model": model_name,
            "history_length": len(self.history),
            "estimated_tokens": self.total_tokens_estimated,
            "pruned_messages": pruned,
            "timestamp": model_msg["timestamp"]
        }

    def _call_gemini_api(self, api_key: str) -> str:
        """Call Google Gemini API using urllib/requests with conversation payload."""
        import urllib.request
        
        # Build contents structure for Gemini API REST Endpoint
        contents = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload_data = {
            "contents": contents,
            "system_instruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "No response generated.")
                return "Model returned empty response."
        except Exception as e:
            # Fallback if API call fails
            return f"(API Connection Note: {str(e)})\n\n[Fallback Response] " + self._generate_contextual_mock_response(self.history[-1]["content"])

    def _generate_contextual_mock_response(self, latest_input: str) -> str:
        """
        Intelligent in-memory stateful generator that demonstrates context memory
        by scanning previous turns in the history array!
        """
        lower = latest_input.lower()
        
        # Check if user asks about previous context (e.g. "what is my name", "what did I say", etc.)
        user_messages = [m["content"] for m in self.history if m["role"] == "user"]
        
        if "what is my name" in lower or "my name is" in lower:
            for msg in user_messages:
                if "my name is" in msg.lower() or "i am " in msg.lower():
                    words = msg.split()
                    return f"Based on our active session history, you previously mentioned your name! History array length: {len(self.history)} messages."
            return f"You haven't explicitly told me your name yet in this session. (Active history count: {len(self.history)} messages)."

        if "remember" in lower or "recall" in lower or "previous" in lower or "what did i" in lower:
            if len(user_messages) > 1:
                prev = user_messages[-2]
                return f"Yes! I am maintaining an active in-memory session array. Your previous message was: \"{prev}\". Total stored messages: {len(self.history)}."
            else:
                return "This is your first message in our current active session array."

        if "who are you" in lower or "help" in lower or "decodelabs" in lower:
            return (
                f"I am the DecodeLabs Project 1 Stateful Chatbot ({self.PERSONAS.get(self.active_persona, {}).get('name', 'Custom AI')}). "
                f"I maintain an active in-memory list array storing {len(self.history)} interactions in real time!"
            )

        if "hello" in lower or "hi" in lower or "hey" in lower:
            return f"Hello! Welcome to DecodeLabs Project 1 Chatbot with Memory. How can I assist you today? (Session active: {len(self.history)} items stored)"

        return (
            f"Thank you for your input! I have appended your message to my in-memory session array. "
            f"Currently holding {len(self.history)} turns in history under active context window."
        )

    def export_json(self) -> str:
        """Export session history as JSON string."""
        return json.dumps({
            "persona": self.active_persona,
            "system_prompt": self.system_prompt,
            "created_at": self.created_at,
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(self.history),
            "estimated_tokens": self.total_tokens_estimated,
            "history": self.history
        }, indent=2)

    def export_markdown(self) -> str:
        """Export session history as formatted Markdown document."""
        md = f"# DecodeLabs AI Chatbot Session Transcript\n\n"
        md += f"- **Exported At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"- **Persona**: {self.active_persona}\n"
        md += f"- **Total Messages**: {len(self.history)}\n"
        md += f"- **Estimated Tokens**: {self.total_tokens_estimated}\n\n"
        md += f"## System Prompt\n> {self.system_prompt}\n\n---\n\n"

        for i, msg in enumerate(self.history, 1):
            role_title = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
            md += f"### {i}. {role_title} `[{msg['timestamp']}]`\n\n{msg['content']}\n\n"

        return md

if __name__ == "__main__":
    # Quick self-test
    session = StatefulChatSession()
    res1 = session.send_message("Hi, my name is Alex!")
    print("Turn 1:", res1["response"])
    res2 = session.send_message("What did I say my name was?")
    print("Turn 2:", res2["response"])
    print("History Length:", len(session.history))
