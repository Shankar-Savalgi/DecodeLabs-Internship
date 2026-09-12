import sys
import os
import json
from chatbot_engine import StatefulChatSession

def print_banner():
    print("=" * 65)
    print(" 🚀 DECODELABS INDUSTRIAL TRAINING KIT - BATCH 2026")
    print("    GENERATIVE AI PROJECT 1: CUSTOM AI CHATBOT WITH MEMORY")
    print("=" * 65)
    print(" Multi-turn contextual chat terminal with active in-memory session state.")
    print(" Type '/help' for options | '/export' to save transcript | '/quit' to exit.")
    print("-" * 65)

def main():
    session = StatefulChatSession()
    print_banner()
    print(f"Active Persona: {session.PERSONAS[session.active_persona]['name']}")
    print(f"System Prompt: {session.system_prompt[:80]}...")
    print("-" * 65)

    while True:
        try:
            user_input = input("\n🧑 You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit", "exit", "quit"]:
                print("\n👋 Thank you for using DecodeLabs AI Chatbot! Goodbye.")
                break

            if user_input.lower() == "/help":
                print("\n📌 Available Commands:")
                print("  /clear        - Reset the active in-memory conversation history")
                print("  /persona      - List available assistant personas and switch active persona")
                print("  /history      - Print raw in-memory conversation history array")
                print("  /export       - Save full conversation transcript to a Markdown file")
                print("  /stats        - Display session memory statistics and token usage")
                print("  /quit         - Exit the chatbot terminal")
                continue

            if user_input.lower() == "/clear":
                session.clear_history()
                print("\n🧹 In-memory conversation session history has been cleared.")
                continue

            if user_input.lower() == "/history":
                print(f"\n📜 In-Memory History Array ({len(session.history)} items):")
                print(json.dumps(session.history, indent=2))
                continue

            if user_input.lower() == "/stats":
                print(f"\n📊 Session Memory Stats:")
                print(f"  - Total Messages Stored: {len(session.history)}")
                print(f"  - Estimated Token Usage: {session.total_tokens_estimated} / {session.max_context_tokens}")
                print(f"  - Active Persona: {session.PERSONAS.get(session.active_persona, {}).get('name', 'Custom')}")
                continue

            if user_input.lower() == "/export":
                export_path = os.path.join(os.path.dirname(__file__), "chat_transcript.md")
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write(session.export_markdown())
                print(f"\n📄 Session transcript successfully exported to: {export_path}")
                continue

            if user_input.lower().startswith("/persona"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print("\n🎭 Available Personas:")
                    for key, val in session.PERSONAS.items():
                        current = " [ACTIVE]" if key == session.active_persona else ""
                        print(f"  - {key}: {val['name']}{current}")
                    print("\nUsage: /persona <persona_key>")
                else:
                    target_key = parts[1].strip()
                    msg = session.set_persona(target_key)
                    print(f"\n🎭 {msg}")
                continue

            # Process AI Message
            print("🤖 Assistant > Thinking...", end="\r")
            res = session.send_message(user_input)
            
            # Print Assistant Response
            print(f"🤖 Assistant [{res['timestamp']}] > {res['response']}")
            print(f"   └─ Memory: {res['history_length']} messages in history | Tokens: ~{res['estimated_tokens']}")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing message: {e}")

if __name__ == "__main__":
    main()
