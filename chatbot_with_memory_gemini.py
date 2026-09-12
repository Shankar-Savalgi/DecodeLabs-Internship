"""
Project 1: Custom AI Chatbot with Memory (Google Gemini - FREE)
------------------------------------------------------------------
A simple terminal chatbot that remembers the conversation
during a live session using an in-memory history list.

Setup:
1. pip install google-genai
2. Get a FREE API key from https://aistudio.google.com/apikey
3. Set it as an environment variable:
   - Windows:   set GEMINI_API_KEY=your-key-here
   - Mac/Linux: export GEMINI_API_KEY="your-key-here"
4. Run: python chatbot_with_memory_gemini.py
"""

import os
from google import genai
from google.genai import types

# ---- CONFIG ----
MAX_HISTORY_MESSAGES = 20  # sliding window limit (FIFO pruning)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# This is our "memory" — an in-memory list that stores the conversation
conversation_history = []


def trim_history(history, max_messages=MAX_HISTORY_MESSAGES):
    """Keep only the most recent messages to avoid hitting token limits."""
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def chat(user_input: str) -> str:
    global conversation_history

    # Step 1: validate input (avoid sending empty messages)
    if not user_input.strip():
        return "Please type something."

    # Step 2: append user message to history
    conversation_history.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )

    # Step 3: trim history if it's getting too long
    conversation_history = trim_history(conversation_history)

    # Step 4: send the whole history to the model
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history,
    )

    reply = response.text

    # Step 5: append the model's reply to history too
    conversation_history.append(
        types.Content(role="model", parts=[types.Part(text=reply)])
    )

    return reply


def main():
    print("Chatbot with Memory (Gemini) — type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            print("Bye!")
            break

        reply = chat(user_input)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
