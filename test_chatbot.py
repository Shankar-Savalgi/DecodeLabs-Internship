import unittest
import os
import json
from chatbot_engine import StatefulChatSession

class TestStatefulChatSession(unittest.TestCase):

    def setUp(self):
        self.session = StatefulChatSession(max_history_messages=6, max_context_tokens=1000)

    def test_initial_state(self):
        self.assertEqual(len(self.session.history), 0)
        self.assertEqual(self.session.total_tokens_estimated, 0)
        self.assertEqual(self.session.active_persona, "decodelabs")

    def test_multi_turn_context_retention(self):
        res1 = self.session.send_message("My favorite programming language is Python.")
        self.assertEqual(len(self.session.history), 2) # 1 user + 1 model
        
        res2 = self.session.send_message("What did I say my favorite programming language was?")
        self.assertEqual(len(self.session.history), 4) # 2 user + 2 model
        self.assertIn("Python", res2["response"])

    def test_payload_structure(self):
        self.session.send_message("Hello AI!")
        payload = self.session.get_payload()
        self.assertEqual(payload[0]["role"], "system")
        self.assertEqual(payload[1]["role"], "user")
        self.assertEqual(payload[2]["role"], "model")

    def test_history_pruning(self):
        # Fill up history beyond max_history_messages (6 items limit = 3 turns max)
        for i in range(5):
            self.session.send_message(f"Turn number {i+1}")
        
        # History should be pruned to at most 6 items
        self.assertLessEqual(len(self.session.history), 6)

    def test_persona_switching(self):
        msg = self.session.set_persona("code_assistant")
        self.assertEqual(self.session.active_persona, "code_assistant")
        self.assertIn("Senior", msg)

    def test_export_formats(self):
        self.session.send_message("Testing export feature.")
        json_export = self.session.export_json()
        md_export = self.session.export_markdown()
        
        data = json.loads(json_export)
        self.assertEqual(len(data["history"]), 2)
        self.assertIn("# DecodeLabs AI Chatbot Session Transcript", md_export)

    def test_clear_history(self):
        self.session.send_message("Message to be cleared")
        self.session.clear_history()
        self.assertEqual(len(self.session.history), 0)

if __name__ == "__main__":
    unittest.main()
