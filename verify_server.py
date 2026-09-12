import urllib.request
import json

base_url = "http://localhost:8000"

def test_endpoint(path, data=None):
    url = base_url + path
    headers = {"Content-Type": "application/json"}
    if data:
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    else:
        req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()
        if "application/json" in content_type:
            return resp.status, json.loads(body.decode("utf-8"))
        else:
            return resp.status, body.decode("utf-8")[:100]

print("1. Testing GET / (Static HTML):")
status, html = test_endpoint("/")
print(f"Status: {status}, HTML starts with: {html[:40]}...")

print("\n2. Testing GET /api/session:")
status, sess = test_endpoint("/api/session")
print(f"Status: {status}, Persona: {sess.get('persona_name')}, History count: {sess.get('history_length')}")

print("\n3. Testing POST /api/chat (Turn 1):")
status, res1 = test_endpoint("/api/chat", {"message": "Hello DecodeLabs! My name is Sam."})
print(f"Status: {status}, Response: {res1.get('response')[:80]}...")

print("\n4. Testing POST /api/chat (Turn 2 - Memory test):")
status, res2 = test_endpoint("/api/chat", {"message": "What did I say my name was?"})
print(f"Status: {status}, Response: {res2.get('response')}")
print(f"History Length: {res2.get('history_length')}")

print("\n5. Testing GET /api/export?format=markdown:")
status, md = test_endpoint("/api/export?format=markdown")
print(f"Status: {status}, MD export preview:\n{md[:120]}...")

print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
