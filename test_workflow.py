import requests
import json

# Replace with your actual PythonAnywhere domain
base_url = "https://marshthedarsh55.pythonanywhere.com"

def create_session():
    url = f"{base_url}/createSession"
    response = requests.get(url)
    data = response.json()
    print("Create Session Response:", json.dumps(data, indent=2))
    return data["session_id"]

def get_instructions(step="concept"):
    url = f"{base_url}/getGPTInstructions?step={step}"
    response = requests.get(url)
    data = response.json()
    print("Get Instructions:", json.dumps(data, indent=2))

def update_session(session_id, action, extra_data):
    url = f"{base_url}/updateSession"
    payload = {"session_id": session_id, "action": action}
    payload.update(extra_data)
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    print("Update Session Response:", json.dumps(data, indent=2))
    return data

def get_session_state(session_id):
    url = f"{base_url}/getSessionState?session_id={session_id}"
    response = requests.get(url)
    data = response.json()
    print("Get Session State:", json.dumps(data, indent=2))
    return data

# --- Full Test Workflow ---

# 1. Create a new session
session_id = create_session()

# 2. Get initial instructions
get_instructions("concept")

# 3. Update the session: set the step to "hook"
update_session(session_id, "set_step", {"step": "hook"})

# 4. Retrieve and verify the updated session state
get_session_state(session_id)

# 5. (Optional) Wait a minute or so, then retrieve the session state again to simulate resuming later
# get_session_state(session_id)