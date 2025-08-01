# test_lyrics_endpoints.py
import requests
import json
import time

# Replace with your actual PythonAnywhere domain
base_url = "https://marshthedarsh55.pythonanywhere.com"

def test_store_lyrics():
    """Test the lyrics storage endpoint"""
    url = f"{base_url}/storeLyrics"

    # Sample lyrics for testing
    sample_lyrics = """
    Verse 1:
    Walking through the city lights
    Memories flash before my eyes
    Every corner has a story
    Of what was and what could be

    Chorus:
    This is my journey
    This is my road
    Every step I take
    Leads me back home
    """

    payload = {"lyrics": sample_lyrics}
    headers = {"Content-Type": "application/json"}

    print("Sending request to store lyrics...")
    response = requests.post(url, headers=headers, json=payload)

    print(f"Response Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
        return response.json().get('entry', {}).get('id')
    except:
        print("Response Text:")
        print(response.text)
        return None

def test_get_all_lyrics():
    """Test retrieving all lyrics"""
    url = f"{base_url}/getAllLyrics"

    print("\nRetrieving all lyrics...")
    response = requests.get(url)

    print(f"Response Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Found {len(data.get('entries', []))} lyrics entries")
        return data
    except:
        print("Response Text:")
        print(response.text)
        return None

def test_get_lyrics_by_genre():
    """Test retrieving lyrics by genre"""
    genre = "pop"  # The genre we're looking for
    url = f"{base_url}/getLyricsByGenre?genre={genre}"

    print(f"\nRetrieving lyrics for genre: {genre}")
    response = requests.get(url)

    print(f"Response Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Found {len(data.get('entries', []))} entries for genre {genre}")
        return data
    except:
        print("Response Text:")
        print(response.text)
        return None

def test_get_rule_sources():
    """Test retrieving lyrics by rule source"""
    rule = "Rule 17"  # The rule we're looking for
    url = f"{base_url}/getRuleSources?rule={rule}"

    print(f"\nRetrieving lyrics that use rule: {rule}")
    response = requests.get(url)

    print(f"Response Status Code: {response.status_code}")
    try:
        data = response.json()
        print(f"Found {len(data.get('entries', []))} entries using rule {rule}")
        return data
    except:
        print("Response Text:")
        print(response.text)
        return None

def test_delete_lyrics(entry_id):
    """Test deleting lyrics by ID"""
    if not entry_id:
        print("\nSkipping delete test - no entry ID provided")
        return False

    url = f"{base_url}/deleteLyrics?id={entry_id}"

    print(f"\nDeleting lyrics with ID: {entry_id}")
    response = requests.delete(url)

    print(f"Response Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except:
        print("Response Text:")
        print(response.text)
        return False

if __name__ == "__main__":
    print("=== Testing Lyrics API ===")

    # First store some lyrics
    entry_id = test_store_lyrics()

    # Give the server some time to process
    time.sleep(1)

    # Test retrieval endpoints
    test_get_all_lyrics()
    test_get_lyrics_by_genre()
    test_get_rule_sources()

    # Test deleting the entry we created
    if entry_id:
        print("\n=== Testing Delete ===")
        test_delete_lyrics(entry_id)

        # Verify deletion
        print("\n=== Verifying Deletion ===")
        all_lyrics = test_get_all_lyrics()