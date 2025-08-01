# app.py
from flask import Flask, request, jsonify
import os, json, logging, requests, pronouncing
import difflib
from datetime import datetime
from functools import wraps
from uuid import uuid4
from gpt_integration import gpt_generate, gpt_analyze_lyrics
from song_state import SessionManager
from references import upload_reference_file, get_all_references, delete_reference, ensure_references_dir
from rhyming_manager import get_ranked_slant_rhymes, update_usage_counts

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
RHYME_CACHE_FILE = "rhyme_cache.json"
INSTRUCTION_FILE = "instructions.json"
CONTRIBUTIONS_FILE = "user_contributions.json"
LYRICS_STORAGE_FILE = "lyrics_storage.json"

# JSON helper functions
def load_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {path}")
            return {}
    return {}

def save_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {str(e)}")
        return False

# Load necessary files on startup
rhyme_cache = load_json(RHYME_CACHE_FILE)
instructions = load_json(INSTRUCTION_FILE)
if not os.path.exists(CONTRIBUTIONS_FILE):
    save_json({"rock": {"for_artist_X": [], "billboard_top_100": []},
               "pop": {"for_artist_Y": []}}, CONTRIBUTIONS_FILE)
user_contributions = load_json(CONTRIBUTIONS_FILE)
if not os.path.exists(LYRICS_STORAGE_FILE):
    save_json({}, LYRICS_STORAGE_FILE)

session_manager = SessionManager()

# Admin auth
def check_auth(u, p):
    return u == ADMIN_USER and p == ADMIN_PASS

def authenticate():
    return jsonify({'error': 'Authentication required'}), 401

def requires_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return wrapper

def validate_session(session):
    required = ['current_step', 'locked_sections', 'used_rhyme_families',
                'song_structure', 'style_direction', 'concept', 'hook']
    missing = [k for k in required if k not in session.data]
    return (False, f"Missing keys {missing}") if missing else (True, 'valid')

@app.route('/')
def home():
    return "Lyric Writer RIME Stack API is running!"

@app.route('/createSession')
def create_session():
    s = session_manager.create_session()
    ok, msg = validate_session(s)
    return jsonify({'session_id': s.session_id,
                    'initial_state': s.get_song_overview(),
                    'validation': msg})

@app.route('/getSessionState')
def get_session_state():
    sid = request.args.get('session_id')
    if not sid:
        return jsonify({'error': 'session_id required'}), 400
    s = session_manager.get_session(sid)
    if not s:
        return jsonify({'error': 'Session not found'}), 404
    ok, msg = validate_session(s)
    state = s.get_song_overview()
    state['validation'] = msg
    return jsonify(state)

@app.route('/updateSession', methods=['POST'])
def update_session():
    data = request.json or {}
    sid = data.get('session_id')
    if not sid:
        return jsonify({'error': 'session_id required'}), 400
    s = session_manager.get_session(sid)
    if not s:
        return jsonify({'error': 'Session not found'}), 404
    action = data.get('action')
    if action == 'set_concept':
        s.set_concept(data.get('concept', ''))
    elif action == 'set_hook':
        s.set_hook(data.get('hook', ''))
    elif action == 'set_step':
        s.update_step(data.get('step'))
    else:
        return jsonify({'error': 'Unknown action'}), 400
    ok, msg = validate_session(s)
    state = s.get_song_overview()
    state['validation'] = msg
    return jsonify({'current_state': state, 'validation': msg})

@app.route('/getFilteredRhymes')
def get_filtered_rhymes():
    word = request.args.get('word', '').lower()
    if not word:
        return jsonify({'error': 'word required'}), 400
    rhymes = pronouncing.rhymes(word)
    return jsonify({'seedWord': word, 'finalRhymeList': rhymes})

@app.route('/get_slant_rhymes', methods=['POST'])
def get_slant_rhymes():
    data = request.json or {}
    word = data.get('word')
    if not word:
        return jsonify({'error': 'word required'}), 400
    try:
        resp = requests.get(f'https://rhymebrain.com/talk?function=getRhymes&word={word}&lang=en')
        resp.raise_for_status()
        js = resp.json()
        return jsonify({'word': word, 'rhymes': [e['word'] for e in js]})
    except Exception as e:
        logger.error(e)
        return jsonify({'error': str(e)}), 500

@app.route('/getRankedSlantRhymes')
def get_ranked_slant_rhymes_endpoint():
    word = request.args.get('word', '').lower()
    if not word:
        return jsonify({'error': 'word required'}), 400
    ranked = get_ranked_slant_rhymes(word)
    return jsonify({'word': word, 'ranked_slant_rhymes': ranked})

@app.route('/updateRhymeUsage', methods=['POST'])
def update_rhyme_usage():
    data = request.json or {}
    word = data.get('word')
    candidate = data.get('candidate')
    if not word or not candidate:
        return jsonify({'error': 'word and candidate required'}), 400
    usage = update_usage_counts(word, candidate)
    return jsonify({'word': word, 'candidate': candidate, 'new_usage': usage})

def calculate_similarity(text1, text2):
    """
    Calculate similarity between two text strings using difflib.
    Returns a similarity score between 0.0 and 1.0
    """
    # Convert to lowercase and strip whitespace to improve matching
    text1 = ' '.join(text1.lower().strip().split())
    text2 = ' '.join(text2.lower().strip().split())

    # Use difflib's SequenceMatcher to calculate similarity
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    logger.info(f"Similarity score: {similarity}")
    return similarity

# --- Endpoints for Lyrics Management ---

def load_lyrics_storage():
    storage = load_json(LYRICS_STORAGE_FILE)
    logger.info(f"Loaded lyrics storage with {len(storage)} entries")
    return storage

def save_lyrics_storage(data):
    success = save_json(data, LYRICS_STORAGE_FILE)
    if success:
        logger.info(f"Saved lyrics storage with {len(data)} entries")
    else:
        logger.error("Failed to save lyrics storage")
    return success

@app.route('/storeLyrics', methods=['POST'])
def store_lyrics():
    try:
        logger.info("storeLyrics: Received request")

        # Check if request is JSON
        if not request.is_json:
            logger.error("storeLyrics: Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.json

        # Validate request data
        if not data:
            logger.error("storeLyrics: No data in request")
            return jsonify({'error': 'No data provided'}), 400

        lyrics = data.get('lyrics')
        if not lyrics:
            logger.error("storeLyrics: No lyrics provided")
            return jsonify({'error': 'lyrics field is required'}), 400

        logger.info(f"storeLyrics: Lyrics received: {lyrics[:50]}...")  # Log first 50 chars

        # Load existing storage
        try:
            storage = load_lyrics_storage()
            logger.info(f"storeLyrics: Storage loaded. Current keys: {list(storage.keys())}")
        except Exception as e:
            logger.error(f"storeLyrics: Error loading storage: {e}")
            storage = {}

        # Check for duplicate or similar entries
        SIMILARITY_THRESHOLD = 0.85  # Adjust this threshold as needed
        similar_entries = []

        for entry_id, entry in storage.items():
            existing_lyrics = entry.get('lyrics', '')
            similarity = calculate_similarity(lyrics, existing_lyrics)

            if similarity > SIMILARITY_THRESHOLD:
                similar_entries.append({
                    'id': entry_id,
                    'similarity': similarity,
                    'entry': entry
                })

        if similar_entries:
            # Sort by similarity (highest first)
            similar_entries.sort(key=lambda x: x['similarity'], reverse=True)
            most_similar = similar_entries[0]

            logger.info(f"storeLyrics: Found similar entry with ID {most_similar['id']} (similarity: {most_similar['similarity']})")

            # If we have an exact or near-exact match, return the existing entry
            if most_similar['similarity'] > 0.95:
                logger.info(f"storeLyrics: Found nearly identical entry, returning existing entry")
                return jsonify({
                    'message': 'Similar lyrics already exist in the database',
                    'entry': most_similar['entry'],
                    'similarity': most_similar['similarity']
                })

            # For entries that are similar but not identical, we can either:
            # 1. Return the similar entry with a warning
            # 2. Store as a new entry but mark the relationship
            # 3. Merge the entries (if appropriate)

            # Let's implement option 1 for now:
            if most_similar['similarity'] > SIMILARITY_THRESHOLD:
                logger.info(f"storeLyrics: Found similar entry, returning with warning")
                return jsonify({
                    'message': 'Similar lyrics already exist in the database',
                    'entry': most_similar['entry'],
                    'similarity': most_similar['similarity'],
                    'action_needed': 'review',
                    'suggestion': 'These lyrics are similar to existing content. Please review before proceeding.'
                })

        # Analyze the lyrics
        try:
            # In production you'd use gpt_analyze_lyrics(lyrics)
            analysis = {
                "genre": "pop",
                "style": "commercial",
                "BPM": 120,
                "key": "C Major",
                "song_length": "3:30",
                "rule_sources": ["Rule 17", "Rule 23"]
            }
            logger.info("storeLyrics: Analysis complete")
        except Exception as e:
            logger.error(f"storeLyrics: Error during analysis: {e}")
            analysis = {"error": str(e)}

        # Create a new entry
        entry_id = str(uuid4())
        entry = {
            "id": entry_id,
            "lyrics": lyrics,
            "analysis": analysis,
            "is_reference": data.get('is_reference', False),
            "similar_to": [s['id'] for s in similar_entries] if similar_entries else [],
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"storeLyrics: Entry created, ID: {entry_id}")

        # Save the new entry
        storage[entry_id] = entry

        save_success = save_lyrics_storage(storage)
        if not save_success:
            logger.error("storeLyrics: Failed to save to storage")
            return jsonify({'error': 'Failed to save lyrics'}), 500

        logger.info("storeLyrics: Storage saved successfully")

        return jsonify({'message': 'Lyrics stored successfully', 'entry': entry})
    except Exception as e:
        logger.error(f"storeLyrics: Exception occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/getAllLyrics')
def get_all_lyrics():
    try:
        logger.info("getAllLyrics: Request received")

        storage = load_lyrics_storage()
        if not storage:
            logger.warning("getAllLyrics: Empty storage")
            return jsonify({'entries': []})

        entries = list(storage.values())
        logger.info(f"getAllLyrics: Returning {len(entries)} entries")

        return jsonify({'entries': entries})
    except Exception as e:
        logger.error(f"getAllLyrics: Exception occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/getLyricsByGenre')
def get_lyrics_by_genre():
    try:
        genre = request.args.get('genre')
        if not genre:
            logger.error("getLyricsByGenre: No genre provided")
            return jsonify({'error': 'genre query parameter is required'}), 400

        logger.info(f"getLyricsByGenre: Looking for genre: {genre}")

        storage = load_lyrics_storage()
        if not storage:
            logger.warning("getLyricsByGenre: Empty storage")
            return jsonify({'genre': genre, 'entries': []})

        filtered = []
        for entry_id, entry in storage.items():
            entry_genre = entry.get("analysis", {}).get("genre", "").lower()
            if entry_genre == genre.lower():
                filtered.append(entry)

        logger.info(f"getLyricsByGenre: Found {len(filtered)} entries for genre {genre}")

        return jsonify({'genre': genre, 'entries': filtered})
    except Exception as e:
        logger.error(f"getLyricsByGenre: Exception occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/getRuleSources')
def get_rule_sources():
    try:
        rule = request.args.get('rule')
        if not rule:
            logger.error("getRuleSources: No rule provided")
            return jsonify({'error': 'rule query parameter is required'}), 400

        logger.info(f"getRuleSources: Looking for rule: {rule}")

        storage = load_lyrics_storage()
        if not storage:
            logger.warning("getRuleSources: Empty storage")
            return jsonify({'rule': rule, 'entries': []})

        filtered = []
        for entry_id, entry in storage.items():
            rule_sources = entry.get("analysis", {}).get("rule_sources", [])
            if rule in rule_sources:
                filtered.append(entry)

        logger.info(f"getRuleSources: Found {len(filtered)} entries for rule {rule}")

        return jsonify({'rule': rule, 'entries': filtered})
    except Exception as e:
        logger.error(f"getRuleSources: Exception occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/deleteLyrics')
def delete_lyrics():
    try:
        entry_id = request.args.get('id')
        if not entry_id:
            logger.error("deleteLyrics: No id provided")
            return jsonify({'error': 'id query parameter is required'}), 400

        logger.info(f"deleteLyrics: Attempting to delete entry with ID: {entry_id}")

        storage = load_lyrics_storage()
        if entry_id not in storage:
            logger.warning(f"deleteLyrics: Entry with ID {entry_id} not found")
            return jsonify({'error': 'Entry not found'}), 404

        del storage[entry_id]

        save_success = save_lyrics_storage(storage)
        if not save_success:
            logger.error("deleteLyrics: Failed to save updated storage")
            return jsonify({'error': 'Failed to delete entry'}), 500

        logger.info(f"deleteLyrics: Successfully deleted entry with ID: {entry_id}")

        return jsonify({'message': f'Entry with ID {entry_id} deleted successfully'})
    except Exception as e:
        logger.error(f"deleteLyrics: Exception occurred: {e}")
        return jsonify({'error': str(e)}), 500

# --- Endpoints for Songwriting Process Steps ---

@app.route('/generateConcept', methods=['POST'])
def generate_concept():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()
        mode = data.get('mode', 'artist')
        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get('concept', {})
        instr = rules.get('instructions', '')
        details = rules.get(f'{mode}_mode', {}).get('details', [])
        prompt = instr + "\n" + "\n".join(details)
        concept = gpt_generate(prompt)
        session.set_concept(concept)
        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg
        return jsonify({'generated_concept': concept, 'session_state': state, 'session_id': session.session_id})
    except Exception as e:
        logger.error(e)
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/generateTshirtConcept', methods=['POST'])
def generate_tshirt_concept():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()
        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get('tshirt_concept', {})
        instr = rules.get('instructions', '')
        details = rules.get('artist_mode', {}).get('details', [])
        prompt = instr + "\n" + "\n".join(details)
        tshirt_concept = gpt_generate(prompt)
        session.set_tshirt_concept(tshirt_concept)
        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg
        return jsonify({'generated_tshirt_concept': tshirt_concept, 'session_state': state, 'session_id': session.session_id})
    except Exception as e:
        logger.error(e)
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/generateChorus', methods=['POST'])
def generate_chorus():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()
        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get('chorus', {})
        instr = rules.get('instructions', '')
        details = rules.get('details', [])
        prompt = instr + "\n" + "\n".join(details)
        chorus = gpt_generate(prompt)
        session.set_chorus(chorus)
        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg
        return jsonify({'generated_chorus': chorus, 'session_state': state, 'session_id': session.session_id})
    except Exception as e:
        logger.error(e)
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/generateVerse', methods=['POST'])
def generate_verse():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()
        verse_number = data.get('verse_number', 1)

        if not isinstance(verse_number, int) or verse_number < 1 or verse_number > 3:
            logger.error(f"generateVerse: Invalid verse number: {verse_number}")
            return jsonify({'error': 'verse_number must be 1, 2, or 3'}), 400

        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            logger.error(f"generateVerse: Session not found: {sid}")
            return jsonify({'error': 'Session not found'}), 404

        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get(f'verse_{verse_number}', {})
        instr = rules.get('instructions', '')
        details = rules.get('details', [])
        prompt = instr + "\n" + "\n".join(details)

        verse = gpt_generate(prompt)
        session.set_verse(verse_number, verse)

        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg

        return jsonify({
            'generated_verse': verse,
            'verse_number': verse_number,
            'session_state': state,
            'session_id': session.session_id
        })
    except Exception as e:
        logger.error(f"generateVerse: Exception occurred: {e}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/generatePreChorus', methods=['POST'])
def generate_pre_chorus():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()

        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            logger.error(f"generatePreChorus: Session not found: {sid}")
            return jsonify({'error': 'Session not found'}), 404

        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get('pre_chorus', {})
        instr = rules.get('instructions', '')
        details = rules.get('details', [])
        prompt = instr + "\n" + "\n".join(details)

        pre_chorus = gpt_generate(prompt)
        session.set_pre_chorus(pre_chorus)

        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg

        return jsonify({
            'generated_pre_chorus': pre_chorus,
            'session_state': state,
            'session_id': session.session_id
        })
    except Exception as e:
        logger.error(f"generatePreChorus: Exception occurred: {e}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/generateBridge', methods=['POST'])
def generate_bridge():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()

        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            logger.error(f"generateBridge: Session not found: {sid}")
            return jsonify({'error': 'Session not found'}), 404

        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get('bridge', {})
        instr = rules.get('instructions', '')
        details = rules.get('details', [])
        prompt = instr + "\n" + "\n".join(details)

        bridge = gpt_generate(prompt)
        session.set_bridge(bridge)

        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg

        return jsonify({
            'generated_bridge': bridge,
            'session_state': state,
            'session_id': session.session_id
        })
    except Exception as e:
        logger.error(f"generateBridge: Exception occurred: {e}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/generatePostChorus', methods=['POST'])
def generate_post_chorus():
    try:
        data = request.json or {}
        sid = data.get('session_id', '').strip()

        session = session_manager.create_session() if not sid else session_manager.get_session(sid)
        if not session:
            logger.error(f"generatePostChorus: Session not found: {sid}")
            return jsonify({'error': 'Session not found'}), 404

        steps = load_json(INSTRUCTION_FILE).get('steps', {})
        rules = steps.get('post_chorus', {})
        instr = rules.get('instructions', '')
        details = rules.get('details', [])
        prompt = instr + "\n" + "\n".join(details)

        post_chorus = gpt_generate(prompt)
        session.set_post_chorus(post_chorus)

        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg

        return jsonify({
            'generated_post_chorus': post_chorus,
            'session_state': state,
            'session_id': session.session_id
        })
    except Exception as e:
        logger.error(f"generatePostChorus: Exception occurred: {e}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/setSongStructure', methods=['POST'])
def set_song_structure():
    try:
        data = request.json or {}
        sid = data.get('session_id')
        structure = data.get('structure')

        if not sid:
            logger.error("setSongStructure: No session ID provided")
            return jsonify({'error': 'session_id required'}), 400

        if not structure or not isinstance(structure, dict):
            logger.error("setSongStructure: Invalid structure provided")
            return jsonify({'error': 'Valid structure object required'}), 400

        session = session_manager.get_session(sid)
        if not session:
            logger.error(f"setSongStructure: Session not found: {sid}")
            return jsonify({'error': 'Session not found'}), 404

        result = session.set_song_structure(structure)

        ok, msg = validate_session(session)
        state = session.get_song_overview()
        state['validation'] = msg

        return jsonify({
            'structure': result,
            'session_state': state
        })
    except Exception as e:
        logger.error(f"setSongStructure: Exception occurred: {e}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/getGPTInstructions')
def get_gpt_instructions():
    try:
        step = request.args.get('step', 'concept')
        instr = load_json(INSTRUCTION_FILE)

        if 'steps' not in instr or step not in instr['steps']:
            logger.warning(f"getGPTInstructions: Step not found: {step}")
            return jsonify({'error': f'Instructions for step {step} not found'}), 404

        step_instr = instr['steps'][step]
        global_instr = instr.get('global', {})

        return jsonify({
            'step': step,
            'instructions': step_instr,
            'global': global_instr
        })
    except Exception as e:
        logger.error(f"getGPTInstructions: Exception occurred: {e}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)