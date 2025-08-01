# song_state.py
import json
import os
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def session_file_path(session_id):
    return os.path.join(BASE_DIR, f"session_{session_id}.json")

class SongSession:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.data = {
            "current_step": "concept",
            "locked_sections": {},
            "used_rhyme_families": [],
            "song_structure": {},
            "style_direction": None,
            "concept": None,
            "tshirt_concept": None,
            "hook": None,
            "chorus": None,
            "verse_1": None,
            "pre_chorus": None,
            "verse_2": None,
            "bridge": None,
            "post_chorus": None
        }
        self.load()
        if not os.path.exists(session_file_path(self.session_id)):
            self.save()

    def load(self):
        filename = session_file_path(self.session_id)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.data = json.load(f)

    def save(self):
        filename = session_file_path(self.session_id)
        with open(filename, "w") as f:
            json.dump(self.data, f, indent=2)

    def update_step(self, step):
        self.data["current_step"] = step
        self.save()
        return step

    def lock_section(self, section_name, content):
        self.data["locked_sections"][section_name] = content
        self.save()
        return self.data["locked_sections"]

    def add_rhyme_family(self, family):
        if family not in self.data["used_rhyme_families"]:
            self.data["used_rhyme_families"].append(family)
            self.save()
        return self.data["used_rhyme_families"]

    def set_song_structure(self, structure):
        self.data["song_structure"] = structure
        self.save()
        return structure

    def get_song_overview(self):
        return {
            "session_id": self.session_id,
            **self.data
        }

    def set_style_direction(self, direction):
        self.data["style_direction"] = direction
        self.save()
        return direction

    def set_concept(self, concept):
        self.data["concept"] = concept
        self.save()
        return concept

    def set_tshirt_concept(self, tshirt_concept):
        self.data["tshirt_concept"] = tshirt_concept
        self.save()
        return tshirt_concept

    def set_hook(self, hook):
        self.data["hook"] = hook
        self.save()
        return hook

    def set_chorus(self, chorus):
        self.data["chorus"] = chorus
        self.save()
        return chorus

    def set_verse(self, verse_num, content):
        self.data[f"verse_{verse_num}"] = content
        self.save()
        return content

    def set_pre_chorus(self, content):
        self.data["pre_chorus"] = content
        self.save()
        return content

    def set_bridge(self, content):
        self.data["bridge"] = content
        self.save()
        return content

    def set_post_chorus(self, content):
        self.data["post_chorus"] = content
        self.save()
        return content

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.load_active_sessions()

    def load_active_sessions(self):
        for filename in os.listdir(BASE_DIR):
            if filename.startswith('session_') and filename.endswith('.json'):
                session_id = filename[8:-5]
                self.sessions[session_id] = SongSession(session_id)

    def create_session(self):
        session = SongSession()
        self.sessions[session.session_id] = session
        return session

    def get_session(self, session_id):
        if session_id in self.sessions:
            return self.sessions[session_id]
        session = SongSession(session_id)
        if os.path.exists(session_file_path(session_id)):
            self.sessions[session_id] = session
            return session
        return None