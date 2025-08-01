# references.py
import os
import json

REFERENCES_DIR = "reference_uploads"
REFERENCES_METADATA_FILE = "references_metadata.json"

def ensure_references_dir():
    if not os.path.exists(REFERENCES_DIR):
        os.makedirs(REFERENCES_DIR)

def load_references_metadata():
    if os.path.exists(REFERENCES_METADATA_FILE):
        with open(REFERENCES_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_references_metadata(metadata):
    with open(REFERENCES_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

def upload_reference_file(file, category, description):
    ensure_references_dir()
    metadata = load_references_metadata()
    filename = file.filename
    file_path = os.path.join(REFERENCES_DIR, filename)
    file.save(file_path)
    # For simplicity, using the filename as the reference ID; you could use a UUID instead.
    ref_id = filename
    metadata[ref_id] = {
        "filename": filename,
        "category": category,
        "description": description,
        "file_path": file_path
    }
    save_references_metadata(metadata)
    return metadata[ref_id]

def get_all_references():
    return load_references_metadata()

def delete_reference(ref_id):
    metadata = load_references_metadata()
    if ref_id in metadata:
        file_path = metadata[ref_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)
        del metadata[ref_id]
        save_references_metadata(metadata)
        return True
    return False