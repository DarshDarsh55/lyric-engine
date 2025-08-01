# rhyming_manager.py
import os
import json
import requests
import logging

SLANT_RHYME_CACHE_FILE = "slant_rhyme_cache.json"

def load_slant_cache():
    if os.path.exists(SLANT_RHYME_CACHE_FILE):
        with open(SLANT_RHYME_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_slant_cache(data):
    with open(SLANT_RHYME_CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def fetch_slant_rhymes_from_api(word):
    """Fetch slant/near rhymes from RhymeBrain for the given word."""
    try:
        resp = requests.get(f'https://rhymebrain.com/talk?function=getRhymes&word={word}&lang=en')
        resp.raise_for_status()
        js = resp.json()
        return [entry['word'] for entry in js]
    except Exception as e:
        logging.error(f"Error fetching rhymes for {word}: {e}")
        return []

def update_usage_counts(seed_word, candidate):
    """Update and record how many times a candidate word has been used."""
    cache = load_slant_cache()
    if seed_word not in cache:
        cache[seed_word] = {"candidates": {}}
    candidates = cache[seed_word]["candidates"]
    if candidate not in candidates:
        candidates[candidate] = {"usage": 0}
    candidates[candidate]["usage"] += 1
    cache[seed_word]["candidates"] = candidates
    save_slant_cache(cache)
    return candidates[candidate]["usage"]

def get_ranked_slant_rhymes(word, min_count=60):
    """
    Retrieve a ranked list of slant rhymes for a given word.
    Candidates with lower usage counts are ranked higher to promote variety.
    """
    cache = load_slant_cache()
    if word in cache:
        candidates = cache[word]["candidates"]
    else:
        candidates = {}

    # Fetch new candidates from the API and merge them with existing ones.
    fetched = fetch_slant_rhymes_from_api(word)
    for candidate in fetched:
        if candidate not in candidates:
            candidates[candidate] = {"usage": 0}

    cache[word] = {"candidates": candidates}
    save_slant_cache(cache)

    # Rank by usage (lowest usage gets ranked highest)
    ranked = sorted(candidates.items(), key=lambda item: item[1]["usage"])
    ranked_words = [word for word, data in ranked]

    # Expand the list if necessary (placeholder: duplicate until min_count reached)
    if len(ranked_words) < min_count:
        while len(ranked_words) < min_count:
            ranked_words.extend(ranked_words)
        ranked_words = ranked_words[:min_count]
    else:
        ranked_words = ranked_words[:min_count]
    return ranked_words