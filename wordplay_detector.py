import re
import os
import json
from datamuse import Datamuse

class WordplayDetector:
    def __init__(self, rhyme_engine, data_dir="rhyme_data"):
        """Initialize the WordplayDetector with necessary resources"""
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.data_dir = data_dir
        self.cache_file = os.path.join(data_dir, "wordplay_cache.json")

        # Initialize Datamuse API client
        self.datamuse = Datamuse()

        # Store reference to rhyme engine
        self.rhyme_engine = rhyme_engine

        # Initialize cache
        self.cache = self.load_cache()

    def load_cache(self):
        """Load the wordplay cache if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading {self.cache_file}, creating new cache")
                return {}
        return {}

    def save_cache(self, data):
        """Save data to the wordplay cache"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def find_wordplay_opportunities(self, word):
        """Find potential wordplay opportunities for a word"""
        word = word.lower().strip()

        # Check cache
        cache_key = f"wordplay_{word}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        results = {
            'homophones': [],
            'contained_words': [],
            'contains_other_words': [],
            'spoonerisms': [],
            'double_meanings': []
        }

        # Find homophones
        homophones = self.rhyme_engine.homophones(word)
        if homophones:
            results['homophones'] = homophones

        # Find words contained within this word
        contained = []
        for i in range(len(word)-1):
            for j in range(i+2, len(word)+1):
                part = word[i:j]
                if len(part) >= 3 and part in self.rhyme_engine.cmu and part != word:
                    contained.append({
                        'word': part,
                        'start_index': i,
                        'end_index': j,
                        'frequency': self.rhyme_engine.get_word_frequency(part)
                    })

        # Sort by frequency and length
        contained.sort(key=lambda x: (x['frequency'], j-i), reverse=True)
        results['contained_words'] = contained[:10]  # Top 10 only

        # Find words that contain this word
        if len(word) >= 3:
            try:
                contains_query = f"*{word}*"
                api_results = self.datamuse.words(sp=contains_query, max=20)
                results['contains_other_words'] = [{
                    'word': r['word'],
                    'score': r.get('score', 0)
                } for r in api_results if r['word'] != word and len(r['word']) > len(word)]
            except Exception as e:
                print(f"Error finding words containing {word}: {e}")

        # Find potential spoonerisms
        if ' ' in word:
            parts = word.split(' ')
            if len(parts) == 2:
                # Simple spoonerism: swap first letters
                if len(parts[0]) > 0 and len(parts[1]) > 0:
                    spoonerism = parts[1][0] + parts[0][1:] + ' ' + parts[0][0] + parts[1][1:]
                    results['spoonerisms'].append(spoonerism)
        else:
            # Find common collocations and try to create spoonerisms
            try:
                collocations = self.datamuse.words(rel_bgb=word, max=10)  # Words that come before
                for collocation in collocations:
                    before_word = collocation['word']
                    if len(before_word) > 0 and len(word) > 0:
                        spoonerism = word[0] + before_word[1:] + ' ' + before_word[0] + word[1:]
                        results['spoonerisms'].append({
                            'original': f"{before_word} {word}",
                            'spoonerism': spoonerism
                        })
            except Exception as e:
                print(f"Error finding spoonerisms for {word}: {e}")

        # Find double meanings
        try:
            meanings = self.datamuse.words(ml=word, max=20)
            # Filter for words with multiple distinct meanings
            double_meaning_candidates = []

            for meaning in meanings:
                if meaning['word'] == word:
                    continue

                # Get all meanings for this word
                try:
                    word_meanings = self.datamuse.words(sp=meaning['word'], md='d', max=5)
                    if len(word_meanings) >= 2:
                        definitions = []
                        for wm in word_meanings:
                            if 'defs' in wm:
                                for definition in wm['defs']:
                                    definitions.append(definition)

                        if len(definitions) >= 2:
                            double_meaning_candidates.append({
                                'word': meaning['word'],
                                'definitions': definitions[:2],  # Just the top 2 definitions
                                'score': meaning.get('score', 0)
                            })
                except Exception as e:
                    continue

            results['double_meanings'] = sorted(double_meaning_candidates, key=lambda x: x['score'], reverse=True)[:5]
        except Exception as e:
            print(f"Error finding double meanings for {word}: {e}")

        # Cache the results
        self.cache[cache_key] = results
        self.save_cache(self.cache)

        return results

    def find_pun_opportunities(self, word, phrase_context=None):
        """Find potential pun opportunities for a word in context"""
        word = word.lower().strip()

        # Create cache key based on word and context
        cache_key = f"puns_{word}"
        if phrase_context:
            cache_key += f"_{phrase_context.replace(' ', '_')}"

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        results = {
            'sound_alike_puns': [],
            'meaning_puns': [],
            'compound_puns': []
        }

        # Find sound-alike words (near homophones)
        try:
            sound_alikes = self.datamuse.words(sl=word, max=20)

            for sa in sound_alikes:
                if sa['word'] == word:
                    continue

                # Only consider if the score is high enough (similar sound)
                if sa.get('score', 0) > 80:
                    # See if the sound-alike has a different meaning
                    meanings = self.datamuse.words(sp=sa['word'], md='d', max=1)

                    definition = None
                    if meanings and 'defs' in meanings[0]:
                        definition = meanings[0]['defs'][0]

                    results['sound_alike_puns'].append({
                        'original': word,
                        'sound_alike': sa['word'],
                        'score': sa.get('score', 0),
                        'definition': definition
                    })
        except Exception as e:
            print(f"Error finding sound-alike puns for {word}: {e}")

        # Find words with multiple meanings for meaning-based puns
        try:
            # Get the word's meanings
            word_meanings = self.datamuse.words(sp=word, md='d', max=5)

            if word_meanings and 'defs' in word_meanings[0]:
                definitions = word_meanings[0]['defs']

                # If the word has multiple definitions, it's a good pun candidate
                if len(definitions) >= 2:
                    results['meaning_puns'].append({
                        'word': word,
                        'definitions': definitions[:3]  # Just the top 3 definitions
                    })
        except Exception as e:
            print(f"Error finding meaning puns for {word}: {e}")

        # Find compound word puns (splitting a word into multiple words)
        if len(word) >= 5:  # Only consider longer words
            # Try various splits of the word
            for i in range(2, len(word)-1):
                part1 = word[:i]
                part2 = word[i:]

                # Check if both parts are valid words
                if part1 in self.rhyme_engine.cmu and part2 in self.rhyme_engine.cmu:
                    results['compound_puns'].append({
                        'original': word,
                        'compound': f"{part1} {part2}",
                        'part1': part1,
                        'part2': part2,
                        'part1_freq': self.rhyme_engine.get_word_frequency(part1),
                        'part2_freq': self.rhyme_engine.get_word_frequency(part2)
                    })

        # Cache the results
        self.cache[cache_key] = results
        self.save_cache(self.cache)

        return results