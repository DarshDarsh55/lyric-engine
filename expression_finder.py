import os
import json
import requests
from datamuse import Datamuse

class ExpressionFinder:
    def __init__(self, data_dir="rhyme_data"):
        """Initialize the ExpressionFinder with necessary resources"""
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.data_dir = data_dir
        self.cache_file = os.path.join(data_dir, "expression_cache.json")

        # Initialize Datamuse API client
        self.datamuse = Datamuse()

        # Initialize cache
        self.cache = self.load_cache()

    def load_cache(self):
        """Load the expression cache if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading {self.cache_file}, creating new cache")
                return {}
        return {}

    def save_cache(self, data):
        """Save data to the expression cache"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def find_expressions(self, word, max_expressions=30):
        """Find common expressions and idioms containing the word"""
        word = word.lower().strip()

        # Check cache
        cache_key = f"expressions_{word}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Use Datamuse to find related terms and phrases
        results = []

        # Get phrases where this word appears
        try:
            # Match word anywhere in phrase
            lc_results = self.datamuse.words(sp=f"*_{word}_*", max=max_expressions)
            results.extend([{
                'expression': r['word'].replace('_', ' '),
                'score': r.get('score', 0)
            } for r in lc_results if '_' in r['word']])

            # Word at start of phrase
            sp_results = self.datamuse.words(sp=f"{word}_*", max=max_expressions)
            results.extend([{
                'expression': r['word'].replace('_', ' '),
                'score': r.get('score', 0)
            } for r in sp_results if '_' in r['word']])

            # Word at end of phrase
            ep_results = self.datamuse.words(sp=f"*_{word}", max=max_expressions)
            results.extend([{
                'expression': r['word'].replace('_', ' '),
                'score': r.get('score', 0)
            } for r in ep_results if '_' in r['word']])
        except Exception as e:
            print(f"Error getting expressions for {word}: {e}")

        # Remove duplicates and sort by score
        unique_results = {}
        for item in results:
            expr = item['expression']
            if expr not in unique_results or item['score'] > unique_results[expr]['score']:
                unique_results[expr] = item

        # Convert back to list and sort by score
        sorted_results = sorted(list(unique_results.values()), key=lambda x: x['score'], reverse=True)

        # Cache the results
        self.cache[cache_key] = sorted_results
        self.save_cache(self.cache)

        return sorted_results

    def find_thematic_expressions(self, theme, max_expressions=30):
        """Find expressions related to a specific theme"""
        theme = theme.lower().strip()

        # Check cache
        cache_key = f"thematic_{theme}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Get words related to the theme
        try:
            related_words = self.datamuse.words(ml=theme, max=10)

            # Get expressions for each related word
            all_expressions = []
            for word_data in related_words:
                word = word_data['word']
                expressions = self.find_expressions(word, max_expressions=10)
                all_expressions.extend(expressions)

            # Remove duplicates and sort by score
            unique_results = {}
            for item in all_expressions:
                expr = item['expression']
                if expr not in unique_results or item['score'] > unique_results[expr]['score']:
                    unique_results[expr] = item

            # Convert back to list and sort by score
            sorted_results = sorted(list(unique_results.values()), key=lambda x: x['score'], reverse=True)

            # Cache the results
            self.cache[cache_key] = sorted_results
            self.save_cache(self.cache)

            return sorted_results
        except Exception as e:
            print(f"Error getting thematic expressions for {theme}: {e}")
            return []

    def find_rhyming_expressions(self, word, rhyme_engine, max_expressions=20):
        """Find expressions that end with words that rhyme with the given word"""
        word = word.lower().strip()

        # Check cache
        cache_key = f"rhyming_expr_{word}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Get perfect rhymes for the word
        rhymes = rhyme_engine.perfect_rhymes(word)
        rhyme_words = [r['word'] for r in rhymes[:20]]

        # Find expressions ending with these rhyming words
        results = []
        for rhyme in rhyme_words:
            try:
                # Find expressions ending with this rhyme
                expr_results = self.datamuse.words(sp=f"*_{rhyme}", max=10)

                for r in expr_results:
                    if '_' in r['word']:
                        results.append({
                            'expression': r['word'].replace('_', ' '),
                            'rhyme': rhyme,
                            'score': r.get('score', 0)
                        })

                if len(results) >= max_expressions:
                    break
            except Exception as e:
                print(f"Error getting rhyming expressions for {rhyme}: {e}")

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

        # Cache the results
        self.cache[cache_key] = sorted_results
        self.save_cache(self.cache)

        return sorted_results

    def find_idiomatic_phrases(self, word, max_phrases=20):
        """Find idiomatic phrases related to a word"""
        word = word.lower().strip()

        # Check cache
        cache_key = f"idioms_{word}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try to find idiomatic phrases using Datamuse
        results = []

        try:
            # Get phrases that are triggered by the word
            trig_results = self.datamuse.words(rel_trg=word, max=max_phrases)
            for r in trig_results:
                if ' ' in r['word']:  # Multi-word phrases only
                    results.append({
                        'phrase': r['word'],
                        'type': 'triggered',
                        'score': r.get('score', 0)
                    })

            # Get phrases that have similar meaning to the word
            ml_results = self.datamuse.words(ml=word, max=max_phrases)
            for r in ml_results:
                if ' ' in r['word'] and r['word'] not in [res['phrase'] for res in results]:
                    results.append({
                        'phrase': r['word'],
                        'type': 'meaning',
                        'score': r.get('score', 0)
                    })
        except Exception as e:
            print(f"Error getting idiomatic phrases for {word}: {e}")

        # Sort by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

        # Cache the results
        self.cache[cache_key] = sorted_results
        self.save_cache(self.cache)

        return sorted_results

    def word_in_expressions(self, word, max_results=20):
        """Split a word and find expressions with parts (like "adjust" -> "just")"""
        word = word.lower().strip()

        # Check cache
        cache_key = f"word_parts_{word}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        results = []

        # Simple string-based splitting approach
        for i in range(1, len(word)):
            part = word[i:]
            if len(part) >= 3:  # Only consider parts of reasonable length
                expressions = self.find_expressions(part, max_expressions=10)
                if expressions:
                    results.append({
                        'part': part,
                        'expressions': expressions[:5]  # Limit to top 5 per part
                    })

                    if len(results) >= max_results:
                        break

        # Cache the results
        self.cache[cache_key] = results
        self.save_cache(self.cache)

        return results