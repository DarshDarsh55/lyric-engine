import os
import json
import re
import string # Keep string if used for punctuation stripping etc. later
import requests
# import numpy as np # Removed, not used
import logging # Use logging instead of print
from collections import defaultdict

# NLTK imports
import nltk
from nltk.corpus import cmudict

# Other imports
import pronouncing
from metaphone import doublemetaphone
from datamuse import Datamuse
from jellyfish import soundex

# Word Frequency import (optional but recommended)
try:
    from wordfreq import word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    WORDFREQ_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RhymeEngine:
    def __init__(self, data_dir="rhyme_data", cache_file=None, expression_cache_file=None):
        """Initialize the RhymeEngine with necessary resources"""
        logger.info(f"Initializing RhymeEngine with data_dir: {data_dir}")
        self.data_dir = data_dir

        # Ensure data directory exists
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
                logger.info(f"Created data directory: {self.data_dir}")
            except OSError as e:
                logger.error(f"Failed to create data directory {self.data_dir}: {e}")
                # Decide if this is critical - maybe proceed but warn?

        self.cache_file = cache_file or os.path.join(data_dir, "rhyme_cache.json")
        self.expression_cache_file = expression_cache_file or os.path.join(data_dir, "expression_cache.json")

        # Initialize NLTK resources (with better error handling)
        self._ensure_nltk_data('corpora/cmudict', 'cmudict')
        self._ensure_nltk_data('corpora/wordnet', 'wordnet') # Ensure wordnet too

        # Load CMU dictionary
        try:
            self.cmu = cmudict.dict()
            if not self.cmu:
                 logger.warning("CMU dictionary loaded but is empty. Check NLTK data installation.")
            else:
                 logger.info(f"CMU Dictionary loaded successfully with {len(self.cmu)} entries.")
        except Exception as e:
            logger.error(f"Fatal error loading cmudict: {e}. Rhyme engine cannot function without it.", exc_info=True)
            raise RuntimeError("Failed to load CMU dictionary. Cannot initialize RhymeEngine.") from e

        # Initialize API client
        self.datamuse = Datamuse()
        logger.info("Datamuse client initialized.")

        # Load caches
        logger.info(f"Loading rhyme cache from: {self.cache_file}")
        self.rhyme_cache = self.load_cache(self.cache_file)
        logger.info(f"Loading expression cache from: {self.expression_cache_file}")
        self.expression_cache = self.load_cache(self.expression_cache_file)

        # Initialize word frequencies using wordfreq if available
        if WORDFREQ_AVAILABLE:
             logger.info("wordfreq library found. Using it for frequency ranking.")
        else:
             logger.warning("wordfreq library not found. Install for better ranking: pip install wordfreq")
             logger.warning("Using basic placeholder word frequencies.")
             self.placeholder_frequencies = defaultdict(lambda: 1) # Use if wordfreq is missing

        logger.info("RhymeEngine initialized successfully.")

    def _ensure_nltk_data(self, find_path, download_name):
        """Checks for NLTK data and downloads if missing."""
        try:
            nltk.data.find(find_path)
            logger.info(f"NLTK data '{download_name}' found.")
        except LookupError:
            logger.warning(f"NLTK data '{download_name}' not found. Downloading...")
            try:
                nltk.download(download_name, quiet=False) # Show download progress
                logger.info(f"NLTK data '{download_name}' downloaded successfully.")
                # Verify after download
                nltk.data.find(find_path)
            except Exception as e:
                logger.error(f"Failed to download NLTK data '{download_name}': {e}", exc_info=True)
                logger.error("Please ensure internet connectivity and sufficient permissions.")
                # Consider raising an error if data is critical, e.g., for cmudict
                if download_name == 'cmudict':
                     raise RuntimeError(f"Critical NLTK data '{download_name}' could not be downloaded.") from e

    # --- Caching and Helper Methods ---

    def cache_key(self, function_name, word, **kwargs):
        """Generate a cache key for a function call"""
        word_key = str(word).lower().strip().replace(' ', '_') # Ensure string, lowercase, strip, replace space
        key = f"{function_name}_{word_key}"
        if kwargs:
            key += "_" + "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        return key

    def load_cache(self, filename):
        """Load cache from a JSON file with UTF-8 encoding."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f: # Specify UTF-8
                    cache_data = json.load(f)
                    logger.info(f"Successfully loaded cache from {filename} with {len(cache_data)} keys.")
                    return cache_data
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON from {filename}. Starting with empty cache.")
                return {}
            except Exception as e:
                 logger.error(f"Error loading cache file {filename}: {e}", exc_info=True)
                 return {} # Return empty cache on other errors too
        else:
             logger.info(f"Cache file {filename} not found. Starting with empty cache.")
             return {}

    def save_cache(self, cache, filename):
        """Save cache to a JSON file with UTF-8 encoding."""
        try:
            # Ensure the directory exists before trying to save
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f: # Specify UTF-8
                json.dump(cache, f, indent=2) # Use indent=2 for consistency
                # logger.debug(f"Successfully saved cache to {filename}") # Debug level maybe
        except IOError as e:
            logger.error(f"Error saving cache to {filename}: {e}", exc_info=True)
        except TypeError as e:
            logger.error(f"Error serializing cache data for {filename}: {e}. Cache might be partially saved or corrupted.", exc_info=True)


    def get_phones(self, word):
        """Get the primary pronunciation (phonemes) for a word from CMU dict"""
        word_lower = str(word).lower().strip() # Ensure string
        if not word_lower: return None
        pronunciations = self.cmu.get(word_lower)
        if pronunciations:
            # logger.debug(f"Phones for '{word_lower}': {pronunciations[0]}")
            return pronunciations[0]
        else:
            # logger.debug(f"'{word_lower}' not found in CMU dictionary.")
            return None

    def get_word_frequency(self, word):
        """Gets word frequency using wordfreq library or placeholder."""
        word_lower = str(word).lower().strip() # Ensure string
        if not word_lower: return 0.0

        if WORDFREQ_AVAILABLE:
            # 'en' for English. wordfreq returns 0.0 if not found.
            # Higher value = more frequent.
            freq = word_frequency(word_lower, 'en', wordlist='best', minimum=0.0)
            # logger.debug(f"Frequency for '{word_lower}': {freq}")
            return freq # Return the raw frequency (higher is more common)
        else:
            # Fallback placeholder: Use length heuristic (shorter = more common)
            length_factor = max(1, 30 - len(word_lower) * 2)
            return float(self.placeholder_frequencies.get(word_lower, length_factor))

    def rank_by_commonality(self, words):
        """
        Rank a list of words by their frequency (most common first) using wordfreq if available.
        Returns a list of dicts [{'word': w, 'frequency': freq}] sorted by freq.
        """
        unique_words = set(filter(None, words)) # Filter out None or empty strings
        ranked_words = []
        for w in unique_words:
             freq = self.get_word_frequency(w) # Higher value = more frequent
             ranked_words.append({'word': w, 'frequency': freq})

        # Sort primarily by frequency (descending - higher freq is more common)
        # secondarily by word (ascending) for consistent order among ties.
        ranked_words.sort(key=lambda x: (-x['frequency'], x['word']))
        return ranked_words

    def count_syllables(self, word):
        """Count syllables in a word using pronouncing library or CMU or fallback heuristic."""
        word_lower = str(word).lower().strip()
        if not word_lower: return 0

        # Try pronouncing library first (often best)
        try:
            phones_list = pronouncing.phones_for_word(word_lower)
            if phones_list:
                syllable_count = pronouncing.syllable_count(phones_list[0])
                if syllable_count > 0:
                    # logger.debug(f"Syllables for '{word_lower}' (pronouncing): {syllable_count}")
                    return syllable_count
        except Exception as e:
            logger.warning(f"Pronouncing syllable count failed for '{word_lower}': {e}. Trying CMU.")

        # Fallback 1: Use CMU phones directly
        phones = self.get_phones(word_lower)
        if phones:
            count = len([p for p in phones if any(c.isdigit() for c in p)])
            # logger.debug(f"Syllables for '{word_lower}' (CMU direct): {count}")
            return max(1, count)

        # Fallback 2: Basic vowel counting heuristic (less accurate)
        logger.debug(f"Word '{word_lower}' not in CMU/pronouncing, using heuristic syllable count.")
        count = 0
        vowels = "aeiouy"
        if word_lower and word_lower[0] in vowels: count += 1
        for index in range(1, len(word_lower)):
            if word_lower[index] in vowels and word_lower[index-1] not in vowels: count += 1
        if word_lower.endswith("e") and count > 1:
             if len(word_lower) > 1 and word_lower[-2] not in vowels and not word_lower.endswith("le"): count -= 1
        if word_lower.endswith("le") and len(word_lower) > 2 and word_lower[-3] not in vowels:
             if count == 0: count = 1
             else: count +=1

        # Basic adjustments (can be expanded)
        diphthongs = ['oi', 'oy', 'ou', 'ow', 'au', 'aw', 'ei', 'ey', 'ea', 'eu', 'ew', 'ai', 'ay', 'ua', 'ui']
        for dip in diphthongs: count -= word_lower.count(dip)

        final_count = max(1, count)
        # logger.debug(f"Syllables for '{word_lower}' (heuristic): {final_count}")
        return final_count


    def is_perfect_rhyme(self, word1, word2):
        """Check if two words are perfect rhymes based on CMU dict"""
        w1_lower = str(word1).lower().strip()
        w2_lower = str(word2).lower().strip()
        if not w1_lower or not w2_lower or w1_lower == w2_lower:
            return False

        phones1 = self.get_phones(w1_lower)
        phones2 = self.get_phones(w2_lower)

        if not phones1 or not phones2:
            # logger.debug(f"Cannot check perfect rhyme: One or both words not in CMU ('{w1_lower}', '{w2_lower}')")
            return False

        # Find the index of the start of the last stressed syllable
        def find_last_stressed_vowel_index(phones):
            last_stressed_idx = -1
            for i in range(len(phones) - 1, -1, -1):
                if any(c in '12' for c in phones[i]): last_stressed_idx = i; break
            if last_stressed_idx == -1: # Fallback to last vowel if no stress marked
                 for i in range(len(phones) - 1, -1, -1):
                     if any(c.isdigit() for c in phones[i]): last_stressed_idx = i; break
            return last_stressed_idx

        idx1 = find_last_stressed_vowel_index(phones1)
        idx2 = find_last_stressed_vowel_index(phones2)

        if idx1 == -1 or idx2 == -1:
            # logger.warning(f"Could not find vowel index for perfect rhyme check ('{w1_lower}', '{w2_lower}')")
            return False

        # 1. Sounds from last stressed vowel onwards must match
        rhyming_part1 = phones1[idx1:]
        rhyming_part2 = phones2[idx2:]
        if rhyming_part1 != rhyming_part2: return False

        # 2. Sounds immediately preceding the last stressed vowel must differ
        preceding_sound1 = phones1[idx1 - 1] if idx1 > 0 else None
        preceding_sound2 = phones2[idx2 - 1] if idx2 > 0 else None
        if preceding_sound1 == preceding_sound2: return False # Includes case where both are None

        return True


    # --- Rhyme Finding Methods ---
    # (Keep structure, ensure consistency with helpers, logging, encoding)

    def perfect_rhymes(self, word):
        """Find perfect rhymes (last stressed syllable onwards matches, preceding sound differs)"""
        word_lower = str(word).lower().strip()
        if not word_lower: return []

        cache_key = self.cache_key("perfect", word_lower)
        if cache_key in self.rhyme_cache:
            return self.rhyme_cache[cache_key]

        rhymes = []
        # Use pronouncing library first
        try:
            if pronouncing.phones_for_word(word_lower):
                pronouncing_rhymes = pronouncing.rhymes(word_lower)
                if pronouncing_rhymes:
                    rhymes.extend(pronouncing_rhymes)
                    # logger.debug(f"Found {len(pronouncing_rhymes)} perfect rhymes for '{word_lower}' via pronouncing.")
            # else: logger.debug(f"'{word_lower}' not found by pronouncing library.")
        except Exception as e:
            logger.warning(f"Pronouncing library error for perfect rhymes of '{word_lower}': {e}")

        # Fallback/Supplement with manual check
        phones1 = self.get_phones(word_lower)
        if not phones1:
            logger.warning(f"Cannot find perfect rhymes for '{word_lower}': Not in CMU dict.")
            self.rhyme_cache[cache_key] = []
            self.save_cache(self.rhyme_cache, self.cache_file)
            return []

        existing_rhymes = set(rhymes)
        manual_rhymes_found = 0
        # Optimization: Iterate only over words with potentially matching endings? Hard. Full scan for now.
        for candidate in self.cmu:
            if candidate not in existing_rhymes:
                if self.is_perfect_rhyme(word_lower, candidate):
                    rhymes.append(candidate)
                    manual_rhymes_found += 1

        # if manual_rhymes_found > 0: logger.debug(f"Found {manual_rhymes_found} additional perfect rhymes manually.")

        ranked_rhymes = self.rank_by_commonality(rhymes)
        self.rhyme_cache[cache_key] = ranked_rhymes
        self.save_cache(self.rhyme_cache, self.cache_file)
        return ranked_rhymes

    # ... [Keep/Update other rhyme methods: consonance_rhymes, assonance_rhymes, etc.] ...
    # Make sure to apply consistent error handling, logging, caching, use of helpers.
    # Use word_lower = str(word).lower().strip() consistently.
    # Use set() for candidates where appropriate to avoid duplicates before ranking.

    # --- EXAMPLE UPDATE FOR CONSISTENCY ---
    def consonance_rhymes(self, word, min_match=1):
        """Find consonance rhymes (consonant sounds match, especially later ones)."""
        word_lower = str(word).lower().strip()
        if not word_lower: return []

        cache_key = self.cache_key("consonance", word_lower, min_match=min_match)
        if cache_key in self.rhyme_cache:
            return self.rhyme_cache[cache_key]

        candidates = set()
        phones = self.get_phones(word_lower)

        if not phones:
            logger.warning(f"Cannot find consonance for '{word_lower}': Not in CMU dict. Falling back to Datamuse.")
            # Datamuse fallback (using rel_cns or sl)
            try:
                results = self.datamuse.words(rel_cns=word_lower, max=50) # Consonant match
                if not results: results = self.datamuse.words(sl=word_lower, max=50) # Sounds like
                dm_candidates = [r['word'] for r in results if r['word'] != word_lower]
                ranked = self.rank_by_commonality(dm_candidates)
                self.rhyme_cache[cache_key] = ranked
                self.save_cache(self.rhyme_cache, self.cache_file)
                return ranked
            except Exception as e:
                logger.error(f"Datamuse fallback error for consonance '{word_lower}': {e}", exc_info=True)
                self.rhyme_cache[cache_key] = []
                self.save_cache(self.rhyme_cache, self.cache_file)
                return []

        word_consonants = [p for p in phones if not any(c.isdigit() for c in p)]
        if not word_consonants:
             # logger.debug(f"No consonants found in '{word_lower}'.")
             self.rhyme_cache[cache_key] = []
             self.save_cache(self.rhyme_cache, self.cache_file)
             return []

        # Find words with similar ending consonant patterns
        for candidate in self.cmu:
            if candidate == word_lower: continue
            cand_phones_list = self.cmu.get(candidate, [])
            for cand_phones in cand_phones_list:
                cand_consonants = [p for p in cand_phones if not any(c.isdigit() for c in p)]
                if not cand_consonants: continue

                len_w = len(word_consonants)
                len_c = len(cand_consonants)
                if len_w >= min_match and len_c >= min_match:
                    if word_consonants[-min_match:] == cand_consonants[-min_match:]:
                        candidates.add(candidate)
                        break # Found match for this pronunciation

        ranked_candidates = self.rank_by_commonality(list(candidates))
        self.rhyme_cache[cache_key] = ranked_candidates
        self.save_cache(self.rhyme_cache, self.cache_file)
        return ranked_candidates


    # --- FIX INDENTATION HERE ---
    # The original error was likely here. Ensure this line and the ones
    # before/after it have exactly the same indentation level as other methods.
    def feminine_para_rhymes(self, word):
        """Find feminine para rhymes (multi-syllable para rhymes)"""
        word_lower = str(word).lower().strip()
        if not word_lower: return []

        cache_key = self.cache_key("feminine_para", word_lower)
        if cache_key in self.rhyme_cache:
            return self.rhyme_cache[cache_key]

        # Get all para rhymes first
        para_results = self.para_rhymes(word_lower) # This is already ranked [{word: w, freq: f}]
        para_candidate_words = [r['word'] for r in para_results]

        # Filter for multi-syllabic words (2 or more syllables)
        feminine_para_words = []
        for rhyme_word in para_candidate_words:
            if self.count_syllables(rhyme_word) >= 2:
                feminine_para_words.append(rhyme_word)

        # Rank the filtered words again (might change order slightly if frequencies differ)
        ranked_feminine_para = self.rank_by_commonality(feminine_para_words)

        self.rhyme_cache[cache_key] = ranked_feminine_para
        self.save_cache(self.rhyme_cache, self.cache_file)
        return ranked_feminine_para

    # ... [Rest of the rhyme methods, ensure consistent indentation and updates] ...

    # --- Phrase Rhyming Methods ---
    def phrase_rhymes(self, phrase, max_results=20):
        """Find phrases rhyming with the input phrase, focusing on the last word."""
        phrase_lower = str(phrase).lower().strip()
        if not phrase_lower: return []

        cache_key = self.cache_key("phrase", phrase_lower, max_results=max_results)
        if cache_key in self.rhyme_cache:
            return self.rhyme_cache[cache_key]

        words = phrase_lower.split()
        if not words: return []

        last_word = words[-1]
        # Get rhymes (consider using 'intelligent' or combination for more options?)
        rhymes = self.perfect_rhymes(last_word) # List of {word: w, frequency: f}

        results = []
        seen_phrases = set([phrase_lower]) # Avoid returning the original phrase
        for r in rhymes:
             if len(results) >= max_results: break
             rhyme_word = r['word']
             new_phrase = ' '.join(words[:-1] + [rhyme_word])
             if new_phrase not in seen_phrases:
                 results.append({
                     'phrase': new_phrase,
                     'rhyme_word': rhyme_word,
                     'frequency': r['frequency'] # Frequency of the rhyme word
                 })
                 seen_phrases.add(new_phrase)

        # Results are already ranked by the rhyme word's frequency via perfect_rhymes
        self.rhyme_cache[cache_key] = results
        self.save_cache(self.rhyme_cache, self.cache_file)
        return results

    def phrase_similarity(self, phrase1, phrase2):
        """
        Calculate phonetic similarity between two phrases.
        NOTE: This is a placeholder for a complex task. Current implementation is basic.
        Consider using more advanced sequence alignment algorithms (like Needleman-Wunsch)
        on the concatenated phoneme lists for a better score.
        """
        logger.warning("phrase_similarity is using a very basic algorithm and may not be accurate.")
        phones1 = [p for w in str(phrase1).lower().strip().split() if (p := self.get_phones(w))]
        phones2 = [p for w in str(phrase2).lower().strip().split() if (p := self.get_phones(w))]

        if not phones1 or not phones2: return 0.0

        # Basic: Compare last word's ending phonemes (as in original) - limited value
        last_phones1 = phones1[-1]
        last_phones2 = phones2[-1]
        similarity_score = 0.0
        i = 1
        while i <= min(len(last_phones1), len(last_phones2)):
            if last_phones1[-i] == last_phones2[-i]:
                 similarity_score += 1.0
            else: break
            i += 1

        # Normalize crudely by total phonemes? Not ideal.
        # total_phonemes = sum(len(p) for plist in phones1 for p in plist) + sum(len(p) for plist in phones2 for p in plist)
        # return (similarity_score / total_phonemes) * 10 if total_phonemes > 0 else 0.0

        # Alternative: Return raw match count or length-normalized score for last word
        max_len = max(len(last_phones1), len(last_phones2))
        return similarity_score / max_len if max_len > 0 else 0.0


    # --- Master Functions ---

    def find_all_rhyme_types(self, word):
        """Find and categorize all implemented types of rhymes for a word."""
        word_lower = str(word).lower().strip()
        if not word_lower:
             return {rtype: [] for rtype in self.get_all_rhyme_type_names()}

        logger.info(f"Finding all rhyme types for: '{word_lower}'")
        results = {}
        # Call each method - they handle internal caching
        results['perfect'] = self.perfect_rhymes(word_lower)
        results['assonance'] = self.assonance_rhymes(word_lower)
        results['consonance'] = self.consonance_rhymes(word_lower)
        results['multi_syllable_2'] = self.multi_syllable_rhymes(word_lower, syllable_count=2)
        # results['multi_syllable_3'] = self.multi_syllable_rhymes(word_lower, syllable_count=3)
        results['homophones'] = self.homophones(word_lower)
        results['alliteration'] = self.alliteration(word_lower)
        # results['first_syllable'] = self.first_syllable_rhymes(word_lower) # Often overlaps Alliteration
        # results['final_syllable'] = self.final_syllable_rhymes(word_lower) # Often overlaps Perfect/Assonance
        results['metaphone'] = self.metaphone_rhymes(word_lower)
        results['soundex'] = self.soundex_rhymes(word_lower)
        results['para'] = self.para_rhymes(word_lower)
        results['feminine_para'] = self.feminine_para_rhymes(word_lower)
        results['light'] = self.light_rhymes(word_lower)
        results['family'] = self.family_rhymes(word_lower)
        results['broken'] = self.broken_rhymes(word_lower)
        results['reverse'] = self.reverse_rhymes(word_lower)
        results['weakened'] = self.weakened_rhymes(word_lower)
        results['trailing'] = self.trailing_rhymes(word_lower)
        results['full_consonance'] = self.full_consonance(word_lower) # Same as para
        results['full_assonance'] = self.full_assonance(word_lower)
        results['double_consonance'] = self.double_consonance(word_lower)
        results['double_assonance'] = self.double_assonance(word_lower)
        results['diminished'] = self.diminished_rhymes(word_lower)
        results['additive'] = self.additive_rhymes(word_lower)
        results['intelligent'] = self.intelligent_rhymes(word_lower)
        results['related'] = self.related_rhymes(word_lower)

        # Expressions are handled separately usually, but can include here if desired
        # results['expressions'] = self.find_expressions(word_lower)
        # results['rhyming_expressions'] = self.find_rhyming_expressions(word_lower)

        logger.info(f"Finished finding all rhyme types for: '{word_lower}'")
        return results

    def get_all_rhyme_type_names(self):
        """Returns a list of keys used in find_all_rhyme_types"""
        return [
            'perfect', 'assonance', 'consonance', 'multi_syllable_2', #'multi_syllable_3',
            'homophones', 'alliteration', #'first_syllable', 'final_syllable',
            'metaphone', 'soundex', 'para', 'feminine_para', 'light', 'family',
            'broken', 'reverse', 'weakened', 'trailing', 'full_consonance',
            'full_assonance', 'double_consonance', 'double_assonance',
            'diminished', 'additive', 'intelligent', 'related'
        ]

    # --- Improved Methods for Phrase Analysis ---

    def analyze_phrase_rhymes(self, phrase):
        """Comprehensive analysis of a phrase for rhyming patterns."""
        phrase_lower = str(phrase).lower().strip()
        words = phrase_lower.split()
        if not words: return {"error": "Empty phrase"}

        results = {
            "phrase": phrase_lower,
            "word_count": len(words),
            "last_word_rhymes": [],
            "internal_rhymes": [],
            "alliteration_patterns": []
        }

        # Analyze last word rhymes
        last_word = words[-1]
        results["last_word_rhymes"] = self.perfect_rhymes(last_word)[:10] # Top 10 perfect

        # Check for internal rhymes (using is_perfect_rhyme)
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                if self.is_perfect_rhyme(word1, word2):
                    results["internal_rhymes"].append({
                        "word1": word1, "word2": word2, "positions": [i, j]
                    })

        # Check for alliteration using PHONETICS
        alliteration_groups = defaultdict(list) # {initial_sound_tuple: [pos1, pos2]}
        for i, word in enumerate(words):
            phones = self.get_phones(word)
            if not phones: continue

            # Determine initial sound sequence (consonants or first vowel)
            initial_sounds = []
            for p in phones:
                is_vowel = any(c.isdigit() for c in p)
                if not initial_sounds and is_vowel: # First sound is vowel
                     initial_sounds.append(p)
                     break
                elif is_vowel: # Reached first vowel after consonants
                     break
                else: # Add consonant
                     initial_sounds.append(p)

            if initial_sounds: # Can be empty if word has no phones or only consonants? Unlikely.
                # Use tuple of sounds as key (stress markers are usually included here)
                # To ignore stress for broader match:
                # initial_key = tuple(re.sub(r'\d', '', p) for p in initial_sounds)
                initial_key = tuple(initial_sounds)
                alliteration_groups[initial_key].append(i)

        # Format results for groups with >= 2 words
        for sound_tuple, positions in alliteration_groups.items():
            if len(positions) >= 2:
                results["alliteration_patterns"].append({
                    "sound": " ".join(sound_tuple), # Join phonemes for display
                    "positions": positions
                })

        return results


# Example Usage Block (for running `python rhyme_engine.py` directly)
if __name__ == '__main__':
    print("Running RhymeEngine standalone test...")
    if not WORDFREQ_AVAILABLE:
         print("NOTE: wordfreq library not found. Frequency ranking will be basic.")
         print("      Install using: pip install wordfreq")

    # Create instance (will print logs from __init__)
    try:
         # Ensure data dir path is correct relative to execution location
         engine = RhymeEngine(data_dir="rhyme_data")

         test_words = ["apple", "example", "testing", "blue", "time", "house", "love"]
         test_phrases = ["a stitch in time saves nine", "the rain in spain stays mainly on the plain"]

         for test_word in test_words:
              print(f"\n--- Finding all rhymes for: {test_word} ---")
              all_rhymes = engine.find_all_rhyme_types(test_word)
              for rhyme_type, rhyme_results in all_rhymes.items():
                    print(f"\n{rhyme_type.replace('_', ' ').title()}:")
                    limit = 10
                    if isinstance(rhyme_results, list):
                        if not rhyme_results: print("  (None found)")
                        elif rhyme_results:
                            first_item = rhyme_results[0]
                            if isinstance(first_item, dict) and 'word' in first_item:
                                display_list = [r['word'] for r in rhyme_results[:limit]]
                                print(f"  {display_list}{'...' if len(rhyme_results) > limit else ''}")
                            elif isinstance(first_item, dict) and 'combination' in first_item: # Broken
                                display_list = [r['combination'] for r in rhyme_results[:limit]]
                                print(f"  {display_list}{'...' if len(rhyme_results) > limit else ''}")
                            elif isinstance(first_item, str): # Should be rare now
                                print(f"  {rhyme_results[:limit]}{'...' if len(rhyme_results) > limit else ''}")
                            else: print(f"  Unexpected list item format: {rhyme_results[:5]}...")
                    else: print(f"  Unexpected format (not a list): {type(rhyme_results)}")

         for test_phrase in test_phrases:
              print(f"\n--- Analyzing phrase: '{test_phrase}' ---")
              analysis = engine.analyze_phrase_rhymes(test_phrase)
              print(json.dumps(analysis, indent=2))

              print(f"\n--- Finding phrase rhymes for: '{test_phrase}' ---")
              phrase_rhyme_results = engine.phrase_rhymes(test_phrase)
              print(json.dumps(phrase_rhyme_results, indent=2))


    except RuntimeError as e:
         print(f"\nFATAL ERROR during RhymeEngine initialization: {e}")
    except Exception as e:
         print(f"\nAn unexpected error occurred during testing: {e}")
         import traceback
         traceback.print_exc()

    print("\n--- Standalone test complete ---")