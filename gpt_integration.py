# gpt_integration.py
import os
import json
import requests
import openai
from rhyme_engine import RhymeEngine
from expression_finder import ExpressionFinder
from wordplay_detector import WordplayDetector

# Keep original simple functions for backward compatibility
def gpt_generate(prompt):
    """Generate text using GPT based on a prompt - original function maintained for compatibility"""
    # Placeholder for GPT output — replace with a real GPT API call later.
    return f"Generated concept based on the prompt: {prompt}"

def gpt_analyze_lyrics(lyrics):
    """Analyze lyrics using GPT - original function maintained for compatibility"""
    # Placeholder for GPT analysis — replace with real GPT prompt logic later.
    return {
        "genre": "pop",
        "style": "commercial",
        "BPM": 120,
        "key": "C Major",
        "song_length": "3:30",
        "rule_sources": ["Rule 17", "Rule 23"]
    }

class LyricGPTIntegration:
    """Integrated class that combines rhyming capabilities with broader lyric generation"""

    def __init__(self, api_key=None, data_dir="rhyme_data"):
        """Initialize with API key and data directory"""
        # Set up OpenAI API if key is provided
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key

        # Initialize rhyming components
        self.rhyme_engine = RhymeEngine(data_dir=data_dir)
        self.expression_finder = ExpressionFinder(data_dir=data_dir)
        self.wordplay_detector = WordplayDetector(self.rhyme_engine, data_dir=data_dir)

        # Cache files
        self.data_dir = data_dir
        self.rhyme_cache_file = os.path.join(data_dir, "rhyme_gpt_cache.json")
        self.verse_cache_file = os.path.join(data_dir, "verse_gpt_cache.json")
        self.rhyme_cache = self.load_cache(self.rhyme_cache_file)
        self.verse_cache = self.load_cache(self.verse_cache_file)

    def load_cache(self, cache_file):
        """Load a cache file if it exists"""
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading {cache_file}, creating new cache")
                return {}
        return {}

    def save_cache(self, data, cache_file):
        """Save data to a cache file"""
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    # =============== RHYME-FOCUSED METHODS ===============

    def get_rhyme_data(self, word, rhyme_types=None):
        """Get comprehensive rhyming data for a word"""
        # Default to common rhyme types if none specified
        if not rhyme_types:
            rhyme_types = ["perfect", "multi_syllable", "assonance", "consonance", "intelligent"]

        results = {}
        for rhyme_type in rhyme_types:
            method_name = f"{rhyme_type}_rhymes"
            if hasattr(self.rhyme_engine, method_name):
                method = getattr(self.rhyme_engine, method_name)
                # Handle special case for multi_syllable
                if rhyme_type == "multi_syllable":
                    results[rhyme_type] = method(word, 2)  # Default to 2 syllables
                else:
                    results[rhyme_type] = method(word)

        return results

    def get_wordplay_data(self, word):
        """Get wordplay opportunities for a word"""
        return self.wordplay_detector.find_wordplay_opportunities(word)

    def get_expression_data(self, word, theme=None):
        """Get expressions and idioms related to a word or theme"""
        data = {
            'word_expressions': self.expression_finder.find_expressions(word)
        }

        # If theme is provided, get thematic expressions too
        if theme:
            data['thematic_expressions'] = self.expression_finder.find_thematic_expressions(theme)

        # Get rhyming expressions
        data['rhyming_expressions'] = self.expression_finder.find_rhyming_expressions(
            word, self.rhyme_engine
        )

        return data

    def enhance_prompt_with_rhymes(self, prompt, theme=None, style=None, genre=None, rhyme_types=None):
        """Enhance a GPT prompt with rhyming and wordplay data"""
        # Extract theme if not provided
        if not theme:
            # Simple extraction - in a real system, use NLP
            words = prompt.lower().split()
            for candidate in ["love", "money", "success", "pain", "hope"]:
                if candidate in words:
                    theme = candidate
                    break

            # Default theme if none found
            if not theme:
                theme = "life"

        # Get rhyming options for theme and related words
        related_words = [theme]
        try:
            # Get related words to enrich our rhyme options
            related_results = self.rhyme_engine.datamuse.words(ml=theme, max=3)
            related_words.extend([r['word'] for r in related_results])
        except:
            pass

        # Gather rhyme data
        rhyme_data = {}
        wordplay_data = {}
        expression_data = {}

        for word in related_words:
            rhyme_data[word] = self.get_rhyme_data(word, rhyme_types)
            wordplay_data[word] = self.get_wordplay_data(word)
            expression_data[word] = self.get_expression_data(word)

        # Get thematic expressions if we have a theme
        if theme:
            thematic_expressions = self.expression_finder.find_thematic_expressions(theme)
        else:
            thematic_expressions = []

        # Format the enhancement
        enhancement = self._format_rhyme_enhancement(
            rhyme_data, wordplay_data, expression_data,
            thematic_expressions, theme, style, genre
        )

        # Return enhanced prompt
        return f"{prompt}\n\n{enhancement}"

    def _format_rhyme_enhancement(self, rhyme_data, wordplay_data, expression_data,
                                 thematic_expressions, theme, style, genre):
        """Format the rhyme data for prompt enhancement"""
        enhancement = "RHYMING AND WORDPLAY SUGGESTIONS:\n\n"

        # Add rhyme families
        enhancement += "Rhyme Families:\n"
        for word, data in rhyme_data.items():
            enhancement += f"Word: {word}\n"

            # Add perfect rhymes
            if 'perfect' in data and data['perfect']:
                perfect_str = ", ".join([r['word'] for r in data['perfect'][:7]])
                enhancement += f"  Perfect rhymes: {perfect_str}\n"

            # Add multi-syllable rhymes
            if 'multi_syllable' in data and data['multi_syllable']:
                multi_str = ", ".join([r['word'] for r in data['multi_syllable'][:5]])
                enhancement += f"  Multi-syllable rhymes: {multi_str}\n"

            # Add intelligent rhymes if available
            if 'intelligent' in data and data['intelligent']:
                intel_str = ", ".join([r['word'] for r in data['intelligent'][:5]])
                enhancement += f"  Clever/surprising rhymes: {intel_str}\n"

            enhancement += "\n"

        # Add wordplay opportunities
        enhancement += "Wordplay Opportunities:\n"
        for word, data in wordplay_data.items():
            # Skip if no interesting wordplay
            if (not data['homophones'] and not data['contained_words'] and
                not data['spoonerisms'] and not data['double_meanings']):
                continue

            enhancement += f"Word: {word}\n"

            # Add homophones
            if data['homophones']:
                homophones_str = ", ".join([h['word'] for h in data['homophones'][:5]])
                enhancement += f"  Homophones: {homophones_str}\n"

            # Add contained words
            if data['contained_words']:
                if isinstance(data['contained_words'][0], dict):
                    contained_str = ", ".join([c['word'] for c in data['contained_words'][:5]])
                else:
                    contained_str = ", ".join(data['contained_words'][:5])
                enhancement += f"  Contained words: {contained_str}\n"

            # Add double meanings
            if data['double_meanings']:
                enhancement += f"  Double meanings: {data['double_meanings'][0]['word']}\n"

            enhancement += "\n"

        # Add expressions and idioms
        enhancement += "Expressions and Idioms:\n"
        for word, data in expression_data.items():
            if 'word_expressions' in data and data['word_expressions']:
                expr_str = "; ".join([e['expression'] for e in data['word_expressions'][:5]])
                enhancement += f"  With '{word}': {expr_str}\n"

        # Add thematic expressions
        if thematic_expressions:
            thematic_str = "; ".join([e['expression'] for e in thematic_expressions[:5]])
            enhancement += f"  Thematic: {thematic_str}\n"

        # Add style-specific guidance
        if style or genre:
            enhancement += "\nSTYLE GUIDANCE:\n"

            if style:
                if style.lower() == "eminem":
                    enhancement += "- Use complex multi-syllabic rhymes\n"
                    enhancement += "- Include intense wordplay and double entendres\n"
                    enhancement += "- Use aggressive flow with frequent internal rhymes\n"
                elif style.lower() == "drake":
                    enhancement += "- Balance emotional content with clever wordplay\n"
                    enhancement += "- Use melodic flow with catchy hooks\n"
                    enhancement += "- Include introspective content with relatable themes\n"
                elif style.lower() == "taylor swift":
                    enhancement += "- Emphasize narrative storytelling\n"
                    enhancement += "- Use vivid imagery and emotional metaphors\n"
                    enhancement += "- Create memorable hooks with clean rhymes\n"
                else:
                    enhancement += f"- Match the flow and delivery style of {style}\n"
                    enhancement += "- Use appropriate rhyme density for the artist\n"

            if genre:
                if genre.lower() == "rap":
                    enhancement += "- Prioritize flow and rhythm over melody\n"
                    enhancement += "- Use higher rhyme density with internal rhymes\n"
                elif genre.lower() == "pop":
                    enhancement += "- Focus on catchy, memorable phrases\n"
                    enhancement += "- Use simple, clean rhyme schemes\n"
                    enhancement += "- Create universal, relatable content\n"
                elif genre.lower() == "country":
                    enhancement += "- Use storytelling with concrete imagery\n"
                    enhancement += "- Include place-based references\n"
                    enhancement += "- Use simpler rhyme schemes with clear cadence\n"

        return enhancement

    # =============== FULL LYRIC GENERATION METHODS ===============

    def generate_verse(self, theme, style=None, genre=None, length=16,
                      rhyme_types=None, technical_requirements=None):
        """Generate a complete verse using the rhyming engine and GPT"""
        # Prepare rhyme families
        rhyme_families = self.prepare_rhyme_families(theme)

        # Prepare wordplay opportunities
        wordplay_options = self.prepare_wordplay_options(theme)

        # Prepare cultural references
        cultural_references = self.prepare_cultural_references(theme)

        # Build chain of thought prompt
        prompt = self.build_chain_of_thought_prompt(
            theme=theme,
            style=style,
            genre=genre,
            length=length,
            rhyme_families=rhyme_families,
            wordplay_options=wordplay_options,
            cultural_references=cultural_references,
            technical_requirements=technical_requirements
        )

        # Get response from GPT
        response = self.get_gpt_response(prompt, cache_file=self.verse_cache_file)

        # Parse the response
        parsed_response = self.parse_gpt_response(response)

        return parsed_response

    def prepare_rhyme_families(self, theme):
        """Prepare rhyme families for a theme"""
        # Get related words to the theme
        try:
            related_words = []
            related_results = self.rhyme_engine.datamuse.words(ml=theme, max=5)
            related_words = [r['word'] for r in related_results]

            # Add the theme itself
            related_words = [theme] + related_words

            # Prepare rhyme families
            families = []
            for word in related_words:
                # Get perfect rhymes
                perfect_rhymes = self.rhyme_engine.perfect_rhymes(word)

                # Get multi-syllabic rhymes
                multi_rhymes = self.rhyme_engine.multi_syllable_rhymes(word, 2)

                # Only include if we have enough rhymes
                if perfect_rhymes and len(perfect_rhymes) >= 3:
                    families.append({
                        'seed': word,
                        'perfect_rhymes': perfect_rhymes[:7],
                        'multi_syllable_rhymes': multi_rhymes[:7] if multi_rhymes else []
                    })

                # Limit to 4 families
                if len(families) >= 4:
                    break

            return families
        except Exception as e:
            print(f"Error preparing rhyme families: {e}")
            return []

    def prepare_wordplay_options(self, theme):
        """Prepare wordplay options for a theme"""
        wordplay_options = []

        try:
            # Get related words to the theme
            related_results = self.rhyme_engine.datamuse.words(ml=theme, max=10)
            related_words = [theme] + [r['word'] for r in related_results]

            # Get wordplay for each related word
            for word in related_words[:5]:  # Limit to 5 words
                wordplay = self.wordplay_detector.find_wordplay_opportunities(word)

                # Add to options if we have interesting results
                if wordplay['homophones'] or wordplay['contained_words'] or wordplay['spoonerisms']:
                    wordplay_options.append({
                        'word': word,
                        'homophones': wordplay['homophones'][:3] if wordplay['homophones'] else [],
                        'contained_words': wordplay['contained_words'][:3] if wordplay['contained_words'] else [],
                        'spoonerisms': wordplay['spoonerisms'][:2] if wordplay['spoonerisms'] else []
                    })

                # Also check for pun opportunities
                puns = self.wordplay_detector.find_pun_opportunities(word)
                if puns['sound_alike_puns'] or puns['meaning_puns']:
                    pun_option = {
                        'word': word,
                        'sound_alike_puns': puns['sound_alike_puns'][:3] if puns['sound_alike_puns'] else [],
                        'meaning_puns': puns['meaning_puns'][:2] if puns['meaning_puns'] else []
                    }

                    # Add to options if not already included
                    if word not in [opt['word'] for opt in wordplay_options]:
                        wordplay_options.append(pun_option)
                    else:
                        # Merge with existing entry
                        for opt in wordplay_options:
                            if opt['word'] == word:
                                opt.update(pun_option)
                                break

            return wordplay_options
        except Exception as e:
            print(f"Error preparing wordplay options: {e}")
            return []

    def prepare_cultural_references(self, theme):
        """Prepare cultural references for a theme"""
        try:
            # Get related words to the theme
            related_results = self.rhyme_engine.datamuse.words(ml=theme, max=10)
            related_words = [r['word'] for r in related_results]

            # For each related word, find expressions
            expressions = []
            for word in related_words[:3]:  # Limit to 3 words
                word_expressions = self.expression_finder.find_expressions(word, max_expressions=5)
                expressions.extend(word_expressions)

            # Convert to cultural references format
            references = []
            for expr in expressions[:10]:  # Limit to 10 references
                references.append({
                    'expression': expr['expression'],
                    'score': expr.get('score', 0)
                })

            return references
        except Exception as e:
            print(f"Error preparing cultural references: {e}")
            return []

    def build_chain_of_thought_prompt(self, theme, style=None, genre=None, length=16,
                                      rhyme_families=None, wordplay_options=None,
                                      cultural_references=None, technical_requirements=None):
        """Build a detailed chain-of-thought prompt for lyric generation"""
        # Default technical requirements if none provided
        if not technical_requirements:
            technical_requirements = self.get_default_requirements(style, genre, length)

        # Build prompt base
        prompt = f"""You are a professional lyricist writing {'in the style of ' + style if style else 'a ' + genre + ' song' if genre else 'lyrics'} about {theme}.

Follow this step-by-step thought process to create technically impressive lyrics:

STEP 1: RHYME SCHEME PLANNING
"""

        # Add rhyme families
        if rhyme_families:
            for i, family in enumerate(rhyme_families):
                prompt += f"Rhyme Family {i+1} - Seed: {family['seed']}\n"

                # Add perfect rhymes
                perfect_rhymes_str = ", ".join([r['word'] for r in family['perfect_rhymes'][:7]])
                prompt += f"  Perfect rhymes: {perfect_rhymes_str}\n"

                # Add multi-syllable rhymes if available
                if family['multi_syllable_rhymes']:
                    multi_rhymes_str = ", ".join([r['word'] for r in family['multi_syllable_rhymes'][:7]])
                    prompt += f"  Multi-syllabic rhymes: {multi_rhymes_str}\n"

                prompt += "\n"

        # Add wordplay options
        prompt += "STEP 2: WORDPLAY SELECTION\n"

        if wordplay_options:
            for option in wordplay_options:
                prompt += f"Word: {option['word']}\n"

                if 'homophones' in option and option['homophones']:
                    homophones_str = ", ".join([h['word'] for h in option['homophones'][:3]])
                    prompt += f"  Homophones: {homophones_str}\n"

                if 'contained_words' in option and option['contained_words']:
                    contained_str = ", ".join([c['word'] for c in option['contained_words'][:3]])
                    prompt += f"  Contained words: {contained_str}\n"

                if 'sound_alike_puns' in option and option['sound_alike_puns']:
                    puns_str = ", ".join([p['sound_alike'] for p in option['sound_alike_puns'][:3]])
                    prompt += f"  Sound-alike puns: {puns_str}\n"

                if 'meaning_puns' in option and option['meaning_puns']:
                    prompt += f"  Multiple meanings: {option['meaning_puns'][0]['word']} - "
                    for definition in option['meaning_puns'][0]['definitions'][:2]:
                        prompt += f"{definition}, "
                    prompt = prompt.rstrip(", ") + "\n"

                prompt += "\n"
        else:
            prompt += "Use wordplay techniques like homophones, double meanings, and contained words.\n\n"

        # Add cultural references
        prompt += "STEP 3: CULTURAL REFERENCE INTEGRATION\n"

        if cultural_references:
            for ref in cultural_references[:7]:  # Limit to 7 references
                prompt += f"- \"{ref['expression']}\"\n"
        else:
            prompt += "Incorporate cultural references and common expressions related to the theme.\n"

        prompt += "\n"

        # Add style-specific instructions
        prompt += "STEP 4: STYLE CHARACTERISTICS\n"

        if style:
            if style.lower() == "eminem":
                prompt += """- Fast flow with complex rhyme patterns
- Heavy use of internal rhymes and assonance
- Multiple syllable rhymes (3+ syllables when possible)
- Witty wordplay and double entendres
- Dark humor and self-deprecation
- Aggressive delivery and battle rap elements
- Narrative structure that builds to punchlines\n\n"""
            elif style.lower() == "missy elliott":
                prompt += """- Innovative and futuristic flow
- Playful, confident delivery
- Clever wordplay focused on empowerment
- Catchy, memorable phrases
- Unexpected rhythm switches
- Call and response elements
- Frequent use of slang and invented language\n\n"""
            elif style.lower() == "drake":
                prompt += """- Melodic flow with sing-song delivery
- Introspective and confessional content
- Mix of braggadocio and vulnerability
- References to luxury and success
- Relationship-focused themes
- Use of the "Drake formula" (setup, punchline, reflection)\n\n"""
            elif style.lower() == "taylor swift":
                prompt += """- Strong narrative storytelling
- Vivid imagery and details
- Emotional authenticity
- Mix of vulnerability and strength
- Clear, clean rhymes
- Memorable, quotable lines
- Relatable personal experiences\n\n"""
            else:
                prompt += f"""- Capture the unique voice and flow of {style}
- Incorporate signature themes and content
- Match the typical rhyme density and pattern
- Reflect characteristic attitudes and perspectives
- Include signature phrases or references\n\n"""
        elif genre:
            if genre.lower() == "rap":
                prompt += """- Higher rhyme density with internal rhymes
- Focus on flow and rhythm
- Include metaphors and similes
- Use wordplay and double meanings
- Create vivid imagery
- Maintain consistent cadence\n\n"""
            elif genre.lower() == "pop":
                prompt += """- Catchy, memorable hooks
- Universal, relatable themes
- Simple, clean rhyme schemes
- Focus on emotional connection
- Short, repeatable phrases
- Accessible vocabulary\n\n"""
            elif genre.lower() == "country":
                prompt += """- Storytelling with narrative arc
- References to place and identity
- Concrete imagery over abstract concepts
- Traditional values and experiences
- Simpler rhyme schemes with clear cadence
- Emotional authenticity\n\n"""
            elif genre.lower() == "r&b":
                prompt += """- Smooth, melodic flow
- Emotional and sensual themes
- Soulful delivery with vocal runs in mind
- Relationship-focused content
- Mix of vulnerability and confidence
- Room for vocal expression\n\n"""
            else:
                prompt += """- Match the typical conventions of the genre
- Use appropriate vocabulary and themes
- Balance originality with genre expectations
- Structure lyrics for the typical song format\n\n"""
        else:
            prompt += """- Develop a clear voice and perspective
- Use internal consistency in tone and approach
- Balance technical skill with emotional impact
- Create memorable phrases and hooks
- Build toward meaningful conclusions\n\n"""

        # Add technical requirements
        prompt += "STEP 5: TECHNICAL REQUIREMENTS\n"
        for req, details in technical_requirements.items():
            prompt += f"- {req}: {details}\n"

        prompt += f"""
STEP 6: VERSE CONSTRUCTION
Now, construct your verse using the above elements. For each line:
1. Decide which rhyme family to use
2. Select a metaphor or cultural reference to incorporate
3. Look for opportunities to include wordplay
4. Ensure the flow matches the {'style of ' + style if style else genre + ' genre' if genre else 'desired tone'}
5. Build toward impactful punchlines

First, show your reasoning for each line (explain which techniques you're using), then provide the complete verse at the end.

Begin your reasoning now:"""

        return prompt

    def get_default_requirements(self, style=None, genre=None, length=16):
        """Get default technical requirements based on style or genre"""
        requirements = {
            "Verse Length": f"{length} lines",
            "Rhyme Pattern": "Use at least 2 different rhyme families"
        }

        # Style-specific requirements
        if style:
            if style.lower() == "eminem":
                requirements.update({
                    "Wordplay": "Include at least 3 instances of wordplay",
                    "Internal Rhymes": "At least 2 internal rhymes per 4 bars",
                    "Syllable Count": "10-15 syllables per line",
                    "Flow": "Fast-paced with technical complexity"
                })
            elif style.lower() == "drake":
                requirements.update({
                    "Wordplay": "Include at least A2 instances of wordplay",
                    "Tone Shifts": "Mix confidence with vulnerability",
                    "Syllable Count": "8-12 syllables per line",
                    "Flow": "Melodic with rhythmic switches"
                })
            elif style.lower() == "taylor swift":
                requirements.update({
                    "Imagery": "Use vivid, specific details",
                    "Narrative": "Tell a clear story with emotional arc",
                    "Syllable Count": "8-10 syllables per line",
                    "Flow": "Natural, conversational with clear emphasis"
                })
            else:
                requirements.update({
                    "Wordplay": "Include at least 2 instances of wordplay",
                    "Cultural References": "Incorporate at least 1 cultural reference",
                    "Flow": "Match the typical cadence of the artist"
                })
        # Genre-specific requirements
        elif genre:
            if genre.lower() == "rap":
                requirements.update({
                    "Wordplay": "Include at least 3 instances of wordplay",
                    "Rhyme Density": "High with internal rhymes",
                    "Flow": "Prioritize rhythm and cadence"
                })
            elif genre.lower() == "pop":
                requirements.update({
                    "Hook Potential": "Include at least one memorable phrase",
                    "Universality": "Use relatable themes and emotions",
                    "Simplicity": "Clear, accessible language"
                })
            elif genre.lower() == "country":
                requirements.update({
                    "Storytelling": "Clear narrative with concrete details",
                    "Authenticity": "Genuine emotion and experience",
                    "Imagery": "Visual, place-based descriptions"
                })
            else:
                requirements.update({
                    "Genre Conventions": f"Follow standard {genre} lyrical approaches",
                    "Authenticity": "Match the emotional tone of the genre"
                })
        # Default requirements
        else:
            requirements.update({
                "Wordplay": "Include at least 2 instances of wordplay",
                "Cultural References": "Incorporate at least 1 cultural reference",
                "Flow": "Maintain a consistent rhythm with variations"
            })

        return requirements

    def get_gpt_response(self, prompt, cache_file=None):
        """Get a response from GPT with caching"""
        # Determine which cache to use
        cache = self.rhyme_cache if cache_file == self.rhyme_cache_file else self.verse_cache

        # Check cache
        cache_key = prompt[:100] + prompt[-100:]  # Use start and end as cache key
        if cache_key in cache:
            return cache[cache_key]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Or "gpt-3.5-turbo" if preferred
                messages=[
                    {"role": "system", "content": "You are a professional lyricist with expertise in rhyming, wordplay, and different styles."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )

            result = response['choices'][0]['message']['content']

            # Cache result
            cache[cache_key] = result
            self.save_cache(cache, cache_file or self.verse_cache_file)

            return result
        except Exception as e:
            print(f"Error getting GPT response: {e}")
            return "Error generating lyrics. Please try again."

    def parse_gpt_response(self, response):
        """Parse GPT response to extract verse and reasoning"""
        try:
            # Split into lines
            lines = response.split('\n')

            # Look for verse markers
            verse_start = None
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("final verse") or \
                   line.strip().lower().startswith("complete verse") or \
                   line.strip().lower() == "verse:":
                    verse_start = i + 1
                    break

            # If no marker found, estimate verse location
            if verse_start is None:
                # Look for a blank line followed by content that doesn't start with "line"
                for i in range(len(lines)-1):
                    if not lines[i].strip() and lines[i+1].strip() and not lines[i+1].lower().startswith("line"):
                        verse_start = i + 1
                        break

                # If still not found, assume it's in the last third
                if verse_start is None:
                    verse_start = max(len(lines) - len(lines) // 3, 0)

            # Extract verse and reasoning
            reasoning = '\n'.join(lines[:verse_start]).strip()
            verse = '\n'.join(lines[verse_start:]).strip()

            return {
                'verse': verse,
                'reasoning': reasoning
            }
        except Exception as e:
            print(f"Error parsing GPT response: {e}")
            return {
                'verse': response,
                'reasoning': "Error parsing reasoning."
            }

    # =============== CHORUS/SONG STRUCTURE METHODS ===============

    def generate_chorus(self, theme, hook=None, verse_context=None, style=None, genre=None):
        """Generate a chorus based on a theme and optional hook phrase"""
        # Build a chorus-specific prompt
        prompt = f"""Write a catchy chorus {'in the style of ' + style if style else 'for a ' + genre + ' song' if genre else ''} about {theme}"""

        if hook:
            prompt += f" using the hook phrase: \"{hook}\""

        if verse_context:
            prompt += f"\n\nIt should complement this verse:\n{verse_context}"

        # Add rhyming enhancment
        enhance_prompt = self.enhance_prompt_with_rhymes(
            prompt, theme, style, genre,
            rhyme_types=["perfect", "multi_syllable"]  # Focus on stronger rhymes for chorus
        )

        # Get chorus from GPT
        response = self.get_gpt_response(enhance_prompt, cache_file=self.verse_cache_file)

        # Simple parsing for chorus (usually shorter and more direct than verse)
        chorus = response.strip()

        # If we get reasoning + chorus format, extract just the chorus
        if "Chorus:" in chorus:
            parts = chorus.split("Chorus:")
            chorus = parts[1].strip()

        return chorus

    def generate_song(self, theme, title=None, style=None, genre=None,
                     structure=None, hooks=None):
        """Generate a complete song with multiple sections"""
        # Default structure if none provided
        if not structure:
            if genre and genre.lower() == "rap":
                structure = ["intro", "verse", "chorus", "verse", "chorus", "outro"]
            else:
                structure = ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]

        # Generate title if needed
        if not title:
            title_prompt = f"Create a catchy title for a {'rap' if genre == 'rap' else 'song'} about {theme}"
            title_response = self.get_gpt_response(title_prompt)
            title = title_response.strip().strip('"\'')

        # Initialize song sections
        song = {
            'title': title,
            'theme': theme,
            'style': style,
            'genre': genre,
            'sections': {}
        }

        # Generate each section
        verse_count = 0
        chorus_content = None

        for section in structure:
            if section == "verse":
                verse_count += 1
                verse_key = f"verse{verse_count}"
                # Generate verse with reference to chorus if we have one
                verse_context = chorus_content if chorus_content and verse_count > 1 else None
                verse_result = self.generate_verse(
                    theme, style, genre,
                    length=16 if genre and genre.lower() == "rap" else 8,
                    technical_requirements=None
                )
                song['sections'][verse_key] = verse_result['verse']

            elif section == "chorus":
                # Use provided hook if available
                hook = hooks[0] if hooks else None
                # Generate chorus with reference to first verse if available
                verse_context = song['sections'].get('verse1', None)
                chorus_content = self.generate_chorus(
                    theme, hook, verse_context, style, genre
                )
                song['sections']['chorus'] = chorus_content

            elif section == "bridge":
                # Generate a bridge that contrasts with chorus
                bridge_prompt = f"Write a bridge for a song about {theme} that provides contrast to the chorus"
                if 'chorus' in song['sections']:
                    bridge_prompt += f"\n\nChorus to contrast with:\n{song['sections']['chorus']}"
                bridge_response = self.get_gpt_response(bridge_prompt)
                bridge_content = bridge_response.strip()
                if "Bridge:" in bridge_content:
                    parts = bridge_content.split("Bridge:")
                    bridge_content = parts[1].strip()
                song['sections']['bridge'] = bridge_content

            elif section in ["intro", "outro"]:
                # Generate shorter intro/outro sections
                section_prompt = f"Write a short {section} for a {'rap' if genre == 'rap' else 'song'} about {theme}"
                if section == "outro" and 'intro' in song['sections']:
                    section_prompt += f"\n\nIt should provide closure to the intro:\n{song['sections']['intro']}"
                section_response = self.get_gpt_response(section_prompt)
                section_content = section_response.strip()
                if f"{section.capitalize()}:" in section_content:
                    parts = section_content.split(f"{section.capitalize()}:")
                    section_content = parts[1].strip()
                song['sections'][section] = section_content

        return song


# Enhanced version of original function that leverages the new rhyming capabilities
def enhanced_gpt_generate(prompt, theme=None, style=None, genre=None, api_key=None):
    """Enhanced version of gpt_generate that uses the rhyming engine"""
    # Create integration instance
    integration = LyricGPTIntegration(api_key)

    # If it looks like a lyric generation request, use the full capabilities
    if any(keyword in prompt.lower() for keyword in ['verse', 'lyrics', 'song', 'rap', 'rhyme']):
        # Extract theme if not provided
        if not theme:
            words = prompt.lower().split()
            for candidate in ["love", "money", "success", "pain", "hope"]:
                if candidate in words:
                    theme = candidate
                    break
            # Default theme if none found
            if not theme:
                theme = "life"

        # Generate enhanced lyrics
        return integration.generate_verse(theme, style, genre)['verse']
    else:
        # Use simple enhancement for other types of requests
        enhanced_prompt = integration.enhance_prompt_with_rhymes(prompt, theme, style, genre)

        # Call GPT with the enhanced prompt
        if api_key:
            try:
                response = integration.get_gpt_response(enhanced_prompt)
                return response
            except:
                return gpt_generate(enhanced_prompt)
        else:
            # Use original function if no API key
            return gpt_generate(enhanced_prompt)