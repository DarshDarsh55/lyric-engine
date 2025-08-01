from flask import Flask, request, jsonify
import os
import sys
import json
from rhyme_engine import RhymeEngine
from expression_finder import ExpressionFinder
from wordplay_detector import WordplayDetector

app = Flask(__name__)

# Initialize components
data_dir = "rhyme_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

rhyme_engine = RhymeEngine(data_dir=data_dir)
expression_finder = ExpressionFinder(data_dir=data_dir)
wordplay_detector = WordplayDetector(rhyme_engine, data_dir=data_dir)

# Root endpoint
@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'message': 'Advanced Rhyming Engine API is running!',
        'endpoints': [
            '/rhymes/<type>/<word>',
            '/all_rhymes/<word>',
            '/expressions/<word>',
            '/rhyming_expressions/<word>',
            '/thematic_expressions/<theme>',
            '/wordplay/<word>',
            '/puns/<word>'
        ]
    })

# Rhyme endpoints
@app.route('/rhymes/<rhyme_type>/<word>')
def get_rhymes(rhyme_type, word):
    try:
        # Map rhyme type to method name
        rhyme_types = {
            'perfect': 'perfect_rhymes',
            'consonance': 'consonance_rhymes',
            'assonance': 'assonance_rhymes',
            'multi_syllable': 'multi_syllable_rhymes',
            'first_syllable': 'first_syllable_rhymes',
            'final_syllable': 'final_syllable_rhymes',
            'alliteration': 'alliteration',
            'metaphone': 'metaphone_rhymes',
            'soundex': 'soundex_rhymes',
            'para': 'para_rhymes',
            'feminine_para': 'feminine_para_rhymes',
            'homophones': 'homophones',
            'light': 'light_rhymes',
            'family': 'family_rhymes',
            'broken': 'broken_rhymes',
            'reverse': 'reverse_rhymes',
            'weakened': 'weakened_rhymes',
            'trailing': 'trailing_rhymes',
            'full_consonance': 'full_consonance',
            'full_assonance': 'full_assonance',
            'double_consonance': 'double_consonance',
            'double_assonance': 'double_assonance',
            'diminished': 'diminished_rhymes',
            'intelligent': 'intelligent_rhymes',
            'related': 'related_rhymes',
            'additive': 'additive_rhymes'
        }

        if rhyme_type not in rhyme_types:
            return jsonify({
                'error': f'Unknown rhyme type: {rhyme_type}',
                'available_types': list(rhyme_types.keys())
            }), 400

        # Get method by name
        method = getattr(rhyme_engine, rhyme_types[rhyme_type])

        # Handle special case for multi-syllable rhymes
        if rhyme_type == 'multi_syllable':
            syllable_count = request.args.get('syllables', 2, type=int)
            results = method(word, syllable_count)
        else:
            results = method(word)

        return jsonify({
            'word': word,
            'rhyme_type': rhyme_type,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/all_rhymes/<word>')
def get_all_rhymes(word):
    try:
        results = rhyme_engine.find_all_rhyme_types(word)

        # Limit the result size
        for key in results:
            if isinstance(results[key], list) and len(results[key]) > 10:
                results[key] = results[key][:10]

        return jsonify({
            'word': word,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Expression endpoints
@app.route('/expressions/<word>')
def get_expressions(word):
    try:
        max_expressions = request.args.get('max', 20, type=int)
        results = expression_finder.find_expressions(word, max_expressions=max_expressions)

        return jsonify({
            'word': word,
            'expressions': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/rhyming_expressions/<word>')
def get_rhyming_expressions(word):
    try:
        max_expressions = request.args.get('max', 20, type=int)
        results = expression_finder.find_rhyming_expressions(word, rhyme_engine, max_expressions=max_expressions)

        return jsonify({
            'word': word,
            'rhyming_expressions': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/thematic_expressions/<theme>')
def get_thematic_expressions(theme):
    try:
        max_expressions = request.args.get('max', 30, type=int)
        results = expression_finder.find_thematic_expressions(theme, max_expressions=max_expressions)

        return jsonify({
            'theme': theme,
            'thematic_expressions': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

# Wordplay endpoints
@app.route('/wordplay/<word>')
def get_wordplay(word):
    try:
        results = wordplay_detector.find_wordplay_opportunities(word)

        return jsonify({
            'word': word,
            'wordplay_opportunities': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/puns/<word>')
def get_puns(word):
    try:
        context = request.args.get('context', None)
        results = wordplay_detector.find_pun_opportunities(word, phrase_context=context)

        return jsonify({
            'word': word,
            'context': context,
            'pun_opportunities': results
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)