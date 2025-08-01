import argparse
import json
from rhyme_engine import RhymeEngine
from expression_finder import ExpressionFinder
from wordplay_detector import WordplayDetector

def main():
    parser = argparse.ArgumentParser(description='Advanced Rhyming Engine CLI')

    # Main arguments
    parser.add_argument('word', help='Word to analyze')
    parser.add_argument('--type', '-t', default='perfect',
                        help='Rhyme type (perfect, consonance, assonance, etc.)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Show all rhyme types')
    parser.add_argument('--expressions', '-e', action='store_true',
                        help='Show expressions containing the word')
    parser.add_argument('--wordplay', '-w', action='store_true',
                        help='Show wordplay opportunities')
    parser.add_argument('--limit', '-l', type=int, default=10,
                        help='Limit number of results')

    args = parser.parse_args()

    # Initialize components
    rhyme_engine = RhymeEngine()
    expression_finder = ExpressionFinder()
    wordplay_detector = WordplayDetector(rhyme_engine)

    # Process based on arguments
    if args.all:
        # Get all rhyme types
        results = rhyme_engine.find_all_rhyme_types(args.word)

        # Limit each result category
        for key in results:
            if isinstance(results[key], list) and len(results[key]) > args.limit:
                results[key] = results[key][:args.limit]

        print(json.dumps(results, indent=2))
    elif args.expressions:
        # Get expressions
        results = expression_finder.find_expressions(args.word, max_expressions=args.limit)
        print(json.dumps(results, indent=2))
    elif args.wordplay:
        # Get wordplay opportunities
        results = wordplay_detector.find_wordplay_opportunities(args.word)
        print(json.dumps(results, indent=2))
    else:
        # Get specific rhyme type
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

        if args.type not in rhyme_types:
            print(f"Error: Unknown rhyme type '{args.type}'")
            print(f"Available types: {', '.join(rhyme_types.keys())}")
            return

        # Get method by name
        method = getattr(rhyme_engine, rhyme_types[args.type])

        # Handle special case for multi-syllable rhymes
        if args.type == 'multi_syllable':
            results = method(args.word, 2)  # Default to 2 syllables
        else:
            results = method(args.word)

        # Limit results
        if len(results) > args.limit:
            results = results[:args.limit]

        print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()