import os
import json
import argparse
from rhyme_engine import RhymeEngine
from expression_finder import ExpressionFinder
from wordplay_detector import WordplayDetector
from gpt_integration import LyricGPTIntegration

def main():
    parser = argparse.ArgumentParser(description='Test the Advanced Rhyming System')
    
    # Main arguments
    parser.add_argument('--word', '-w', default='love',
                        help='Word to analyze')
    parser.add_argument('--theme', '-t', default='success',
                        help='Theme for lyric generation')
    parser.add_argument('--style', '-s', default='eminem',
                        help='Style for lyric generation')
    parser.add_argument('--test', choices=['rhymes', 'expressions', 'wordplay', 'gpt'], 
                        default='rhymes', help='Component to test')
    parser.add_argument('--output', '-o', default='test_output.json',
                        help='Output file for results')
    parser.add_argument('--api-key', '-k', default=None,
                        help='OpenAI API key (required for GPT test)')
    
    args = parser.parse_args()
    
    # Initialize data directory
    data_dir = "rhyme_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Run the selected test
    if args.test == 'rhymes':
        test_rhyme_engine(args.word, args.output)
    elif args.test == 'expressions':
        test_expression_finder(args.word, args.output)
    elif args.test == 'wordplay':
        test_wordplay_detector(args.word, args.output)
    elif args.test == 'gpt':
        if not args.api_key:
            print("Error: OpenAI API key is required for GPT test")
            print("Use --api-key or -k to provide your API key")
            return
        test_gpt_integration(args.theme, args.style, args.api_key, args.output)

def test_rhyme_engine(word, output_file):
    """Test the RhymeEngine with a word"""
    print(f"Testing RhymeEngine with word: {word}")
    
    # Initialize rhyme engine
    rhyme_engine = RhymeEngine()
    
    # Get all rhyme types
    results = rhyme_engine.find_all_rhyme_types(word)
    
    # Limit result size for readability
    for key in results:
        if isinstance(results[key], list) and len(results[key]) > 5:
            results[key] = results[key][:5]
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for rhyme_type, rhymes in results.items():
        if isinstance(rhymes, list):
            print(f"  {rhyme_type}: {len(rhymes)} rhymes found")
        elif isinstance(rhymes, dict):
            print(f"  {rhyme_type}: {len(rhymes)} items found")

def test_expression_finder(word, output_file):
    """Test the ExpressionFinder with a word"""
    print(f"Testing ExpressionFinder with word: {word}")
    
    # Initialize expression finder
    expression_finder = ExpressionFinder()
    
    # Get expressions
    expressions = expression_finder.find_expressions(word)
    
    # Get thematic expressions
    thematic = expression_finder.find_thematic_expressions(word)
    
    # Initialize rhyme engine for rhyming expressions
    rhyme_engine = RhymeEngine()
    
    # Get rhyming expressions
    rhyming = expression_finder.find_rhyming_expressions(word, rhyme_engine)
    
    # Create results
    results = {
        'word': word,
        'expressions': expressions[:10],
        'thematic_expressions': thematic[:10],
        'rhyming_expressions': rhyming[:10]
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Expressions: {len(expressions)} found")
    print(f"  Thematic Expressions: {len(thematic)} found")
    print(f"  Rhyming Expressions: {len(rhyming)} found")

def test_wordplay_detector(word, output_file):
    """Test the WordplayDetector with a word"""
    print(f"Testing WordplayDetector with word: {word}")
    
    # Initialize rhyme engine
    rhyme_engine = RhymeEngine()
    
    # Initialize wordplay detector
    wordplay_detector = WordplayDetector(rhyme_engine)
    
    # Get wordplay opportunities
    wordplay = wordplay_detector.find_wordplay_opportunities(word)
    
    # Get pun opportunities
    puns = wordplay_detector.find_pun_opportunities(word)
    
    # Create results
    results = {
        'word': word,
        'wordplay_opportunities': wordplay,
        'pun_opportunities': puns
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for key, items in wordplay.items():
        if isinstance(items, list):
            print(f"  {key}: {len(items)} items found")

def test_gpt_integration(theme, style, api_key, output_file):
    """Test the GPT integration with a theme and style"""
    print(f"Testing GPT integration with theme: {theme}, style: {style}")
    
    # Initialize GPT integration
    gpt_integration = LyricGPTIntegration(api_key)
    
    # Generate lyrics
    result = gpt_integration.generate_verse(theme, style=style)
    
    # Save result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print verse
    print("\nGenerated Verse:")
    print(result['verse'])

if __name__ == '__main__':
    main()
