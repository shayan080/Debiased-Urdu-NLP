import json
import os

def load_professions():
    """
    Loads professions along with their gender bias metrics from a JSON file.
    """
    # Construct the path to the professions file relative to this script
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    professions_file = os.path.normpath(os.path.join(pkg_dir, '../data', 'urdu_professions.json'))
    
    try:
        with open(professions_file, 'r', encoding='utf-8') as f:
            professions = json.load(f)
        print('Loaded professions\n' +
              'Format:\n' +
              'word,\n' +
              'definitional female -1.0 -> definitional male 1.0\n' +
              'stereotypical female -1.0 -> stereotypical male 1.0')
        return professions
    except FileNotFoundError:
        print(f"Error: The file '{professions_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: There was an issue decoding '{professions_file}'. Check if it's valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    return []  # Return an empty list in case of error
