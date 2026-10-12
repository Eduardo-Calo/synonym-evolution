import pandas as pd
import nltk
from pathlib import Path
import argparse
from nltk.corpus import wordnet as wn

nltk.download('wordnet', quiet=True)


def get_wordnet_pos(pos_tag):
    if pos_tag == 'NOUN':
        return wn.NOUN
    elif pos_tag == 'VERB':
        return wn.VERB
    elif pos_tag == 'ADJ':
        return wn.ADJ
    return None


def process_csv(csv_path, output_path, pos):
    df = pd.read_csv(csv_path)
    if 'lemma' not in df.columns or 'coarse_pos' not in df.columns:
        print(f"Skipping {csv_path}: missing 'lemma' or 'coarse_pos' columns")
        return

    # Convert numeric columns to appropriate types
    for col in ['start', 'end', 'n_usages']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Map each unique synset set to a numeric ID
    synsets_list = []

    for lemma, pos_tag in zip(df['lemma'], df['coarse_pos']):
        lemma = str(lemma).strip()
        wn_pos = get_wordnet_pos(pos_tag)
        syns = set(syn.name() for syn in wn.synsets(lemma.replace(" ", "_"), wn_pos)) if wn_pos else set()
        synsets_list.append(syns)

    df['synsets'] = [','.join(sorted(syns)) for syns in synsets_list]
    df.to_csv(output_path, index=False)

    # Print stats
    total_lemmas = len(synsets_list)
    lemmas_with_synsets = sum(1 for s in synsets_list if s)
    print(f"For {pos}: {total_lemmas} lexical units, {lemmas_with_synsets} with synsets ({lemmas_with_synsets/total_lemmas*100:.1f}%)")

    print(f"Saved {output_path} with synsets.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parsed-dir', type=Path, default=Path('../data/parsed'), help='Directory with POS CSV files')
    args = parser.parse_args()

    for csv_file in args.parsed_dir.glob('*.csv'):
        if '_wn' in csv_file.stem:
            continue  # Skip already processed _wn files
        pos = csv_file.stem
        if pos not in ['NOUN', 'VERB', 'ADJ']:
            continue  # Skip POS not in WordNet
        output_file = args.parsed_dir / f"{pos}_wn.csv"
        process_csv(csv_file, output_file, pos)


if __name__ == "__main__":
    main()
