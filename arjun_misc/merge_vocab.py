"""
merge_vocab.py
- Merge vocab from Saycam and Llava datasets
"""
import json

def merge_vocabs(saycam_vocab_path, llava_vocab_path, save_path):
    # Load SayCam vocab (dictionary format)
    with open(saycam_vocab_path, 'r') as f:
        saycam_vocab = json.load(f)

    # Load Llava vocab (list of strings)
    with open(llava_vocab_path, 'r') as f:
        llava_vocab = json.load(f)

    # Initialize index for new words (starting from 2350)
    new_index = 2350

    # Track counts
    already_in_saycam = 0
    newly_added = 0

    # Merge vocabularies
    for word in llava_vocab:
        if word in saycam_vocab:
            already_in_saycam += 1
        else:
            saycam_vocab[word] = new_index
            new_index += 1
            newly_added += 1

    # Print statistics
    print(f"Number of Llava words already in SayCam: {already_in_saycam}")
    print(f"Number of new words added: {newly_added}")

    # Save merged vocab
    with open(save_path, 'w') as f:
        json.dump(saycam_vocab, f, indent=4)

    print(f"Merged vocab saved to {save_path}")


saycam_vocab = '/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/multimodal/vocab.json'
llava_vocab = '/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/arjun_misc/llava_vocab.json'
save_path = '/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/arjun_misc/vocab.json'

merge_vocabs(saycam_vocab_path=saycam_vocab, llava_vocab_path=llava_vocab, save_path=save_path)

