"""
prepare_vocab.py
- Prepare vocab.json file for LLaVA pretrain transferred dataset 
- Following details in supplementary info (with some changes to include more vocab): 
We also applied the  spaCy tokenizer on the filtered utterances from the training set to build the vocabulary, 
replacing  anything annotated as inaudible and any tokens with a frequency less than 3 in the dataset
with an <UNK> token, resulting in a vocabulary size of 2350. All transcripts were lowercased 
(although in some of our figures some child-directed utterances are capitalized for ease of reading). 
All utterances were truncated to a maximum length of 25 tokens.
"""

from tqdm import tqdm
import json
import spacy
from collections import Counter

def get_llava_vocab(data_path, save_path):

    nlp = spacy.load(
        'en_core_web_sm',
        exclude=[
            'attribute_ruler', 'lemmatizer', 'ner',
            'senter', 'parser', 'tagger', 'tok2vec']
    )

    def tokenize(s, kind='spacy'):
        if kind == 'spacy':
            return nlp.tokenizer(s)
        elif kind == 'space':
            return s.split()
        else:
            raise Exception(f"Unrecognized {kind=}")


    # Initialize variables
    token_counts = Counter()
    truncated_count = 0

    # Open the JSON file and load data
    with open(data_path, 'r') as file:
        data = json.load(file)

    num_llava = 0

    # Iterate over the list of JSON objects
    for item in tqdm(data, desc="Processing captions", unit="item"):
        if item['image'].startswith('llava'):
            num_llava += 1
            conversations = item.get('conversations', [])
            if len(conversations) > 1:
                #get caption and convert to lower case 
                caption = conversations[1].get('value', '').lower()
                
                # Tokenize the caption
                tokens = [token.text for token in tokenize(caption)]

                # Truncate to 25 tokens
                if len(tokens) > 25:
                    tokens = tokens[:25]
                    truncated_count += 1
                
                # Update the token counts
                token_counts.update(tokens)

    print(token_counts)

    print(f"Processed {num_llava} LLaVA images")
  
    # Print info about truncation
    print(f'Number of sentences truncated to 25 tokens: {truncated_count}')

    # Remove tokens with frequency less than 3 - this step removes ~1500 tokens of 4000
    tokens_to_remove = [token for token, count in token_counts.items() if count < 3]
    # for token in tokens_to_remove:
    #     del token_counts[token]

    # Print the number of tokens removed
    print(f'Number of tokens removed (frequency < 3): {len(tokens_to_remove)}')

    # Save the vocab to a file
    with open(save_path, 'w') as vocab_file:
        json.dump(list(token_counts.keys()), vocab_file, indent=4)



# Define paths
llava_data_path = '/projectnb/ivc-ml/wsashawn/LLaVA/llava_SAYCam_mixed_data.json'
#saycam_vocab = '/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/multimodal/vocab.json'

get_llava_vocab(llava_data_path, 
save_path='/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/arjun_misc/llava_vocab.json')