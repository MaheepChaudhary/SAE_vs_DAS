from imports import *
from Pythia.config import *

# dataset hyperparameters
dataset = load_dataset(gender_dataset)

# data preparation hyperparameters
# batch_size = 512

SEED = 42

# To fit on 24GB VRAM GPU, I set the next 2 default batch_sizes to 64
def get_data(DEVICE, train=True, ambiguous=True, batch_size=1, seed=SEED):
    if train:
        data = dataset['train']
    else:
        data = dataset['test']
    if ambiguous:
        neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 0]
        pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 1]
        n = min([len(neg), len(pos)])
        neg, pos = neg[:n], pos[:n]
        data = neg + pos
        labels = [0]*n + [1]*n
        idxs = list(range(2*n))
        random.Random(seed).shuffle(idxs)
        data, labels = [data[i] for i in idxs], [labels[i] for i in idxs]
        true_labels = spurious_labels = labels
    else:
        neg_neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 0]
        neg_pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 1]
        pos_neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 0]
        pos_pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 1]
        n = min([len(neg_neg), len(neg_pos), len(pos_neg), len(pos_pos)])
        neg_neg, neg_pos, pos_neg, pos_pos = neg_neg[:n], neg_pos[:n], pos_neg[:n], pos_pos[:n]
        data = neg_neg + neg_pos + pos_neg + pos_pos
        true_labels     = [0]*n + [0]*n + [1]*n + [1]*n
        spurious_labels = [0]*n + [1]*n + [0]*n + [1]*n
        idxs = list(range(4*n))
        random.Random(seed).shuffle(idxs)
        data, true_labels, spurious_labels = [data[i] for i in idxs], [true_labels[i] for i in idxs], [spurious_labels[i] for i in idxs]

    batches = [
        (data[i:i+batch_size], t.tensor(true_labels[i:i+batch_size], device=DEVICE), t.tensor(spurious_labels[i:i+batch_size], device=DEVICE)) for i in range(0, len(data), batch_size)
    ]

    return batches

def get_subgroups(DEVICE, train=True, ambiguous=True, batch_size=1, seed=SEED):
    if train:
        data = dataset['train']
    else:
        data = dataset['test']
    if ambiguous:
        neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 0]
        pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 1]
        neg_labels, pos_labels = (0, 0), (1, 1)
        subgroups = [(neg, neg_labels), (pos, pos_labels)]
    else:
        neg_neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 0]
        neg_pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[male_prof] and x['gender'] == 1]
        pos_neg = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 0]
        pos_pos = [x['hard_text'] for x in data if x['profession'] == profession_dict[female_prof] and x['gender'] == 1]
        neg_neg_labels, neg_pos_labels, pos_neg_labels, pos_pos_labels = (0, 0), (0, 1), (1, 0), (1, 1)
        subgroups = [(neg_neg, neg_neg_labels), (neg_pos, neg_pos_labels), (pos_neg, pos_neg_labels), (pos_pos, pos_pos_labels)]
    
    out = {}
    for data, label_profile in subgroups:
        out[label_profile] = []
        for i in range(0, len(data), batch_size):
            text = data[i:i+batch_size]
            out[label_profile].append(
                (
                    text,
                    t.tensor([label_profile[0]]*len(text), device=DEVICE),
                    t.tensor([label_profile[1]]*len(text), device=DEVICE)
                )
            )
    return out