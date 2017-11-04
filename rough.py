
from helpers import *
import random


def create_type_pairs(opts, train_entities, train_relations, entity_count, relation_count, verbose=False):
    l = train_entities.shape[0]
    inverse_map_entity_head = [[]]*entity_count
    inverse_map_entity_tail = [[]]*entity_count
    inverse_map_relation = [[]]*relation_count
    head_relations = [set() for x in range(entity_count)]
    tail_relations = [set() for x in range(entity_count)]
    for i in range(l):
        inverse_map_relation[train_relations[i]].append(i)
        inverse_map_entity_head[train_entities[i,0]].append(i)
        inverse_map_entity_tail[train_entities[i,1]].append(i)
        head_relations[train_entities[i, 0]].add(train_relations[i])
        tail_relations[train_entities[i, 1]].add(train_relations[i])
    positive_size = opts.positive_type_samples
    negative_size = opts.negative_type_samples
    positive = []
    negative = []
    i = 0
    while(i < positive_size):
        e1 = random.randrange(entity_count)
        if(inverse_map_entity_tail[e1] or inverse_map_entity_head[e1]):
            head = True
            if(not inverse_map_entity_head[e1] or random.randrange(2)==1):
                head = False
            ls = inverse_map_entity_head[e1] if head else inverse_map_entity_tail[e1]
            r = train_relations[random.choice(ls)]
            sentence = random.choice(inverse_map_relation[r])
            e2 = train_entities[sentence, 0] if head else train_entities[sentence, 1]
            if(e1 != e2):
                positive.append([e1, e2])
                if(verbose):
                    print(e1, r, e2, head, sentence)
                i += 1
                if(verbose and i%100 == 0):
                    print(i)
    i = 0
    while(i < negative_size):
        e1 = random.randrange(entity_count)
        e2 = random.randrange(entity_count)
        if(head_relations[e1].isdisjoint(head_relations[e2]) and tail_relations[e1].isdisjoint(tail_relations[e2])):
            negative.append([e1, e2])
            i += 1
            if(verbose and i%100 == 0):
                print(i)
    return positive, negative

def dummy_kb(size, entity_count, relation_count):
    train_entities = np.random.randint(low=0, high=entity_count, size=(size, 2), dtype='int32')
    train_relations = np.random.randint(low=0, high=relation_count, size=(size), dtype='int32')
    return train_entities, train_relations

def load_type_pairs(opts):
    return

def save_type_pairs(opts, positive_pairs, negative_pairs):
    return



class Empty:
    pass
opts = Empty()
opts.dataset = '/home/prachij/code/joint_embedding/DATA_REPOSITORY/fb15k/original/encoded_data/without-text'
opts.evalDev = None
opts.train = False
opts.positive_type_samples = 10000
opts.negative_type_samples = 10000
y = dummy_kb(7, 4, 5)
print(y)
x = get_train_data_tensor(opts)
x_entity_count = 14951
x_relation_count = 1345
pnx = create_type_pairs(opts, x[0], x[1], x_entity_count, x_relation_count, True)

