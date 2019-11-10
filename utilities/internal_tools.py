def re_index(word2id, vocab_size):
    id2word = {id: word for word, id in word2id.items()}
    index = 0
    id2word_prime = {}
    for i in range(vocab_size):
        if i in id2word:
            id2word_prime[index] = id2word[i]
            index += 1
    word2id = {word: id for id, word in id2word_prime.items()}
    return word2id

def tree_find_trace(tree, key):
    def _tree_find_trace(tree, key, box):
        if key in tree:
            box.append(key)
            return True
        have = False
        for node in tree:
            if tree[node]:
                if _tree_find_trace(tree[node], key, box):
                    box.append(node)
                    have = True
        return have
    box = []
    _tree_find_trace(tree, key, box)
    box.reverse()
    return box

def tree_find_relevant_nodes(tree, leaves):
    relevant_nodes = []
    for leaf in leaves:
        trace = tree_find_trace(tree, leaf)
        relevant_nodes += trace
    relevant_nodes = set(relevant_nodes)
    return relevant_nodes

def tree_find_kids(tree, key):
    if key in tree:
        return list(tree[key].keys())
    for node in tree:
        if tree[node]:
            result = tree_find_kids(tree[node], key)
            if isinstance(result, list):
                return result
    return None

