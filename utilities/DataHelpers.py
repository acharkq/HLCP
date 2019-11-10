from utilities.SavedData import *



def get_valid_cause(cut_fn=10, lixiang=False, civil=False):
    if not civil:
        keyword2index, cause2index, law2index, result2index, tree = load_indexes(load_cause_dict=True)
        if lixiang:
            id2info = id2zuiming_fn(sta=True)
            causes = []
            for label, item in id2info.items():
                case = []

                if tree_find(tree=tree, key=item[0], box=case):
                    causes += case
                else:
                    print(item[0])
                    raise ('报警')
            causes = set(causes)
            causes = set([cause2index[cause] for cause in causes])
        else:
            with open(getPath('cause_statistics'), "r") as f:
                cause_sta = json.load(f)
            causes = set([int(cause) for cause, count in cause_sta.items() if count > cut_fn])
        causes.add(243)
        return causes
    with open(os.path.join(getPath('civil_bin_dir'), 'final_sta.json'), 'r') as f:
        id2sta = json.load(f)
    causes = set([int(id) for id, sta in id2sta.items() if sta >= cut_fn])
    # add 民事
    causes.add(0)
    return causes
