from utilities.internal_tools import tree_find_trace


def get_cause_with_layer(cause_tree, ):
    rank2cause = {i: set() for i in range(6)}

    def _get_cause_with_layer(cause_tree, rank=0):
        for node, branch in cause_tree.items():
            rank2cause[rank].add(node)
            if branch:
                _get_cause_with_layer(branch, rank + 1)

    _get_cause_with_layer(cause_tree)
    rank2cause.pop(0)
    return rank2cause


def evaluate_on_former_layer(index2cause, cause_tree, predicts, targets, leaves, cause2num):
    '''
    I also want to do evaluation on few shot causes
    '''

    from utilities.internal_tools import tree_find_relevant_nodes

    all_causes = tree_find_relevant_nodes(tree=cause_tree, leaves=[index2cause(cause) for cause in leaves])
    all_causes = set(all_causes)
    predicts = [index2cause(predict) for predict in predicts]
    targets = [index2cause(target) for target in targets]
    predicts = [tree_find_trace(cause_tree, predict)[1:] for predict in predicts]
    targets = [tree_find_trace(cause_tree, target)[1:] for target in targets]
    tpfp_recorder = {cause: [0, 0, 0] for cause in all_causes}

    # cause [TP, FP, FN]
    def _precision(tpfp):
        try:
            return tpfp[0] / (tpfp[0] + tpfp[1])
        except ZeroDivisionError:
            return 0

    def _recall(tpfp):
        try:
            return tpfp[0] / (tpfp[0] + tpfp[2])
        except ZeroDivisionError:
            return 0

    def _fscore(tpfp):
        try:
            return 2 * tpfp[0] / (2 * tpfp[0] + tpfp[1] + tpfp[2])
        except ZeroDivisionError:
            return 0

    for i, causes in enumerate(targets):
        for cause in causes:
            if cause in predicts[i]:
                tpfp_recorder[cause][0] += 1
            else:
                tpfp_recorder[cause][2] += 1
    for i, causes in enumerate(predicts):
        for cause in causes:
            if cause not in targets[i]:
                tpfp_recorder[cause][1] += 1

    score_recorder = {cause: (_precision(tpfp), _recall(tpfp), _fscore(tpfp)) for cause, tpfp in
                      tpfp_recorder.items()}

    def analysis_layer():
        rank2cause = get_cause_with_layer(cause_tree)
        rank2cause = {rank: [cause for cause in causes if cause in all_causes] for rank, causes in rank2cause.items()}
        rank2score = {rank: [] for rank in rank2cause}
        for rank, causes in rank2cause.items():
            yes = 0
            for i in range(len(targets)):
                if len(targets[i]) < rank or len(predicts[i]) < rank:
                    continue
                if targets[i][rank - 1] == predicts[i][rank - 1]:
                    yes += 1
            num = 0
            for causes_list in targets:
                if len(causes_list) >= rank:
                    num += 1
            try:
                accuracy = yes / num
            except ZeroDivisionError:
                accuracy = 0
            precision = 0
            recall = 0
            fscore = 0
            for cause in causes:
                precision += score_recorder[cause][0]
                recall += score_recorder[cause][1]
                fscore += score_recorder[cause][2]
            num = len(causes)
            try:
                precision /= num
                recall /= num
                fscore /= num
            except ZeroDivisionError:
                precision = 0
                recall = 0
                fscore = 0
            rank2score[rank] = (accuracy, precision, recall, fscore)
        for i in range(len(rank2score)):
            print(i + 1, rank2score[i + 1])

    def analysis_num():
        '''
        30 - 100:
        100 - 1000:
        1000 - 无穷:
        '''

        def get_by_inteval(low, high=1e10):
            return [cause for cause, num in cause2num.items() if low <= num < high]

        inteval2cause = {'30': get_by_inteval(30, 100), '100': get_by_inteval(100, 1000),
                         '1000': get_by_inteval(1000, )}
        inteval2score = {}
        for inteval, causes in inteval2cause.items():
            precision = 0
            recall = 0
            fscore = 0
            for cause in causes:
                precision += score_recorder[cause][0]
                recall += score_recorder[cause][1]
                fscore += score_recorder[cause][2]
            num = len(causes)
            try:
                precision /= num
                recall /= num
                fscore /= num
            except ZeroDivisionError:
                precision = 0
                recall = 0
                fscore = 0
            inteval2score[inteval] = (precision, recall, fscore)
        for inteval, score, in inteval2score.items():
            print(inteval, score)

    analysis_layer()
    analysis_num()
