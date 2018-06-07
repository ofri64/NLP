from PCFG import PCFG
import math
import numpy as np

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def cky(pcfg, sent):
    ### YOUR CODE HERE

    pi_dict = {}
    bp_dict = {}
    sent_tokens = sent.split(" ")
    n = len(sent_tokens)

    # Initializtion part

    for x in pcfg._rules.keys():
        x_rules = pcfg._rules.get(x)
        x_sum = pcfg._sums.get(x)

        for i in range(1, n + 1): # i=1 -> i=n
            key = (i, i, x)
            word_i = sent_tokens[i-1]
            for rhs, w in x_rules:
                if pcfg.is_preterminal(rhs) and rhs[0] == word_i:
                    pi_dict[key] = w / x_sum
                    bp_dict[key] = rhs
                    break

    # Check if sentence can be parsed by grammar
    # We have to find for every word in the sentence at least one derivation rule with prob > 0
    # Meaning the is x such that pi(i, i, x) for word i is greater than 0

    for i in range(1, n+1):
        cant_parse = True
        for x in pcfg._rules.keys():
            if pi_dict.get((i, i, x), 0) != 0:
                cant_parse = False
                break

        if cant_parse:
            return "FAILED TO PARSE!"

    # Algorithm

    for l in range(1, n): # l=1 -> l=n-1
        for i in range(1, n-l+1): # i=1 -> i=n-l
            j = i + l
            for x in pcfg._rules.keys():

                x_rules = pcfg._rules.get(x)
                x_sum = pcfg._sums.get(x)
                rules_and_s_keys = []
                rules_and_s_scores = []

                for rhs, w in x_rules:
                    for s in range(i, j): # s=i -> s=j-1

                        rules_and_s_keys.append((rhs, s))
                        if pcfg.is_preterminal(rhs):
                            rules_and_s_scores.append(0.0)

                        else:
                            q_rule = w / x_sum
                            Y, Z = rhs[0], rhs[1]
                            Y_i_s_prob = pi_dict.get((i, s, Y), 0)
                            Z_s1_j_prob = pi_dict.get((s+1, j, Z), 0)
                            rules_and_s_scores.append(q_rule * Y_i_s_prob * Z_s1_j_prob)

                score_argmax = np.argmax(rules_and_s_scores)
                pi_dict[(i, j, x)] = rules_and_s_scores[score_argmax]
                bp_dict[(i, j, x)] = rules_and_s_keys[score_argmax]

    # Parsing to a tree
    def gen_tree_cky(i, j, symbol):
        rule = bp_dict.get((i, j, symbol), None)

        if not rule:
            return "cannot parse"

        if len(rule) == 1:
            return "(" + symbol + " " + rule[0] + ")"

        rhs, s = rule[0], rule[1]
        if len(rhs) == 1:
            return "cannot parse"
        Y, Z = rhs[0], rhs[1]

        return "(" + symbol + " " + " ".join([gen_tree_cky(i, s, Y), gen_tree_cky(s+1, j, Z)]) + ")"

    tree = gen_tree_cky(1, n, 'ROOT')

    if "cannot parse" in tree:
        return "FAILED TO PARSE!"
    else:
        return tree

    ### END YOUR CODE

if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print cky(pcfg, sent)
