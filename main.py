from itertools import product
import os

def main():
    try:
        os.remove("submission.txt")
    except OSError:
        pass

    try:
        os.remove("submission.zip")
    except OSError:
        pass
    vocab, all_postags, prob_all_transition, prob_all_emission = training()
    testing(vocab, all_postags, prob_all_transition, prob_all_emission)


def testing(vocab, all_postags, prob_all_transition, prob_all_emission):
    # Now start reading testing set
    print("Now testing...")

    with open("test.conll", encoding="utf8") as f:
        #initialization
        words_to_print = []
        lang_to_print = []
        tag_to_print = []

        prev_position = 0
        trellis = []
        backtrace = []
        trellis.append({})
        backtrace.append({})
        line = f.readline()
        while line:
            cur_position = prev_position + 1
            # If the line is not at the end of the sentence
            if line != "\n":
                line_separated = line.split()
                trellis.append({})
                backtrace.append({})
                cur_word = line_separated[0]
                words_to_print.append(cur_word)
                if cur_word not in vocab:
                    if cur_position == 1 or cur_word[0].isupper():
                        cur_word = "<name>"
                    else:
                        cur_word = "<unseen>"
                if cur_position == 1:
                    for each_tag in all_postags:
                        transition = (each_tag, '<s>')
                        prob_transition = prob_all_transition.get(transition, 0.0)
                        # if prob_transition == 0:
                        #     prob_transition = prob_all_transition[('<unseen>', each_tag)]
                        emission = (cur_word, each_tag)
                        prob_emission = prob_all_emission.get(emission, 0.0)
                        # if prob_emission == 0:
                        #     prob_emission = prob_all_emission[('<unseen>', each_tag)]
                        trellis[cur_position][each_tag] = prob_emission * prob_transition
                        backtrace[cur_position][each_tag] = '<s>'
                else:
                    for each_cur_tag in all_postags:
                        best_prob = -9999.0
                        best_prev_tag = ''
                        emission = (cur_word, each_cur_tag)
                        prob_emission = prob_all_emission.get(emission, 0.0)
                        # if prob_emission == 0:
                        #     prob_emission = prob_all_emission[('<unseen>', each_tag)]
                        for each_prev_tag in all_postags:
                            transition = (each_cur_tag, each_prev_tag)
                            prob_transition = prob_all_transition.get(transition, 0.0)
                            # if prob_transition == 0:
                            #     prob_transition = prob_all_transition[('<unseen>', each_tag)]
                            prev_prob = trellis[prev_position][each_prev_tag]
                            cur_prob = prev_prob * prob_emission * prob_transition
                            if cur_prob > best_prob:
                                best_prob = cur_prob
                                best_prev_tag = each_prev_tag
                        trellis[cur_position][each_cur_tag] = best_prob
                        # store backtrace as dict {next_tag: prev_tag}
                        backtrace[cur_position][each_cur_tag] = best_prev_tag


                lang_to_print.append(line_separated[1])
                prev_position = cur_position
            # If the line is the end of the sentence
            else:
                ending_prob = -9999.0
                ending_tag = ''
                for each_tag in all_postags:
                    transition = ('</s>', each_tag)
                    prob_transition = prob_all_transition.get(transition, 0.0)
                    # if prob_transition == 0:
                    #     prob_transition = prob_all_transition[('<unseen>', each_tag)]
                    prev_prob = trellis[prev_position][each_tag]
                    prob_to_ending = prob_transition * prev_prob
                    if prob_to_ending > ending_prob:
                        ending_prob = prob_to_ending
                        ending_tag = each_tag
                tag_to_print.insert(0, ending_tag)
                cur_tag = ending_tag
                while len(backtrace) > 2:
                    prev_transition = backtrace.pop()
                    prev_tag = prev_transition[cur_tag]
                    tag_to_print.insert(0, prev_tag)
                    cur_tag = prev_tag

                with open("submission.txt", 'a+') as fout:
                    for i in range(0, len(tag_to_print)):
                        fout.write(words_to_print[i] + '\t' + lang_to_print[i] + '\t' + tag_to_print[i] + '\n')
                    fout.write(line)
                words_to_print = []
                lang_to_print = []
                tag_to_print = []
                prev_position = 0

                trellis = []
                backtrace = []
                trellis.append({})
                backtrace.append({})
            line = f.readline()


def training():
    count_tags = {}
    count_bigrams = {}
    count_emission = {}
    vocab = []
    # First read the corpus and build the count of bigram and emission
    with open("train.conll", encoding="utf8") as f:
        line = f.readline()
        pos_prev = '<s>'

        while line:
            if pos_prev in count_tags:
                count_tags[pos_prev] += 1
            else:
                count_tags[pos_prev] = 1

            if line != "\n":
                line_separated = line.split()
                pos_cur = line_separated[len(line_separated) - 1]
                if pos_cur == 'PROPN':
                    if "<name>" not in count_emission:
                        count_emission[("<name>", 'PROPN')] = 1
                    else:
                        count_emission[("<name>", 'PROPN')] += 1
                bigram = (pos_cur, pos_prev)

                if bigram in count_bigrams:
                    count_bigrams[bigram] += 1
                else:
                    count_bigrams[bigram] = 1

                word_cur = line_separated[0]

                if word_cur not in vocab:
                    vocab.append(word_cur)
                emission = (word_cur, pos_cur)

                if emission in count_emission:
                    count_emission[emission] += 1
                else:
                    count_emission[emission] = 1

                pos_prev = pos_cur
            else:
                pos_cur = '</s>'
                if pos_cur not in count_tags:
                    count_tags[pos_cur] = 1
                else:
                    count_tags[pos_cur] += 1

                bigram = (pos_cur, pos_prev)
                if bigram in count_bigrams:
                    count_bigrams[bigram] += 1
                else:
                    count_bigrams[bigram] = 1
                pos_prev = '<s>'
            line = f.readline()

    print("Done reading the file, now start training...")
    # Next build the probability
    # all_postags.append('</s>')
    # all_pos_bigram = list(product(all_postags, repeat=2))
    #count_all_bigram = len(all_pos_bigram)
    prob_all_transition = {}
    count_all_postag = len(count_tags)
    for key in count_bigrams:
        count_prev_tag = count_tags.get(key[1], 0)
        bigram_count = count_bigrams.get(key, 0)
        prob_all_transition[key] = bigram_count/count_prev_tag

    prob_all_emission = {}
    #count_all_emission = len(all_emission)
    for key in count_emission:
        count_tag = count_tags.get(key[1], 0)
        emission_count = count_emission.get(key, 0)
        prob_all_emission[key] = float(emission_count + 1)/(count_all_postag + count_tag)

    # Smoothing for unseen words or emission
    count_all_postag = len(count_tags)
    for each_postag in count_tags.keys():
        count_posttag = count_tags.get(each_postag, 0)
        prob_all_emission[('<unseen>', each_postag)] = 1.0/(count_all_postag + count_posttag)
        prob_all_transition[('<unseen>', each_postag)] = 1.0/(count_all_postag + count_posttag)
    return vocab, list(count_tags.keys()), prob_all_transition, prob_all_emission

main()