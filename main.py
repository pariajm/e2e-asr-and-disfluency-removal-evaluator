#!/usr/bin/env python3

import argparse
import os
import min_distance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='finding an alignment or running a unit test', required=True)
    parser.add_argument('--result-path', type=str, help='path to write the results')
    parser.add_argument('--ref', type=str, help='path to the reference sentences')
    parser.add_argument('--hyp', type=str, help='path to the hypothesis sentences')
    args = parser.parse_args()
    assert (args.mode == 'align' or args.mode == 'mod_align' or args.mode == 'test')

    if args.result_path: 
        result_file = open(os.path.join(args.result_path, 'results.txt'), 'w')

    if args.mode == 'test': 
        distance = min_distance.Test()
        print(distance)

    elif args.mode == 'align' or args.mode == 'mod_align':
        sents = zip(open(args.ref), open(args.hyp)) 
        print('Loading {} and {} files ...'.format(
            os.path.split(args.ref)[1],
            os.path.split(args.hyp)[1],
            )
            )
        all_scores = list()
        modified_weights = False if args.mode=='align' else True        
        for idx, sent in enumerate(sents, start=1):
            distance = min_distance.MinDistance(
                Ref=sent[0], 
                Hyp=sent[1],
                modified_weights=modified_weights,
            )
            alignment, scores = distance.align()      
            all_scores.append(scores)    
            alignment = '{1} \nSent #{0} \n{1} \n{2}'.format(idx, '-'*50, alignment)
            print(alignment)          
            if args.result_path: 
                result_file.write(alignment+'\n')

        sum_scores = [sum(score) for score in zip(*all_scores)] 
        # Calculating fluent error rate (FER). If args.mode == 'align' --> FER == WER             
        fluent_m, fluent_s, fluent_d, fluent_i, fluent_all = sum_scores[:5] 
        fer = (fluent_s + fluent_d + fluent_i) / fluent_all          
        print('{}\nFluent Error Rate (FER): {}/{} = {:.3f}'.format(
            '='*50, fluent_s+fluent_d+fluent_i, fluent_all, fer
            ))

        if modified_weights:  
            # Calculating disfluent error rate (DER), Edited F-score, Precision and Recall
            disfluent_m, disfluent_s, disfluent_d, disfluent_i, disfluent_all = sum_scores[5:]                        
            der = (disfluent_s + disfluent_m + disfluent_i) / disfluent_all     
            print('Disfluent Error Rate (DER): {}/{} = {:.3f}'.format(
                disfluent_s+disfluent_m+disfluent_i, disfluent_all, der
                ))                    
            print('\nPrecision: {}/{} = {:.3f} \nRecall: {}/{} = {:.3f}'.format(
                disfluent_d, disfluent_d+fluent_d, disfluent_d/(disfluent_d+fluent_d),
                disfluent_d, disfluent_all, disfluent_d/disfluent_all
                ))
            print('Edited F-score: {}/{} = {:.3f}'.format(
                2*disfluent_d, (disfluent_all+disfluent_d+fluent_d), 
                2*disfluent_d/(disfluent_all+disfluent_d+fluent_d)
                ))


if __name__ == '__main__':
    main()