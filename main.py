#!/usr/bin/env python3

import argparse
import os
import aligner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='finding an alignment or running a unit test', required=True)
    parser.add_argument('--result-path', type=str, help='path to write the results')
    parser.add_argument('--ref', type=str, help='path to the reference sentences')
    parser.add_argument('--hyp', type=str, help='path to the hypothesis sentences')
    args = parser.parse_args()
    
    # mode:
    # s_align --> standard alignment algorithm used to calculate WER
    # m_align --> modified alignment algorithm used to calculate FER and DER
    # test --> unit test for modified alignment algorithm
    assert (args.mode in ['s_align', 'm_align', 'test'])

    if args.result_path: 
        result_file = open(os.path.join(args.result_path, 'results.txt'), 'w')

    if args.mode == 'test': 
        distance = aligner.Test()
        print(distance)

    else:        
        sents = zip(open(args.ref), open(args.hyp)) 
        print('Loading {} and {} files ...'.format(
            os.path.split(args.ref)[1],
            os.path.split(args.hyp)[1]
            ))
        total_scores = list()      
        for indx, sent in enumerate(sents, start=1):
            distance = aligner.MinDistance(
                Ref=sent[0], 
                Hyp=sent[1],
                m_weights= True if args.mode=='m_align' else False
            )
            alignment, scores = distance.align()      
            total_scores.append(scores) 
            alignment = "{1} \nSent #{0} \n{1} \n{2}".format(indx, '-'*50, alignment)
            print(alignment)          
            if args.result_path: 
                result_file.write(alignment)

        sum_scores = list(map(sum, zip(*total_scores)))
        # Calculating fluent error rate (FER). If args.mode == 's_align' --> FER == WER             
        mat_f, sub_f, del_f, ins_f, total_f = sum_scores[:5] 
        numerator_fer = sub_f + del_f + ins_f
        denomerator_fer = total_f 
        fer = numerator_fer / denomerator_fer
        total_fer = '{}\nFluent Error Rate (FER): {}/{} = {:.3f}'.format(
            '='*50, numerator_fer, denomerator_fer, fer
            )
            
        if args.mode == 'm_align': 
            # Calculating disfluent error rate (DER), Edited F-score, Precision and Recall
            mat_d, sub_d, del_d, ins_d, total_d = sum_scores[5:]    
            numerator_der = sub_d + mat_d + ins_d
            denominator_der = total_d          
            der = numerator_der / denominator_der   
            total_der = 'Disfluent Error Rate (DER): {}/{} = {:.3f}'.format(
                numerator_der, denominator_der, der
                )
            print(total_fer+'\n'+total_der)

        else:
            print(total_fer.replace(
                'Fluent Error Rate (FER): ', 'Word Error Rate (WER): '
                )
                ) 


if __name__ == "__main__":
    main()
