#!/usr/bin/env python3

"""
This module implements a minimum edit distance algorithm for finding 
an alignment between reference (which may contain disfluencies) and 
hypothesis sentences. The modified algorithm uses different weights
for aligning fluent and disfluent regions, so it can be used for 
evaluating disfluency detection performance of any e2e systems 
which detect disfluencies as part of another task (e.g. ASR and 
disfluency removal systems).

Classes:
* MinDistance --> finds the minimum distance and aligns the strings
* Test --> a unit test using the modified alignment weights 

(c) Paria Jamshid Lou, 7th April 2020.
"""

class MinDistance():
    """
    If not m_weights --> find an alignment with the 
    following costs inspired by Sclite weights: 
        - match = 0
        - deletion = 3 
        - insertion = 3
        - substitution = 4
    
    If m_weights --> align fluent and disfluent words
    using two different sets of weights. Fluent words
    are aligned using the above weights. The following
    modified weights are used for aligning disfluent words: 
        - match = 0 + 1e-7
        - deletion = 3 - 1e-7
        - insertion = 3 + 1e-7
        - substitution = 4 + 1e-7

    Args:
        self.ref: a reference sentence where disfluent words 
        are in UPPERCASE letters
        self.hyp: a hypothesis sentence (i.e. model's output)
        self.insertion: insertion weight (default=3)
        self.deletion: deletion weight (default=3)
        self.substitution: substitution weight (default=4)
        self.m_weights: whether to use the modified weights 
        for disfluent words or not

    Returns:
        An alignment for each pair of reference and hypothesis strings,
        as well as the alignment scores.
    """
    def __init__(self, **kwargs):
        self.ref = kwargs['Ref'].split()
        self.hyp = kwargs['Hyp'].split()   
        self.insertion = kwargs['insertion'] if 'insertion' in kwargs else 3
        self.deletion = kwargs['deletion'] if 'deletion' in kwargs else 3
        self.substitution = kwargs['substitution'] if 'substitution' in kwargs else 4
        self.m_weights = kwargs['m_weights'] 
  
    def cost_matrix(self):
        """
        Calculates the cost matrix for aligning hypothesis and reference words.
        """ 
        previous_row = range(0, len(self.hyp)*self.insertion + 1, self.insertion)
        rows = list()     
        rows.append(list(previous_row))      
        for indx_ref, wrd_ref in enumerate(self.ref):
            small_value = 1e-7 if (wrd_ref.isupper() and self.m_weights) else 0
            wrd_ref = wrd_ref.lower()
            current_row = [(self.deletion - small_value) * (indx_ref + 1)] 
            for indx_hyp, wrd_hyp in enumerate(self.hyp):            
                deletions = previous_row[indx_hyp + 1] + (self.deletion - small_value)
                insertions =  current_row[indx_hyp] + (self.insertion + small_value)
                substitutions = previous_row[indx_hyp] + (
                    self.substitution * (wrd_ref != wrd_hyp) + small_value
                    ) 
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            rows.append(previous_row)
        return rows

    def backtrace(self):
        """
        Backtraces the cost matrix to find the minimum path for aligning 
        the reference and hypothesis strings.
        """
        rows = self.cost_matrix()            
        i, j = len(self.ref), len(self.hyp)        
        edits = list()    
        while(not (i==0 and j==0)):  
            prev_cost = rows[i][j] 
            neighbors = list()
            if (i!=0 and j!=0): neighbors.append(rows[i-1][j-1])
            if (i!=0): neighbors.append(rows[i-1][j])            
            if (j!=0): neighbors.append(rows[i][j-1])      
            small_value = 1e-7 if (self.ref[i-1].isupper() and self.m_weights) else 0   
            min_cost = min(neighbors) + small_value
            if (prev_cost==min_cost):
                i, j = i-1, j-1          
                edits.append(dict(
                    type='match', 
                    eval=''.ljust(len(self.ref[i])+1, ' '),
                    ref=self.ref[i], 
                    hyp=self.hyp[j]
                )
                )         
            elif (i!=0 and j!=0 and prev_cost==rows[i-1][j-1]+(self.substitution+small_value)):              
                i, j = i-1, j-1 
                ln_ref, ln_hyp = len(self.ref[i]), len(self.hyp[j])                
                edits.append(dict(
                    type='substitution', 
                    eval='S'.ljust(max(ln_ref, ln_hyp)+1, ' '),
                    ref=self.ref[i]+' '*(ln_hyp-ln_ref), 
                    hyp=self.hyp[j]+' '*(ln_ref-ln_hyp)
                )
                )                     
            elif (i!=0 and prev_cost==rows[i-1][j]+(self.deletion-small_value)) or (j==0):
                i = i-1
                ln_ref = len(self.ref[i])
                edits.append(dict(
                    type='deletion', 
                    eval='D'.ljust(ln_ref+1, ' '),
                    ref=self.ref[i], 
                    hyp='*'*ln_ref
                )
                )
            elif (j!=0 and prev_cost==rows[i][j-1]+(self.insertion+small_value)) or (i==0):               
                j = j-1 
                ln_hyp = len(self.hyp[j])
                edits.append(dict(
                    type='insertion', 
                    eval='I'.ljust(ln_hyp+1, ' '),
                    ref='*'*ln_hyp, 
                    hyp=self.hyp[j]
                )
                )     
            elif (prev_cost==rows[i-1][j-1]+small_value):
                i, j = i-1, j-1          
                edits.append(dict(
                    type='match', 
                    eval=''.ljust(len(self.ref[i])+1, ' '), 
                    ref=self.ref[i], 
                    hyp=self.hyp[j]
                )
                )
        edits.reverse()
        return edits

    def zero_length(self):
        """
        Handles empty hypothesis strings.
        """
        edits = list()
        for wrd_ref in self.ref:
            edits.append(dict(
                type='deletion', 
                eval='D'.ljust(len(wrd_ref)+1, ' '), 
                ref=wrd_ref, 
                hyp='*'*len(wrd_ref)
            )
            )
        return edits

    def total_score(self, edits):
        """
        Returns numbers of match, sub, del and ins for each sentence.
        """
        operations = list(op['type'] for op in edits)
        return [
            operations.count('match') ,
            operations.count('substitution'),
            operations.count('deletion'),
            operations.count('insertion')
        ]
        
    def region_score(self, edits):
        """
        Returns numbers of match, sub, del and ins for fluent and disfluent 
        regions, separately.
        """
        operations = list((op['type'], op['ref'].isupper()) for op in edits)
        return [
            operations.count(('match', False)),
            operations.count(('substitution', False)),
            operations.count(('deletion', False)),                
            operations.count(('insertion', False)),        
            operations.count(('match', True)),
            operations.count(('substitution', True)),
            operations.count(('deletion', True)),
            operations.count(('insertion', True))
        ]        

    def align(self):
        """
        Prints the best alignment b/w the reference and hypothesis strings.
        """    
        if not len(self.ref): 
            raise ValueError('Reference sentence cannot be an empty line!')
        edits = self.zero_length() if not len(self.hyp) else self.backtrace()   
        scores = self.region_score(edits) if self.m_weights else self.total_score(edits) 
        d_wrds = len(list(filter(lambda w: w.isupper(), self.ref))) # disfluent words
        f_wrds = len(self.ref)-d_wrds if self.m_weights else len(self.ref) # fluent words
        return 'REF: \t{}\n'.format(
                ' '.join(list(e['ref'] for e in edits))
            ) + 'HYP: \t{}\n'.format(
                ' '.join(list(e['hyp'] for e in edits))
            ) + 'Eval: \t{}\n'.format(
                ''.join(list(e['eval'].upper() for e in edits))
            ) + ('Fluent:    (#C #S #D #I) {} {} {} {} \nDisfluent: (#C #S #D #I) {} {} {} {}\n' \
                if self.m_weights else 'Scores: (#C #S #D #I) {} {} {} {}\n').format(*scores), \
            scores[:4] + [f_wrds] + scores[4:] + [d_wrds]

    def __str__(self):
        alignment, _ = self.align()
        return alignment

        
class Test():
    ref = [
            "THE THE THE the sad part about here is that winter starts in",
            "THEY YOU KNOW THEY they wouldn't have been given access to the friend or foe codes",
            "IT'S BEEN it's been lucratim for him", 
            "I I i guess he was RIGH-", 
            "YEAH yeah THEY THEY THEY they will check for alcohol",
            "makes you wanna stick around",
            "yeah",
            "right well I i'm on the benefits committee here",
            "yeah I i totally agree I i think it's kind of appalling",
            "they're pretty fun THEY'RE they're good kids",
            "I i mean i don't really know WHAT THEY CAN how they can really enforce the laws any better",
            "I THINK i think IT WAS IT WAS THERE WERE there were a lot more demographic related interestts",
            "I MEAN i mean it's great"
        ]

    hyp = [
            "the sad part about here is that winter starts in",
            "they wouldn't have been given access to the friend or foe codes",
            "it's been lucri for em", 
            "i guess he was wrong",
            "yeah i mean they check for alcohol",
            "makes you want to stick around",
            "",
            "right well i'm not i'm not the benefits committee here",
            "yeah i i i totally agree i think it's it's kind of a pollen",
            "they are pretty fun they were good kids",
            "i mean i don't really know they can really impose the laws any better",
            "i think there was a lot more geographic related interests",
            "i mean i i i mean it is great"
        ]
    def setUp(self, ref, hyp):
        align = list()
        for indx, sent in enumerate(zip(ref, hyp), start=1):
            min_dis = MinDistance(
                Ref=sent[0], 
                Hyp=sent[1], 
                m_weights=True
            )
            align.append("{1} \nSent #{0} \n{1} \n{2}".format(indx, '-'*50, min_dis))  
        return align
        
    def __str__(self):
        align = self.setUp(self.ref, self.hyp)
        return '\n'.join(align)