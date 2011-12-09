# -*- coding: utf-8 -*-

"""
Simple implementations of routines for
computing IBM model 1

@var tprobs: Conditional translation probs.  tprobs[(f_word,e_word)]=
                t(f_word|e_word)
@var e_dict: word counts for english words


FIXME: it has an error: a(i, j, I, J) ---> a(i, j, J, I)

"""

# Prob of aligned sent. e of length k
epsilon = 1.0
# compute num possible alignments
### Line No 4
en_ex = "i see the small dog"
de_ex = "ich sehe das klein hund"
en_words = en_ex.split()
de_words = de_ex.split()



class Em_aligner (object):
    
    def __init__(self,iterations,f_file,e_file, use_null=True):

        self.iterations = iterations
        self.f_file = f_file
        self.e_file = e_file
        self.f_vocab = {}
        self.total = {}
        self.total_d = {}
        self.a = {}
        self.tprobs = {}
        self.all_rounds_dict = {}
        self.tcounts = {}
        self.d_counts = {}
        self.use_null = use_null

    def compute_em_mod1 (self, verbose=False):
        # Initializes all Model 1 parameters
        self.read_parallel_corpus ()
        print 'Model 1'
        print '----------------------------------'
        if verbose:
            print '         %s\t%s\tProb' % ("f", "e")
            print '         -------------------------'
        rnd = 0
        # print_tprobs(round,tprobs)
        while self.iterations > 0:
            print_tprobs(rnd,self.tprobs,verbose)
            if rnd >= 1 and verbose == True:
                print_tcounts(rnd,self.tcounts,self.total)
            rnd += 1
            print 'Round %d' % (rnd,)            
            self.compute_em_round_slick()
            self.all_rounds(rnd,self.tprobs)
            self.iterations -= 1
            
    def compute_em_mod2 (self, verbose=False):
        '''

            FIXME: it has an error: a(i, j, I, J) ---> a(i, j, J, I)
        '''
        print 'Model 2'
        print '----------------------------------'
        if verbose:
            print '         %s\t%s\tProb' % ("f", "e")
            print '         -------------------------'
        rnd = 0
        # print_tprobs(round,tprobs)
        self.iterations = 20;
        while self.iterations > 0:
            print_tprobs(rnd,self.tprobs,verbose)
            if rnd >= 1 and verbose == True:
                print_tcounts(rnd,self.tcounts,self.total)
            rnd += 1
            print 'Round %d' % (rnd,)
            self.compute_em2_round_slick()
            self.all_rounds(rnd,self.tprobs)
            self.iterations -= 1
        #print self.a
            
    def read_parallel_corpus (self):
        """
        Initialization stuff for Model 1 em.
        Create vocab sets for both languages.
        Initialize tran dict to uniform probs.
        """
        f_fh = open(self.f_file,'r')
        e_fh = open(self.e_file,'r')
        print 'Reading corpus and initializing'
        self.tprobs.clear()
        self.tcounts.clear()
        self.total.clear()
        self.f_vocab.clear()
        alt_tprobs = {}
        self.max_source = 0
        self.max_target = 0
        self.max_ratio = 1.0
        ctr = 0
        for e_line in e_fh:
            ctr += 1
            f_line = f_fh.readline()
            e_words=e_line.split()[0:42]
            if self.use_null:
                e_words = ['NULL'] + e_words
            update_vocab(e_words,self.total)
            f_words=f_line.split()[0:42]


            self.max_source = max(self.max_source,len(f_words))
            self.max_target = max(self.max_target,len(e_words))
            self.max_ratio = max(self.max_ratio,float(len(e_words))/len(f_words))
            
            update_vocab(f_words,self.f_vocab)
            for f_word in f_words:
                for e_word in e_words:
                    ## Initialize with seen pairs
                    self.tprobs[(f_word,e_word)] = 0
                    e_dict =alt_tprobs.setdefault(e_word,{})
                    e_dict[f_word] = 0
                    self.tcounts[(f_word,e_word)] = 0

        l_e = 43;
        l_f = 43;
        for I in xrange(0, l_e):
            u = 1.0 / (I+1)
            for J in xrange(0, l_f):
                for j in xrange(0, J):
                    self.total_d[(j, I, J)] = 0.0
                    for i in xrange(0, I):
                        self.a[(i, j, I, J)] = u
                        self.d_counts[(i, j, I, J)] = 0.0
        f_fh.close()
        e_fh.close()
        self.corpus_size = ctr
        ## Initialize with uniform distribution
        normalization = len(self.f_vocab)
        for (f_word,e_word) in self.tprobs:
            ## Sum t(f_i|e) = 1
            # Just use the set of pairs seen for this word
            # normalization = len(alt_tprobs[e_word])
            u_prob = 1.0/normalization
            self.tprobs[(f_word,e_word)] = u_prob
        self.all_rounds(0,self.tprobs)
        # keep a copy of the inital e_word_freqs around for
        # fertility mixture computations in Rd. 3
        self.e_word_freq = {}
        self.e_word_freq.update(self.total)

    def compute_em2_round_slick(self):
        """
        Assuming total and tprobs have been
        initlialized or have vals from previous rounds.

        FIXME: it has an error: a(i, j, I, J) ---> a(i, j, J, I)
        """
        # Zero out counts from previous rounds
        for e_word in self.total:
            self.total[e_word] = 0
        for tpair in self.tcounts:
            self.tcounts[tpair] = 0
        f_fh = open(self.f_file,'r')
        e_fh = open(self.e_file,'r')
        for e_line in e_fh:
            f_line = f_fh.readline()
            e_sentence = e_line.split()[0:42]
            if self.use_null:
                e_sentence=['NULL'] + e_sentence
            f_sentence=f_line.split()[0:42]
            total_s = {}
            l_e = len(e_sentence)
            l_f = len(f_sentence)
            for (f, f_word) in enumerate(f_sentence):
                total_s[f_sentence[f]] = 0;
                for (e, e_word) in  enumerate(e_sentence):
                    total_s[f_word] += float(self.tprobs[(f_word, e_word)]) * self.a[(e, f, l_e, l_f)]
                for (e, e_word) in  enumerate(e_sentence):
                    c = 1.0 * self.tprobs[(f_word, e_word)] * self.a[(e, f, l_e, l_f)] / total_s[f_word]
                    self.tcounts[(f_sentence[f], e_sentence[e])] += c
                    self.d_counts[(e, f, l_e, l_f)] += c
                    self.total[e_word] += c
                    self.total_d[(f, l_e, l_f)] += c
        
        self.smooth_distortion_counts ()
        for tpair in self.tcounts:
            (f_word,e_word)  = tpair
            self.tprobs[tpair] = float(self.tcounts[tpair]) / self.total[e_word]
            
        for ttetra in self.d_counts:
            (i, j, I, J)  = ttetra
            if(self.total_d[(j, I, J)]):
                self.a[ttetra] = float(self.d_counts[ttetra]) / self.total_d[(j, I, J)]
            
        f_fh.close()
        e_fh.close()

    def smooth_distortion_counts (self):
        laplase = 1.0
        for ttetra in self.d_counts:
            val = self.d_counts[ttetra]
            if(val > 0) and (val < laplase):
                laplase = val
        laplase = 0.5 * laplase
        for ttetra in self.d_counts:
            self.d_counts[ttetra] += laplase;
        for ttetra in self.d_counts:
            (i, j, I, J)  = ttetra
            if(self.total_d[(j, I, J)]):
                self.a[ttetra] = float(self.d_counts[ttetra]) / self.total_d[(j, I, J)]
        for ttri in self.total_d:
            (j, I, J)  = ttri
            self.total_d[ttri] += laplase * J

    def compute_em_round_slick(self):
        """
        Assuming total and tprobs have been
        initlialized or have vals from previous rounds.
        """
        # Zero out counts from previous rounds
        for e_word in self.total:
            self.total[e_word] = 0
        for tpair in self.tcounts:
            self.tcounts[tpair] = 0
        f_fh = open(self.f_file,'r')
        e_fh = open(self.e_file,'r')
        for e_line in e_fh:
            f_line = f_fh.readline()
            e_sentence = e_line.split()[0:42]
            if self.use_null:
                e_sentence=['NULL'] + e_sentence
            f_sentence=f_line.split()[0:42]
            total_s = {}
            for f_word in f_sentence:
                total_s[f_word] = 0
                for e_word in e_sentence:
                    total_s[f_word] += self.tprobs[(f_word,e_word)]
                for e_word in e_sentence:
                    tpair = (f_word,e_word)
                    self.tcounts.setdefault(tpair,0)
                    self.tcounts[tpair] += float(self.tprobs[(f_word,e_word)])/total_s[f_word]
                    self.total[e_word] += self.tprobs[(f_word,e_word)]/total_s[f_word]
                    # print '%s %.6f %.6f' % (tpair, tcounts[tpair], tprobs[tpair])
                    # print e_word, 'count', total[e_word]
                    # With the entire corpus recounted, revise tprobs
        for tpair in self.tprobs:
            (f_word,e_word)  = tpair 
            self.tprobs[tpair] = float(self.tcounts[tpair])/self.total[e_word]
            print 'tprobs = %.6f tcounts = %.6f' % (self.tprobs[tpair], self.tcounts[tpair])
        f_fh.close()
        e_fh.close()

    def all_rounds(self,rd,tprobs):
        rounds_dict = self.all_rounds_dict
        for (k,val) in tprobs.iteritems():
            k_vals = rounds_dict.setdefault(k,[])
            k_vals.append(val)

    def print_all_rounds (self):
        rounds_dict = self.all_rounds_dict
        num_rounds = len(rounds_dict[rounds_dict.keys()[0]])
        header_rt = '%-10s' * (num_rounds + 2)
        print header_rt % tuple(['',''] + range(num_rounds))
        for (pair,val_list) in rounds_dict.iteritems():
            val_list = tuple(['%.4f' % val for val in val_list])
            print_tuple = pair + val_list
            print header_rt % print_tuple
        


############################################################
def compute_em_round_naive(f_file, e_file, tprobs, tcounts,\
                          e_vocab, use_null=True):
    """
    Assuming e_vocab and tprobs have been
    initialized or have vals from previous rounds.
    """
    # Zero out counts from previous rounds
    for e_word in e_vocab:
        e_vocab[e_word] = 0
    for tpair in tcounts:
        tcounts[tpair] = 0
    f_fh = open(f_file,'r')
    e_fh = open(e_file,'r')
    for e_line in e_fh:
        f_line = f_fh.readline()
        e_words = e_line.split()
        if use_null:
            e_words=['NULL'] + e_words
        f_words=f_line.split()
        len_al = len(f_words)
        al_probs = {}
        all_alignments = compute_all_alignments(f_words,e_words)
        # initialize p(f | e) dist for this e
        prob_f_words = 0
        # Compute all unnormalized alignment probs & p(f | e)
        for al in all_alignments:
            t_prob = 1.0
            # print
            # compute this alignments prob
            for ind in range(len_al):
                tpair = get_tpair(al,ind,f_words,e_words)
                # print tpair,
                # print tprobs[tpair],
                t_prob *= tprobs[tpair]
            # print t_prob
            al_probs[al] = t_prob
            prob_f_words += t_prob
        # Normalize alignment probs
        for al in all_alignments:
            al_probs[al] = float(al_probs[al])/prob_f_words
        # update weighted counts
        for al in all_alignments:
            for ind in range(len_al):
                tpair = get_tpair(al,ind,f_words,e_words)
                al_prob =  al_probs[al]
                # next line prob unnecessary
                tcounts.setdefault(tpair,0)
                tcounts[tpair] += al_prob
                e_vocab[tpair[1]] += al_prob
    # Wih the entire corpus recounted, revise tprobs
    for (f_word,e_word) in tprobs:
        tprobs[(f_word,e_word)] = \
                            float(tcounts[(f_word,e_word)])/e_vocab[e_word]
    f_fh.close()
    e_fh.close()



def get_tpair (al,ind,f_words,e_words):
    return (f_words[ind],e_words[al[ind]])

only_one_one = True
def compute_all_alignments (fr,en):
    """
    Assuming C{fr} and C{en} are sequences of words,
    with NULL as the first elem of C{en}, if NULL used.
    """
    try:
        only_one_one == True
        if len(en) < len(fr):
            raise Exception, 'Cant do one-one alignment on shorter target'
        return [L[:len(fr)] for L in perm(range(len(en)))]
    except  NameError:
        return all_fns(fr, en)
    return all_fns(fr, en)

def all_fns (domain,rng):
    res = [()]
    rng_len = len(rng)
    for elem in domain:
        res = sprinkle(rng_len,res)
    return res

def sprinkle (rng,fns):
    res=[]
    for mpg in fns:
        for ind in range(rng):
            res += [(ind,) + mpg]
    return res

def perm(l):
    """Compute the list of all permutations of l"""
    if len(l) <= 1:
        return [l]
    r = []
    for i in range(len(l)):
        s = l[:i] + l[i+1:]
        p = perm(s)
        for x in p:
            r.append(tuple(l[i:i+1] + x))
    return r

def print_alignment (al,fr,en):
    for ind in range(len(al)):
        print '%s <->  %s' % (fr[ind],en[al[ind]]),
    print
    print 'Unaligned f: ',
    for ind in range(len(fr)):
        if al[ind] == 0: print fr[ind]
    print
    print 'Unaligned e: ',
    for ind in range(len(en)):
        if ind not in al: print en[ind]
    

def print_alignments(als, fr,en):
    print fr
    print en
    for al in als:
        print_alignment(al,fr,en)
        print '---'

def print_tprobs (round, tprobs,verbose):
    if verbose:
        print 'Round %d ' % (round,)
        for (f_word, e_word) in tprobs:
            print '        %s\t%s\t%.4f' % (f_word,e_word, tprobs[(f_word,e_word)])


def print_tcounts(rnd,tcounts,e_vocab):
    print
    print '        Counts:'
    print '        ------'
    for (f_word, e_word) in tprobs:
        print '        %s\t%s\t%.4f' % (f_word,e_word, tcounts[(f_word,e_word)])
    print
    for (e_word) in e_vocab:
        print '        %s\t%.4f' % (e_word, e_vocab[e_word])
    
def update_vocab (words,vocab):
    for word in words:
        vocab.setdefault(word,0)
        vocab[word] += 1

####################################################################
###
###   C o m p u t i n g   U t i l i t i e s   f o r  A l i g n m e n t s
###
####################################################################
        

def num_alignments (len_fr,len_en):
    """
    Alignment fns go from foreign positions to english positions
       ad[j] = i
    j ranges from 0 to len_fr [len_fr possible values]
    i ranges from 0 to len_en (if there is a NULL
    that is already included in len_en).

    We assuming each [foreign] word comes from one of
    len_en English words.
    See J&M, 24.5 approx. p. 25.

    So we return the number of possible functions from
    a domain of cardinality len_fr to a range of len_en+1.
    """
    return len_en**len_fr

def prob_fr_align_given_en (en,fr, ad):
    """
    Compute p(f,a|e) as in ibm model 1.
    
    en and fr are sequences of english and foreign words respect.

    ad is a dictionary whose keys are frn word posiiton
    and whose values are en word positions.

    return official ibm1 model 1 prob for p(f,a | e)

    tprobs is a globally available dictionary with all
    the condiitonal translation probs: p(f|e)

    @ rtype: float
    """
    tran_prod = 1.0
    for f_ind in ad:
        tran_prob = tprobs.setdefault((fr[f_ind],en[ad[f_ind]]),0)
        tran_prod *= tran_prob
    return epsilon * num_alignments(len(en),len(fr))**(-1) * tran_prod


def read_word_to_word_alignments (f_file, e_file, a_file, tprobs={}, e_dict={},a_dict={}):
    """
    Given three files, C{e}(nglish) file, C{f}(oreign) file and
    C{a}(lignment) file, fill in tprobs for all (e_word, f_word) pairs
    and e_dict containing counts for all english words.

    a_dict: dict of dicts, keys an int representing a line number,
            returning an alignment dict ad for each line: ad[f]=e
    """
    f_fh = open(f_file,'r')
    e_fh = open(e_file,'r')
    a_fh = open(a_file,'r')
    line_no=0
    tprobs.clear()
    e_dict.clear()
    a_dict.clear()
    for e_line in e_fh:
        line_no+=1
        f_line = f_fh.readline()
        a_line = a_fh.readline()
        e_words=e_line.split()
        f_words=f_line.split()
        print e_words
        print f_words
        print a_line
        ad = dict([(int(f_ind),int(e_ind)) for (f_ind,e_ind) in \
                   [p.split('-') for p in a_line.split()]])
        a_dict[line_no]=ad        
        for (f_word_ind,e_word_ind) in ad.iteritems():
            (f_word,e_word) = (f_words[f_word_ind],e_words[e_word_ind])
            tprobs.setdefault((f_word,e_word),0)
            tprobs[(f_word,e_word)] += 1
            e_dict.setdefault(e_word,0)
            e_dict[e_word] += 1
    f_fh.close()
    e_fh.close()
    a_fh.close()
    for (f_word,e_word) in tprobs:
        tprobs[(f_word,e_word)]= float(tprobs[(f_word,e_word)])/e_dict[e_word]

def print_file(filename):
    fh = open(filename,'r')
    for line in fh:
        print line.rstrip()

if __name__ == '__main__':
    tprobs = {}
    tcounts = {}
    total = {}
    f_vocab = {}
    e_dict = {}
    a_dict = {}
    # read_word_to_word_alignments('korp.es-en.es','korp.es-en.en','korp.es-en',tprobs,editc, a_dict)
    # read_parallel_corpus('korp.de-en.de','korp.de-en.en',tprobs,tcounts, fr_vocab, e_vocab)
    # em = Em_aligner(5,'corpus_dir/koehn_korp.de','corpus_dir/koehn_korp.en',False)
    em = Em_aligner(20,'t.ru','t.en',False)
    em.compute_em_mod1(True)
    #em.compute_em_mod2(True)

    # em_algorithm(5, 'korp2.de-en.de','korp2.de-en.en',tprobs,tcounts, e_vocab, False)

