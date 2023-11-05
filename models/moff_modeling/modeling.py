#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Modular prediction of off-target effects
@author: Wei He
@E-mail: whe3@mdanderson.org
@Date: 05/31/2021

Ofir: cloned in 01.07.2022 - modified the code to my requires
"""

# ################### Import all the packages #####################
from __future__ import division
import json
import os
import pkg_resources
import numpy as np
from itertools import product
from itertools import combinations
from scipy.stats import gmean


RequiredFilePath = pkg_resources.resource_filename(__name__, 'StaticFiles')


def OneHotEndocing(sg_ls):
    '''
    This function is to encode the sgRNA sequences into (16,19) vector with 0,1 presentation of
    certain dinucleotide at certain position.

    o input:  sg_ls: A list of sgRNAs for one-hot encoding

    o Output: The function will return a numpy array of feature matrix for machine learning.
    '''
    di_ls = [s[0]+s[1] for s in list(product('ATGC', repeat=2))]  # Get all possible di-nucleotide combinations
    di_dic = {}
    for i in range(len(di_ls)):
        di_dic[di_ls[i]] = i  # Record the index of each dinucleotide

    ls_all = []  # Initialize a list to add vector for different sgRNA
    for sg in sg_ls:
        vec_all = []
        for i in range(len(sg)-1):
            vec = [0]*len(di_ls)  # Make all position to be 0
            di = sg[i:i+2]
            vec[di_dic[di]] = 1   # Assign 1 if certain dinucleotide appear at certain position
            vec_all.append(vec)

        ls_all.append(np.array(vec_all).T)

    return np.array(ls_all)


def GetMutType(s1, s2):
    '''
    This function is obtain the mismatches between sgRNA and the target

    o input:  1). s1: the sequence of sgRNA; 2). s2: the sequence of target DNA

    o Output: The function will return: 1). A list of positions where mismatch happen.
                                        2). A list of mismatch types at each position.
    '''
    pos_ls = []
    mut_ls = []
    for i in range(20):  # Go through the index along the 20bp sequence
        r = ''
        d = ''
        if s1[i] != s2[i]:
            pos = 20-i  # The index relative to PAM
            r = 'U' if s1[i] == 'T' else s1[i]  # Replace 'T' with 'U' in sgRNA.

            # Get mutation type given the nt at sgRNA and target
            if s2[i] == 'A':
                d = 'T'
            elif s2[i] == 'T':
                d = 'A'
            elif s2[i] == 'C':
                d = 'G'
            elif s2[i] == 'G':
                d = 'C'
            elif s2[i] == 'N':
                d = s1[i]
            pos_ls.append(pos)
            mut_ls.append('p'+str(pos)+'r'+r+'d'+d)  # p3rAdC: mismatch A-G at index 3 to PAM
    return pos_ls, mut_ls


def Multiply(m1_dic, sg_ls, tg_ls):
    '''
    This function is Calculate the off-target effect by multiplying the MDE at each position

    o input:1). m1_dic: Python dic contains MDE of all the possible nucleotide mismatches (12)
                at all possible positions (20)
            2). sg_ls: A list of sgRNAs
            3). tg_ls: A list of DNA targets

    o Output: A list of calculated mismatch-dependent effect.
    '''
    me_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()
        # print (s1,s2)
        mut_ls = GetMutType(s1, s2)[1]
        score = 1
        for mut in mut_ls:  # Multiply all the 1-mismatch effects
            score = score*m1_dic[mut]  # m1_dic: dic file
        me_ls.append(score)

    return me_ls


def MisNum(sg_ls, tg_ls):
    '''
    This function is to get mismatch numbers of gRNA-target pairs
    '''
    num_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()

        num = len(GetMutType(s1, s2)[0])
        num_ls.append(num)

    return num_ls


def MisType(sg_ls, tg_ls):
    tp_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()

        tp = '|'.join(GetMutType(s1, s2)[1])
        tp_ls.append(tp)

    return tp_ls


def CombineGM(m2_dic, sg_ls, tg_ls):
    '''
    This function is Calculate Combinatorial effect (CE) for given mismatch positions

    o input:1). m2_dic: Python dic contains CE of all the possible position combinaitons
            2). sg_ls: A list of sgRNAs
            3). tg_ls: A list of DNA targets

    o Output: A list of calculated combinatorial effects.
    '''
    cm_ls = []
    for i in range(len(sg_ls)):
        s1 = sg_ls[i][0:20].upper()
        s2 = tg_ls[i][0:20].upper()
        pos_ls = sorted(GetMutType(s1, s2)[0])

        # Combinatorial effect at certain position combination.
        di_ls = list(combinations(pos_ls, 2))
        c_ls = [m2_dic[str(di[0])+'&'+str(di[1])] for di in di_ls]

        if len(pos_ls) > 1:
            m = gmean(c_ls)**(len(pos_ls)-1)  # Geometirc mean of all possible combinations
        else:
            m = 1
        cm_ls.append(m)
    return cm_ls


def GMT_score(df):
    from keras import models

    sg_ls = list(df['sgRNA'])  # Get list of input sgRNAs

    np.random.seed(24)  # for reproducibility
    model = models.load_model(os.path.join(RequiredFilePath, 'GOP_model_3.h5'))
    pred_test = list(model.predict(OneHotEndocing([s.upper().replace("-", "")[0:20] for s in sg_ls])))
    df['GOP'] = [g[0] for g in pred_test]

    return df


def MOFF_score(df):
    '''
    This function is predict off-target MOFF score for given gRNA-target pairs

    o input:1). df: A panda dataframe with one column of sgRNA and another column of DNA targets

    o Output: A panda dataframe with off-target predictions using different models (factors)
    '''
    from keras import models

    # m1_dic: Python dic contains MDE of all the possible nucleotide mismatches (12) at all possible positions (20)
    m1_data = os.path.join(RequiredFilePath, 'M1_matrix_dic_D9')
    m1_dic = json.loads(open(m1_data).read())
    # m2_dic: Python dic contains CE of all the possible position combinaitons (20*19)
    m2_data = os.path.join(RequiredFilePath, 'M2_matrix_smooth_MLE')
    m2_dic = json.loads(open(m2_data).read())

    sg_ls = list(df['sgRNA'])  # Get list of input sgRNAs
    tg_ls = list(df['off-target'])  # Get list of input DNA targets

    np.random.seed(24)  # for reproducibility
    model = models.load_model(os.path.join(RequiredFilePath, 'GOP_model_3.h5'))
    pred_test = list(model.predict(OneHotEndocing([s.upper()[0:20] for s in sg_ls])))
    df['GOP'] = [g[0] for g in pred_test]

    df['MDE'] = Multiply(m1_dic, sg_ls, tg_ls)
    df['CE'] = CombineGM(m2_dic, sg_ls, tg_ls)
    df['MMs'] = MisNum(sg_ls, tg_ls)
    df['MisType'] = MisType(sg_ls, tg_ls)
    df['GMT'] = df['GOP']**df['MMs']
    df['MOFF'] = df['MDE']*df['CE']*df['GMT']
    return df
