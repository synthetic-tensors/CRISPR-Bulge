import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

bases = 'ACGT'
bases_and_deletion = bases + '-'
bases_set = set(bases)


def parse_single_effects(fpath):
    pampwm, subpen, subtrans, delpen, inspen, insweight = [eval(line) for line in open(fpath)]
    pampwm = np.array(pampwm)
    subtrans = np.array(subtrans)
    return pampwm, subpen, subtrans, delpen, inspen, insweight


cas9_prots = [
    'WT',
    'Enh',
    'Hypa',
    'HF1',
]
cas9_lower_prots = [prot.lower() for prot in cas9_prots]

cas9_params = [
    parse_single_effects(os.path.join(THIS_DIR, 'params', '{}_single_effects_params.txt'.format(prot.lower())))
    for prot in cas9_prots
]
cas12a_params = parse_single_effects(os.path.join(
    THIS_DIR,
    'params',
    'cas12a_single_effects_params.txt'
))

all_prots = cas9_prots + ['Cas12a']
all_lower_prots = [prot.lower() for prot in all_prots]
all_params = cas9_params + [cas12a_params]
params_given_prot = {prot.lower(): params for prot, params in zip(all_prots, all_params)}

pam_given_prot = {prot.lower(): 'TGG' for prot in cas9_prots}
pam_given_prot['cas12a'] = 'TTTA'

first_time_point = 12
last_time_point = 60000
first_time_point_cleavage_rate = np.log(2) / first_time_point
last_time_point_cleavage_rate = np.log(2) / last_time_point
log10_ub = np.log10(first_time_point_cleavage_rate)
log10_lb = np.log10(last_time_point_cleavage_rate)
data_span = log10_ub - log10_lb


def bandpass_hinge(x):
    return max(log10_lb, min(x, log10_ub))


def single_effects(pam_cro, sub_cro, del_cro, ins_cro,
                   pampwm, subpen, subtrans, delpen, inspen, insweight):
    score = log10_ub
    for i, ri, oi in pam_cro:
        score += pampwm[oi][i]
    for i, ri, oi in sub_cro:
        score += subpen[i] * subtrans[ri, oi]
    for i, ri, oi in del_cro:
        score += delpen[i]
    for i, ri, oi in ins_cro:
        score += inspen[i] * insweight[oi]
    return bandpass_hinge(score) - log10_ub


def build_cro(ref_seq, obs_seq):
    sub_cro, del_cro, ins_cro = [], [], []
    i = 0
    for rb, ob in zip(ref_seq, obs_seq):
        if rb != ob:
            ri = bases_and_deletion.index(rb)
            oi = bases_and_deletion.index(ob)
            cro = (i, ri, oi)
            if rb in bases and ob in bases:
                sub_cro.append(cro)
            elif rb == '-':
                ins_cro.append(cro)
            elif ob == '-':
                del_cro.append(cro)
            else:
                raise ValueError('Unexpected input: ({}, {}, {})'.format(i, rb, ob))
        if rb != '-':
            i += 1
    return sub_cro, del_cro, ins_cro


def log10_crispr_specificity(prot, pam_seq, grna_seq, nts_dna_seq):
    """
    The main model function.

    Input:
        prot        :str: The protein to model. Options: WT, Enh, Hypa, HF1, Cas12a
        pam_seq     :str: The NTS PAM sequence with not indels
        grna_seq    :str: The aligned, 5'->3' guide RNA sequence, with insertions as hyphens
        nts_dna_seq :str: The aligned, 5'->3' NTS DNA target sequence (no PAM), with deletions as hyphens
    """
    pam_seq = pam_seq.upper()
    grna_seq = grna_seq.upper().replace('U', 'T')
    nts_dna_seq = nts_dna_seq.upper()

    prot = prot.lower()
    if prot not in all_lower_prots:
        raise ValueError('No model for protein {}'.format(prot))
    if prot in cas9_lower_prots:
        grna_seq = grna_seq[::-1]
        nts_dna_seq = nts_dna_seq[::-1]
    params = params_given_prot[prot]

    ref_pam = pam_given_prot[prot]
    if (not set(pam_seq) <= bases_set) or len(pam_seq) != len(ref_pam):
        raise ValueError('Invalid PAM: {}'.format(pam_seq))

    pam_cro, pam_del_cro, pam_ins_cro = build_cro(ref_pam, pam_seq)
    if pam_del_cro or pam_ins_cro:
        raise ValueError('Invalid PAM: {}'.format(pam_seq))

    simple_rna = grna_seq.replace('-', '')
    simple_dna = nts_dna_seq.replace('-', '')
    if not set(simple_rna + simple_dna) <= bases_set:
        raise ValueError('Invalid input sequences:\n{}\n{}'.format(grna_seq, nts_dna_seq))
    sub_cro, del_cro, ins_cro = build_cro(grna_seq, nts_dna_seq)

    return single_effects(pam_cro, sub_cro, del_cro, ins_cro, *params)
