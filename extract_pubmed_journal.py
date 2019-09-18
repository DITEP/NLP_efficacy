#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from Bio import Entrez
from Bio.Entrez import efetch
import re
import time


def pubmed_abstract(term):
    Entrez.email = "azertyuiopno@gmail.com"
    req_esearch = Entrez.esearch(db="pubmed", term=term, retmax=1000, usehistory="y")
    search_results = Entrez.read(req_esearch)
    count = search_results["Count"]
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]
    handle = efetch(db='pubmed', webenv=webenv, query_key=query_key, retmax=500)
    data = handle.read()
    resultats = re.findall('Pubmed-entry ::= {(.*?)},\n\s+status', data, re.DOTALL)
    abstracts = []
    journals = []
    for resultat in resultats:
        abstract = re.search("abstract \"([^\"]+\n)+[^\"]+\"", resultat)
        journal = re.search("issn \"(?:[0-9]+-[0-9]+)\",\n\s+name \"(?:[^\"]+)\"", resultat)
        if abstract and journal is not None:
            abstract = abstract.group(0)
            journal = journal.group(0)
            regex = {"abstract": "", "\"": "", "\n": ""}
            # use these three lines to do the replacement
            regex = dict((re.escape(k), v) for k, v in regex.items())
            pat = re.compile("|".join(regex.keys()))
            abstract = pat.sub(lambda m: regex[re.escape(m.group(0))], abstract)
            abstracts.append(abstract)
            regex = re.compile("issn \"(?:[0-9]+-[0-9]+)\",\n\s+name \"")
            journal = regex.sub("", journal)
            journal = journal.replace("\"", "")
            journals.append(journal.lower())
    return abstracts, count, journals


drugs = []
with open("antineoplastic_drugbank.txt", "r", encoding='utf-8') as filin:
    lignes = filin.readlines()
    for ligne in lignes:
        drugs.append(ligne)

rep = {"\n": ""}
# use these three lines to do the replacement
rep = dict((re.escape(k), v) for k, v in rep.items())
pattern = re.compile("|".join(rep.keys()))
medication = []
for drug in drugs:
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], drug)
    medication.append(text)

# load impact factor data and create a dictionary journal, impact factor
journal_dict = {}
impact_factor = pd.read_csv('datas\\2018JournalImpactFactor.csv', header=0, sep=';', encoding='utf-8')
impact_factor = impact_factor.reset_index(drop=True)
for i in range(len(impact_factor)):
    name = impact_factor['Full Journal Title'][i]
    i_f = impact_factor['JournalImpact'][i]
    journal_dict[name.lower()] = i_f

pretrain = pd.DataFrame({'Abstract': [],
                         'Journal': [],
                         'Impact Factor': [],
                         'Count': [],
                         'COMMON_DRUGBANK_ALIAS': []
                         })
timer = 0
for i, med in enumerate(medication):
    timer += 1
    drug_abstract, y, j = pubmed_abstract(med)
    print("{} / {} : {} {} abstracts / {} results".format(i, len(medication), med, len(drug_abstract), y))
    df = pd.DataFrame({'Abstract': drug_abstract})
    if_list = []
    for jour in j:
        try:
            if_list.append(journal_dict[jour])
        except KeyError:
            if_list.append(0)
    df["Journal"] = j
    df["Impact Factor"] = if_list
    df['Count'] = y
    df['COMMON_DRUGBANK_ALIAS'] = med
    pretrain = pd.concat([pretrain, df], axis=0, sort=False)
    if timer == 10:
        time.sleep(30)
    if timer == 50:
        time.sleep(180)
        timer = 0

print("Number of abstract {}".format(len(pretrain)))
pretrain.to_csv('pretrain_data3.txt', sep=';', encoding='utf-8')
