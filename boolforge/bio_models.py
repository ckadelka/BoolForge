#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:48:48 2025

@author: ckadelka
"""

import requests
import boolforge
import pickle
import io


def _get_content_in_remote_folder(url,file_names,file_download_urls):
    folder = requests.get(url)
    folder.raise_for_status()
    folder_json = folder.json()
    for item in folder_json:
        if item['size']>0 and item['download_url'] is not None:
            file_names.append(item['name'])
            file_download_urls.append(item['download_url'])
        else:
            try:
                url = item['url']
                _get_content_in_remote_folder(url,file_names,file_download_urls)
            except:
                pass

def get_content_in_remote_folder(url):
    file_names = []
    file_download_urls = []  
    _get_content_in_remote_folder(url,file_names,file_download_urls)
    return file_names,file_download_urls

def fetch_file(download_url):
    """Fetch raw text content of a file."""
    r = requests.get(download_url)
    r.raise_for_status()
    return r.text

def fetch_file_bytes(download_url):
    r = requests.get(download_url)
    r.raise_for_status()
    return r.content  # return bytes, not str

def load_model(download_url, max_degree=24, possible_separators=['* =','*=','=',','],original_not = 'NOT', original_and  = 'AND', original_or = 'OR'):
    string = fetch_file(download_url)
    for separator in possible_separators:
        for (original_not,original_and,original_or) in [('not','and','or'),('NOT','AND','OR'),('!','&','|')]:
            try:
                bn = boolforge.BooleanNetwork.from_string(string,separator=separator,
                                                          max_degree=max_degree,
                                                          original_not=original_not,
                                                          original_and=original_and,
                                                          original_or=original_or)
                return bn
            except:
                pass 
    

url = 'https://api.github.com/repos/ckadelka/DesignPrinciplesGeneNetworks/contents/update_rules_122_models_Kadelka_SciAdv/'
#url = "https://api.github.com/repos/jcrozum/pystablemotifs/contents/models"
file_names,file_download_urls = get_content_in_remote_folder(url)
for i,(name,download_url) in enumerate(zip(file_names,file_download_urls)):
    if '.txt' in name:
        if 'tabular' in name:
            [F, I, var, constants] = pickle.load(io.BytesIO(fetch_file_bytes(download_url)))
            for i in range(len(constants)):
                F.append([0,1])
                I.append([len(var)+i])
            bn = boolforge.BooleanNetwork(F,I,var+constants)
        else:
            #print(name)
            bn = load_model(download_url)
        if bn is not None:
            print(i, len(bn.F),len(bn.I),len(bn.variables))
            print()
        else:
            print(i,name,'Transformation Error')
            print()

