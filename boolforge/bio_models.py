#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:48:48 2025

@author: ckadelka
"""

import requests
import boolforge

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

url = "https://api.github.com/repos/jcrozum/pystablemotifs/contents/models"
file_names,file_download_urls = get_content_in_remote_folder(url)
for name,download_url in zip(file_names,file_download_urls):
    print(name)
    string = fetch_file(download_url)
    print(string)
    try:
        bn = boolforge.BooleanNetwork.from_bnet(string,separator='*=')
    except boolforge.CustomError:
        bn = boolforge.BooleanNetwork.from_bnet(string,separator='* =')
    print(len(bn.F),len(bn.I),len(bn.variables))
    print()
