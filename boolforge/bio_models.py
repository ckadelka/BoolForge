#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides functionality for retrieving, parsing, and loading
biological Boolean network models from public online repositories.

The :mod:`~boolforge.bio_models` module allows users to programmatically access
and import published Boolean and logical gene regulatory network models from
GitHub repositories such as:
    - `expert-curated (ckadelka)` — manually curated models from the
      *Design Principles of Gene Regulatory Networks* repository.
    - `pystablemotifs (jcrozum)` — models accompanying the PyStableMotifs
      library.
    - `biodivine (sybila)` — models from the Sybila Biodivine Boolean Models
      repository.

Functions are provided to:
    * Recursively list and download files from GitHub folders using the REST API.
    * Fetch raw text or byte content from remote sources.
    * Parse Boolean network models directly into
      :class:`~boolforge.BooleanNetwork` objects.
    * Batch-download and convert all models from supported repositories.

All network parsing routines use pure Python and standard libraries only.
No external dependencies (such as Numba) are required for functionality, though
BoolForge’s Boolean network analysis methods can benefit from Numba-based JIT
acceleration when installed.

This module is intended to facilitate reproducible research by providing
direct access to real-world Boolean GRN models for simulation, comparison, and
benchmarking.

Example
-------
>>> from boolforge import bio_models
>>> bns, ok, failed = bio_models.get_bio_models_from_repository('expert-curated (ckadelka)')
>>> len(bns)
122
>>> bns[0].variables[:5]
['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
"""

import requests
import pickle
import io

from typing import Optional

try:
    from boolforge.boolean_network import BooleanNetwork
except ModuleNotFoundError:
    from boolean_network import BooleanNetwork

def _get_content_in_remote_folder(url : str, file_names : list, file_download_urls : list) -> None:
    """
    Recursively collect file names and download URLs from a remote GitHub folder.

    **Parameters:**
    
        - url (str): GitHub API URL pointing to a repository folder.
        - file_names (list): List to be filled with discovered file names.
        - file_download_urls (list): List to be filled with corresponding raw download URLs.
    """
    folder = requests.get(url)
    folder.raise_for_status()
    folder_json = folder.json()
    for item in folder_json:
        if item['size'] > 0 and item['download_url'] is not None:
            file_names.append(item['name'])
            file_download_urls.append(item['download_url'])
        else:
            try:
                url = item['url']
                _get_content_in_remote_folder(url, file_names, file_download_urls)
            except:
                pass


def get_content_in_remote_folder(url : str) -> tuple:
    """
    Retrieve file names and download URLs from a remote GitHub folder.

    **Parameters:**
    
        - url (str): GitHub API URL pointing to a repository folder.

    **Returns:**
    
        - tuple[list[str], list[str]]: (file_names, file_download_urls), where
          file_names is a list of files in the folder and file_download_urls
          is a list containing the raw download URL.
    """
    file_names = []
    file_download_urls = []  
    _get_content_in_remote_folder(url, file_names, file_download_urls)
    return file_names, file_download_urls


def fetch_file(download_url : str) -> str:
    """
    Download raw text content of a file.

    **Parameters:**
    
        - download_url (str): Direct download URL to the file.

    **Returns:**
    
        - str: File content as plain text.
    """
    r = requests.get(download_url)
    r.raise_for_status()
    return r.text


def fetch_file_bytes(download_url : str) -> bytes:
    """
    Download raw bytes content of a file.

    **Parameters:**
    
        - download_url (str): Direct download URL to the file.

    **Returns:**
    
        - bytes: File content as raw bytes.
    """
    r = requests.get(download_url)
    r.raise_for_status()
    return r.content


def load_model(download_url : str, max_degree : int = 24,
    possible_separators : list = ['* =','*=','=',','], original_not : str = 'NOT',
    original_and : str = 'AND', original_or : str= 'OR',
    IGNORE_FIRST_LINE : bool =False) -> Optional[BooleanNetwork]:
    """
    Load a Boolean network model from a remote text file.

    **Parameters:**
    
        - download_url (str): Direct download URL to the model file.
        - max_degree (int, optional): Maximum in-degree allowed for nodes
          (default: 24).
          
        - possible_separators (list[str], optional): Possible assignment
          separators in model files (default: ['\\* =', '\\*=', '=', ',']).
          
        - original_not (str, optional): Possible logical negation operator in
          the model file.
        
        - original_and (str, optional): Possible logical AND operator in the
          model file.
        
        - original_or (str, optional): Possible logical OR operator in the
          model file.
          
        - IGNORE_FIRST_LINE (bool, optional): If True, skip the first line
          of the file (default: False).

    **Returns:**
    
        - BooleanNetwork: Parsed Boolean network. If parsing fails, returns None.
    """
    string = fetch_file(download_url)
    if IGNORE_FIRST_LINE:
        string = string[string.index('\n')+1:]
    
    try:
        bn = BooleanNetwork.from_string(string, possible_separators, max_degree,
                                        original_not, original_and, original_or)
    except:
        bn = None
    return bn

download_urls_pystablemotifs = None

def get_bio_models_from_repository(repository : str) -> tuple:
    """
    Load Boolean network models from selected online repositories.

    **Parameters:**
    
        - repository (str:{'expert-curated (ckadelka)', 'pystablemotifs (jcrozum)',
          'biodivine (sybila)'}): Source repository identifier.

    **Returns:**
    
        - tuple[list[BooleanNetwork], list[str], list[str]]: (bns,
          successful_download_urls, failed_download_urls) where bns is a list
          of successfully parsed Boolean networks, successful_download_urls
          is a list of URLs of models successfully loaded, and
          failed_download_urls is a list of URLs where models could not be
          parsed.
    """
    repositories = ['expert-curated (ckadelka)', 'pystablemotifs (jcrozum)', 'biodivine (sybila)']
    bns = []
    successful_download_urls = []
    failed_download_urls = []

    if repository == 'expert-curated (ckadelka)':
        download_url_base = 'https://raw.githubusercontent.com/ckadelka/DesignPrinciplesGeneNetworks/main/update_rules_122_models_Kadelka_SciAdv/'
        download_url = download_url_base + 'all_txt_files.csv'
        csv = fetch_file(download_url)
        for line in csv.splitlines():
            download_url = download_url_base + line
            if '.txt' in download_url:
                if 'tabular' in download_url:
                    [F, I, var, constants] = pickle.load(io.BytesIO(fetch_file_bytes(download_url)))
                    for i in range(len(constants)):
                        F.append([0, 1])
                        I.append([len(var)+i])
                    bn = BooleanNetwork(F, I, var+constants)
                else:
                    bn = load_model(download_url, original_and = " AND ",
                                    original_or = " OR ", original_not = " NOT ")
                if bn is None:
                    failed_download_urls.append(download_url)
                else:
                    successful_download_urls.append(download_url)
                    bns.append(bn)

    elif repository == 'pystablemotifs (jcrozum)':
        if download_urls_pystablemotifs is None:
            url = "https://api.github.com/repos/jcrozum/pystablemotifs/contents/models"
            _, download_urls = get_content_in_remote_folder(url)
        else:
            download_urls = download_urls_pystablemotifs
        for download_url in download_urls:
            if '.txt' in download_url:
                bn = load_model(download_url,
                    possible_separators=['*    =','*   =', '*  =', '* =', '*='],
                    original_and = [" and ", "&"],
                    original_or = [" or ", "|"], original_not = [" not ", " !"])
                if bn is None:
                    failed_download_urls.append(download_url)
                else:
                    successful_download_urls.append(download_url)
                    bns.append(bn)

    elif repository == 'biodivine (sybila)':
        download_url_base = 'https://raw.githubusercontent.com/sybila/biodivine-boolean-models/main/models/'
        download_url = download_url_base + 'summary.csv'
        csv = fetch_file(download_url)
        for line in csv.splitlines():
            try:
                ID, name, variables, inputs, regulations = line.split(', ')
                download_url = download_url_base + (
                    '[id-%s]__[var-%s]__[in-%s]__[%s]/model.bnet'
                    % (ID, variables, inputs, name))
                bn = load_model(download_url, original_and = " & ",
                                original_or = " | ", original_not = "!",
                                IGNORE_FIRST_LINE=True)
                bns.append(bn)
                successful_download_urls.append(download_url)
            except:
                failed_download_urls.append(download_url)

    else:
        print('Error: repositories must be one of the following:\n - ' + '\n - '.join(repositories))

    return bns, successful_download_urls, failed_download_urls