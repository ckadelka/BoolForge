#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:48:48 2025

@author: ckadelka
"""

import requests
import pickle
import io

try:
    from boolforge.boolean_network import BooleanNetwork
except ModuleNotFoundError:
    from boolean_network import BooleanNetwork

def _get_content_in_remote_folder(url, file_names, file_download_urls):
    """
    Recursively collect file names and download URLs from a remote GitHub folder.

    Parameters
    ----------
    url : str
        GitHub API URL pointing to a repository folder.
    file_names : list
        List to be filled with discovered file names.
    file_download_urls : list
        List to be filled with corresponding raw download URLs.
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


def get_content_in_remote_folder(url):
    """
    Retrieve file names and download URLs from a remote GitHub folder.

    Parameters
    ----------
    url : str
        GitHub API URL pointing to a repository folder.

    Returns
    -------
    file_names : list of str
        Names of files in the folder.
    file_download_urls : list of str
        Corresponding raw download URLs.
    """
    file_names = []
    file_download_urls = []  
    _get_content_in_remote_folder(url, file_names, file_download_urls)
    return file_names, file_download_urls


def fetch_file(download_url):
    """
    Download raw text content of a file.

    Parameters
    ----------
    download_url : str
        Direct download URL to the file.

    Returns
    -------
    str
        File content as plain text.
    """
    r = requests.get(download_url)
    r.raise_for_status()
    return r.text


def fetch_file_bytes(download_url):
    """
    Download raw bytes content of a file.

    Parameters
    ----------
    download_url : str
        Direct download URL to the file.

    Returns
    -------
    bytes
        File content as raw bytes.
    """
    r = requests.get(download_url)
    r.raise_for_status()
    return r.content


def load_model(download_url, max_degree=24,
               possible_separators=['* =','*=','=',','],
               original_not='NOT', original_and='AND', original_or='OR',
               IGNORE_FIRST_LINE=False):
    """
    Load a Boolean network model from a remote text file.

    Parameters
    ----------
    download_url : str
        Direct download URL to the model file.
    max_degree : int, optional
        Maximum in-degree allowed for nodes (default: 24).
    possible_separators : list of str, optional
        Possible assignment separators in model files (default: ['\* =','\*=','=',',']).
    original_not, original_and, original_or : str, optional
        Possible logical operators in the model file.
    IGNORE_FIRST_LINE : bool, optional
        If True, skip the first line of the file (default: False).

    Returns
    -------
    BooleanNetwork or None
        Parsed Boolean network, or None if parsing fails.
    """
    string = fetch_file(download_url)
    if IGNORE_FIRST_LINE:
        string = string[string.index('\n')+1:]
    for separator in possible_separators:
        for (original_not, original_and, original_or) in [
            ('not', 'and', 'or'), ('NOT', 'AND', 'OR'),
            ('!', '&', '|'), ('~', '&', '|')]:
            try:
                bn = BooleanNetwork.from_string(
                    string,
                    separator=separator,
                    max_degree=max_degree,
                    original_not=original_not,
                    original_and=original_and,
                    original_or=original_or
                )
                return bn
            except:
                pass


download_urls_pystablemotifs = None

def get_bio_models_from_repository(repository):
    """
    Load Boolean network models from selected online repositories.

    Parameters
    ----------
    repository : {'expert-curated (ckadelka)', 'pystablemotifs (jcrozum)', 'biodivine (sybila)'}
        Source repository identifier.

    Returns
    -------
    bns : list of instances of BooleanNetwork
        Successfully parsed Boolean networks.
    successful_download_urls : list of str
        URLs of models successfully loaded.
    failed_download_urls : list of str
        URLs of models that could not be parsed.
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
                    bn = load_model(download_url)
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
                bn = load_model(download_url)
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
                bn = load_model(download_url, IGNORE_FIRST_LINE=True)
                bns.append(bn)
                successful_download_urls.append(download_url)
            except:
                failed_download_urls.append(download_url)

    else:
        print('Error: repositories must be one of the following:\n - ' + '\n - '.join(repositories))

    return bns, successful_download_urls, failed_download_urls


