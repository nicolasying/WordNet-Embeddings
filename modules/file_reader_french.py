# Songsheng YING
# coding=utf-8
# python3.6.7 Anaconda

import os, sys
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
import xml.etree.ElementTree as ET

def data_file_reader_fr(file_name, lang):
    print("    Working on " + file_name)
    if lang == "French":
        path = os.getcwd() + '/data/input/French_wnet/'
    else:
        print("This reader function only supports French file")
        return

    tree = ET.parse(path + file_name)
    root = tree.getroot()
    
    rel_type_dict = {
        'near_antonym': '!',
        'hypernym': '@',
        'instance_hypernym': '@i',
        'hyponym': '~',
        'instance_hyponym': '~i',
        'be_in_state': '#s', # TO VERIFY
        'eng_derivative': '+',
        'subevent': '*', # TO VERIFY
        'also_see': '^',
        'verb_group': '$',
        'category_domain': ';c',
        'derived': '\\',
        'similar_to': '&',
        'usage_domain': ';u',
        'region_domain': ';r',
        'holo_part': '#p',
        'holo_member': '#m',
        'causes': '>',
        'holo_portion': '#p', # TO VERIFY
        'participle': '<'
    }

    file_data = {}
    offset_list = []

    for synset in root:
        synsetWrds = []
        synsetConnections = []
        synsetRelationTypes = []
        connectedSynsetPos = []

        for word in synset.find('SYNONYM').getchildren():
            try:
                synsetWrds.append(word.text.replace(' ', '_'))
            except:
                print(word.tag, word.attrib)
                synsetWrds.append('__unknown__')
        for relation in synset.findall('ILR') :
            synsetRelationTypes.append(rel_type_dict[relation.attrib['type']])
            synsetConnections.append(relation.text)
            connectedSynsetPos.append(relation.text.split('-')[3])

        data = (synsetWrds, synsetConnections, synsetRelationTypes, connectedSynsetPos, None)
        file_data.update({synset.find('ID').text:data})
        offset_list.append(synset.find('ID').text)

    return file_data, offset_list