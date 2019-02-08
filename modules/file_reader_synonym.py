# Songsheng YING
# coding=utf-8
# python3.6.7 Anaconda

from vector_generator import *
from vector_accuracy_checker import *
from vector_distance import *
import time
from time import gmtime, strftime

def testset_extractor(ref_model, lang):
    if lang == "English":
        path = os.getcwd() + '/data/input/English_testset/'
    else:
        print("Language not implemented")
    
    word_set = set()
    for testset in ref_model:
        file = open(path+testset, mode='r')
        file_data = file.readlines()
        for line in file_data:
            word_set.update(line.split(';')[0:2])
        file.close()
    return word_set


def syn_pMatrixBuilder(file_name, lang, to_keep, self_loop, log, ref_model):
    print("    Working on " + file_name)
    if lang == "French":
        # TODO
        return 0, 0
        #path = os.getcwd() + '/data/input/Dutch_wnet/'
    else:
        path = os.getcwd() + '/data/input/English_syn/'

    fl = open(path + file_name, encoding="cp1252")
    dataLine = fl.readlines()
    fl.close()

    file_data = {}
    offset_list = []
    start_time = time.time()
    words_wrdcnt = {}   # Note: wi:cnt(wi)
    
    # First pass: word count
    for lineNum in range(len(dataLine)):
        dataLineParts = dataLine[lineNum].split("\t")
        
        if len(dataLineParts) < 4:
            print("Data line too short. 1")
            print(dataLine[lineNum])
            continue
        
        for word in dataLineParts[2:-1]:
            word = word.replace(' ', '_')
            if word not in words_wrdcnt:
                words_wrdcnt[word] = 1
            else: 
                words_wrdcnt[word] += 1
    
    # Filter words
    if to_keep != "all":
        to_keep = int(to_keep)
        word_list = [k for k in sorted(words_wrdcnt, key=words_wrdcnt.get, reverse=True)]
        word_list = word_list[:to_keep]
        # Add test set words
        word_set = testset_extractor(ref_model, lang)
        word_set = word_set.intersection(words_wrdcnt.keys())
        word_set.update(word_list)
        word_list = list(word_set)
        word_indx = {word:indx for indx, word in enumerate(word_list)}
    
    # Second pass: p_matrix
    dim = len(word_list)
    p_matrix = np.zeros((dim, dim), dtype = np.float16) 
    for lineNum in range(len(dataLine)):
        dataLineParts = dataLine[lineNum].split("\t")
        
        if len(dataLineParts) < 4:
            print("Data line too short. 2")
            continue
        
        if dataLineParts[0].replace(' ', '_') not in word_indx:
            continue
        else:
            header_indx = word_indx[dataLineParts[0].replace(' ', '_')]
            
            if self_loop:
                p_matrix[header_indx, header_indx] = 1    
            for word in dataLineParts[2:-1]:
                word = word.replace(' ', '_')
                if word not in word_indx:
                    continue
                else: 
                    p_matrix[header_indx, word_indx[word]] = 1    
    
    finish_time = time.time()
    print("    Relation matrix is created")
    log.write("\n    Relation matrix was created in %.3f seconds\n"%(finish_time-start_time))
    
    # to check the number of non-zero elements in the p matrix
    print("    Checking the number of non-zero elements in relation matrix")
    non_zero = len(p_matrix[np.nonzero(p_matrix)])
    #non_zero = -10

    print("        %d elements out of %d elements are non-zero" % (non_zero, len(p_matrix) * len(p_matrix)))
    log.write("        %d elements out of %d elements are non-zero\n" % (non_zero, len(p_matrix) * len(p_matrix)))


    return p_matrix, word_list