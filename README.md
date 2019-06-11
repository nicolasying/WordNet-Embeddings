# Micipsa Semantic Similarity Embedding

Songsheng Ying, 2019.

This is a sub repository of 
***
**Similarity and Association:**

Principles of Distributed Semantic Processing in the Human Brain, 
***
a research project conducted within the Cogmaster M2 research internship.

**Author:** 
Songsheng Ying, soshyng@gmail.com

**Supervisors:** 
Sabine Ploux, Laurent Bonnasse-Gahot, Christophe Pallier.
***
**Original project name:** WordNet Embeddings.

## About Micipsa

Micipsa is the code named for the research internship project mentioned above. 
It investigates the two aspects of semantic processing's implication in the human brain. 

[Root repository of Micipsa](https://github.com/nicolasying/micipsa.github.io)

[Masters' thesis](https://github.com/nicolasying/Micipsa-Thesis)

## WordNet Embedding
This project is based on 

**Article**
Saedi, Chakaveh, António Branco, João António Rodrigues and João Ricardo Silva, 2018, ["WordNet Embeddings"](http://www.di.fc.ul.pt/~ahb/pubs/2018SaediBrancoRodriguesEtAL.pdf), In Proceedings, 3rd Workshop on Representation Learning for Natural Language Processing (RepL4NLP), 56th Annual Meeting of the Association for Computational Linguistics, 15-20 July 2018, Melbourne, Australia.

and modified as described in Section 3.2.1 of the [Masters' thesis](https://github.com/nicolasying/Micipsa-Thesis) to extract *similarity* (paradigmatics-based) semantics from WordNet alike ontologies.

## IMPORTANT
The code is not cleaned, thus contains much project related code pieces.

If you are interested in building WordEmbedding from scratch, please refer to the forked project, [WordNetEmbeddings](https://github.com/nlx-group/WordNetEmbeddings).

## Similarity and Association Dissociation

**Examples** for *similarity* and *association* semantic neighbours.

Target word: **Teacher**

*Similar*: instructor, tutee

*Associates*: classroom, student, aunt
___
*Similarity* ~ Similarity + Paradigmatics

Association ~ Relatedness + Syntagmatics + Perceptual

## wnet2vec


### WordNet used in the above paper

**English**: 

[Princeton WordNet 3.1](http://wordnetcode.princeton.edu/wn3.1.dict.tar.gz)

**French**:

[WOLF 1.0b4](https://gforge.inria.fr/frs/download.php/file/33496/wolf-1.0b4.xml.bz2), 

Sagot Benoît et Fišer Darja (2008). Building a free French wordnet from multilingual resources. In Ontolex 2008, Marrakech, Maroc

[WONEF Standard 0.1](https://wonef.fr/data/wonef-fscore-0.1.xml.bz2) from  https://wonef.fr/ 

Quentin Pradet, Gaël de Chalendar and Jeanne Baguenier Desormeaux. January 2014. WoNeF, an improved, expanded and evaluated automatic French translation of WordNet. GWC 2014, Tartu, Estonia. 

Quentin Pradet, Jeanne Baguenier-Desormeaux, Gaël de Chalendar et Laurence Danlos. Juin 2013. WoNeF : amélioration, extension et évaluation d’une traduction française automatique de WordNet. TALN 2013, Les Sables d'Olonne, France.

### Other Data Sources 

[Dico](http://www.atlas-semantiques.eu/index.html?l=FR) proprietary data, the link points to the Atlas Semantiques, a project using the same data source.

Sabine Ploux and Hyungsuk Ji. (2003). A Model for Matching Semantic Maps Between Languages ( French / English, English / French ). Computational Linguistics. 29(2):155-178.  MIT Press.

###  Test sets used in above paper

Notes from the original project:

Please note that the semantic network to semantic space method presented in the above paper includes random-based subprocedures (e.g. selecting one word from a set of words with identical number of outgoing edges). The test scores may present slight fluctuations over different runs of the code.

**English**

[SimLex-999](https://www.cl.cam.ac.uk/~fh295/simlex.html)

[RG1965](http://delivery.acm.org/10.1145/370000/365657/p627-rubenstein.pdf?ip=194.117.40.49&id=365657&acc=ACTIVE%20SERVICE&key=2E5699D25B4FE09E%2E454625C777251F56%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1527501385_f2095c911da3627e99b9a6c8a9769558)

[WordSim-353-Similarity](http://alfonseca.org/eng/research/wordsim353.html)

[WordSim-353-Relatedness](http://alfonseca.org/eng/research/wordsim353.html)

[MEN](http://clic.cimec.unitn.it/~elia.bruni/MEN.html)

[MTurk-771](http://www2.mta.ac.il/~gideon/datasets/)

**French**

French SimLex-999, WordSim353 benchmarks are based on [siabar/Multilingual_Wordpairs](https://github.com/siabar/Multilingual_Wordpairs).

We applied lemmatisation and made corrections, the modified benchmarks are mirrored from [Similarity-Association-Benchmarks](https://github.com/nicolasying/Similarity-Association-Benchmarks).

### Models

The similarity and relatedness dissociation is experimented with semantic relation selection, validated by Word-Pair evaluation task. We target a high score in similarity benchmarks, and a null score in association\relatedess benchmarks for *similarity* models.

Best dissociation (yet tested) is achieved with Synonymy, Hypernymy, Homonymy, adj.part._of_verb, adj.similar, adv.derive_adj, vocabulary 15k most frequent in WordNet, Dim 511/800.



**How to run wn2vec software**

To provide input files to the software the following structure must exist:

```
|-- main.py
|-- data   |-- input
|   |   |-- language_wnet
|   |   |   |-- *wnet_files
|   |   |-- language_testset
|   |   |   |-- *testset_files
|   |-- output
|-- modules
|   |-- input_output.py
|   |-- sort_rank_remove.py
|   |-- vector_accuracy_checker.py
|   |-- vector_distance.py
|   |-- vector_generator.py
|   |-- file_reader_french.py: a lazy implementation of adapter for WONEF/WOLF data format
|   |-- file_reader_synonym.py: a lazy implementation for synonym databases
```

Where *language* is the language that you are using that must be indicated in main.py in the variable **lang**.
If the language isn't supported by the current path routing in the code, which was mainly use for experiments, you may add the path to the directory in the files *input_output.py*, *vector_generator.py* and *vector_accuracy_checker.py*.

Various variables for the output of the model, such as embedding dimension, can be found in *main.py*. 

To run the software, you will need the following packages:

* Numpy
* progressbar
* keras
* sklearn
* scipy
* gensim

Python3.5 was used for the original experimentation.

The code is also Python 3.6.7 compatible.