# Projects within
 ### 1. Protein Embeddings
    Files:
    - prot2vec.py
    - tsne.py
   
    Instructions:
     1. Run tsne.py to view embeddings
     
     2. If you use the reviewed-uniprot-lim_sequences.txt file
      Training times on 1 GPU per epoch ~ 1 hour and 30 minutes
      
      
    If you want to run your own dataset:
     1. Change SequenceData dataset to your text file.
      a. line 1 = 'sequence'
      b. every line should be a protein sequence no seperator between amino acids
       i.e MGSKTLPAPVPIHPSLQLTNYSFLQAFNGL...\n
       
     2. Run prot2vec.py with your dataset
      a. Alter training parameters if need be
      
     3. Run tsne.py to view embeddings
     
     
     
### 2. Ground State Energy Prediction
     - Instructions:
       1. Download jsons: https://www.kaggle.com/datasets/burakhmmtgl/predict-molecular-properties
       2. Run feature_extraction.py, edit the structure such that you have a jsons folder in the data directory
       3. Run gse_prediction.py
       
### 3. Machine Interrogator Turing Test 
     - The write up includes methods, data information, and purpose of project.
     - Within this project there are 3 models:
         - Word Embedding model (Skip-gram word2vec)
         - Machine Responder model (TextGenerate)
         - Machine Interrogator model (MachineInterrogator)
         
     - Used jokes dataset to train the interrogator on question responding. 
        The interrogator was then asked to interpret a human completion of the joke vs a machine completion of the joke.
