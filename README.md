# play_with_proteins
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
     
