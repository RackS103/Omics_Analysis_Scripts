# Omics/Phosphoproteomics Analysis Scripts
Rac Mukkamala, White Lab

A compilation of functions and scripts I've created to streamline analysis of phosphoproteomic data in Python!

- ```PLS_Scripts_RM.py```: List of functions to automate PLSR/PLSDA analysis of omics data. Includes functions for cross-validation, feature selection, VIP feature importance scoring, and automated model fitting

- ```PLSDA_RM.py```: My implementation of PLSDA as a subclass of scikit-learn's ```ClassifierMixin``` and ```TransformerMixin```, so this model can be plugged into all default sklearn methods and pipelines.

- ```Enrichment_Scripts_RM.py```: Collection of functions to automate pathway enrichment via Enrichr, KEA3, and STRING. The functions connect to the API of the pathway enrichment tools and directly download the results from there.
