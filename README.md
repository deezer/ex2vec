# Ex2Vec

This repository provides our Python code to reproduce experiments from the paper **Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect**, accepted for publication in the proceedings of the 17th International ACM Recommender Systems Conference (RecSys 2023). 

(Mere) Exposure2Vec or Ex2Vec is a model that leverages repeat consumption to characterize both users and items (in this case music tracks). 

## Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect
The traditional recommendation framework seeks to connect user and content, by finding the best match possible based on users past interaction. However, a good content recommendation is not necessarily similar to what the user has chosen in the past. As humans, users naturally evolve, learn, forget, get bored, they change their perspective of the world and in consequence, of the recommendable content. One well known mechanism that affects user interest is the **Mere Exposure Effect**: when repeatedly exposed to stimuli, users’ interest tends to rise with the initial exposures, reaching a peak, and gradually decreasing thereafter, resulting in an **inverted-U shape**. Since previous research has shown that the magnitude of the effect depends on a number of interesting factors such as stimulus complexity and familiarity, leveraging this effect is a way to not only improve repeated recommendation but to gain a more in-depth understanding of both users and stimuli. 

In our RecSys paper we present (Mere) Exposure2Vec (Ex2Vec) our model that leverages the Mere Exposure Effect in repeat consumption to derive user and item characterization and track user interest evolution. 

We validate our model through predicting future music consumption based on repetition and discuss its implications for recommendation scenarios where repetition is common.

## Dataset
We provide the used data [here](https://zenodo.org/record/8316236).


## Environment

- python 3.9.16
- torch 2.0.1
- pandas 2.0.3
- numpy 1.25.0
- sklearn 1.3.0

Please cite our paper if you use this code in your own work:

```
@inproceedings{sguerra2023ex2vec,
  title={Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect},
  author={Sguerra, Bruno and Tran, Viet-Anh and Hennequin, Romain},
  booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
  year = {2023}
}
```
