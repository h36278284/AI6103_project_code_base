# text_gcn.pytorch

This project refers ["Graph Convolutional Networks for Text Classification. Yao et al. 2018."](https://arxiv.org/abs/1809.05679) and ["FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling. Chen et al. 2018"](https://doi.org/10.48550/arXiv.1801.10247).

This implementation highly based on official code [yao8839836/text_gcn](<https://github.com/yao8839836/text_gcn>) and [Gkunnan97/FastGCN_pytorch](<https://github.com/gkunnan97/fastgcn_pytorch>).


## Running training and evaluation

1. `cd ./preprocess`
2. Run `python remove_words.py <dataset>`
3. Run `python build_graph.py <dataset>`
4. `cd ..`
5. Run `python train.py <dataset>` for TextGCN
6. Run `python tfgcn.py <dataset>` for FastGCN
7. Run `python tf_gcn.py <dataset>` for TextFGCN
8. Replace `<dataset>` with `20ng`, `R8`, `R52`, `ohsumed` or `mr`


