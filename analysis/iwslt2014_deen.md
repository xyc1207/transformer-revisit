<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Study of IWSLT German&#8594;English

## Dataset
The data is available at [https://wit3.fbk.eu/archive/2014-01/texts](https://wit3.fbk.eu/archive/2014-01/texts). We follow [[edunov2018classical]](www.baidu.com) to split the training/validation/test sets. There are $153k$/$7k$/$7k$ sentence pairs in the training/validation/test sets. All words are lower-cased. 

## Model Configuration
We use the tensorflow based _Transformer_ for all experiments. The _tensor2tensor_ version is 1.2.9. We try the following settings:
 - _transformer_small_ + v2: The hidden dimension and filter size are as 256 and 1024 respectively. We try different dropout rates {0.1, 0.3, 0.4} and different number of layers $L\in\\{2, 4, 6, 8, 10\\}$. The number of the "heads" in MultiHead attention is 4;
 -  _transformer_base_ + v2: The hidden dimension and filter size are as 512 and 1024 respectively. We try different dropout rates {0.4, 0.5} and different number of layers {6, 8}. The number of the "heads" in MultiHead attention is 8;
 - _transformer_small_ + v1: The configurations are the same as _transformer_small_ + v2.
 
All experiments are conducted on a single M40 GPU. The batchsize is $6000$ tokens per GPU. Each v2/v1 setting is independently runned for four/two times.
 
## Inference results
We use beam search with beamwidth 5 and lenght penalty 1.0 to generate candidates. The mean and standard derivation are reported of each setting are reported.
<iframe width="800" height="650" frameborder="0" scrolling="no" src="https://xyc1207.github.io/transformer-revisit/summary/transformer_tf_iwslt2014_deen.html"></iframe>
Therefore, we recommend to use _transformer_small_ + v2 + $8/10$-layer network.

The BLEU scores reported in existing work are summarized in [BLEU scores in existing works](https://xyc1207.github.io/transformer-revisit/summary/baselines_iwslt2014_deen.html)


## Training performance of IWSLT German$\to$English

Most of existing works focus on the test performances of NMT. In this page, we show the training performance of IWSLT.  For ease of reference, we use $L$ to denote the number of layers.


## Training loss w.r.t training iterations
<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~xyc1207/12.embed"></iframe>
The the above picture, the legends are shown in the way ({v1, v2}, #number of layer and dropout). 

We have the following observations:
 + With _transformer_small_, for both v1 and v2 settings, when fixing dropout rate as 0.3, the training loss decreases w.r.t. the layer numbers. Increasing the dropout rate will hurt the training performances.
 + We can obtain similar conclusion for _transformer_base_.
 + When $L\le8$, v1 converges faster and better (i.e., eventually to a lower training loss) than v2; 
 + v1 setting cannot converge when $L\ge12$. (Results not shown here).
 + _transformer_base_ + 6-layer network achieves the lowest training loss among all settings. That is, widening the network can efficiently reduce the training error.

## Training loss w.r.t wall-clock time
We can obtain similar conclusions to the curves w.r.t. training iterations.
<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~xyc1207/14.embed"></iframe>


