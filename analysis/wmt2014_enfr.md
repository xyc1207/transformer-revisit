<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# A Study of WMT 2014 English&#8594;French

## Dataset
We use the data provided by [WMT2014](http://www.statmt.org/wmt14/translation-task.html). There are 36M training sentence sentence pairs. We concatenate _newstest2012_ and _newstest2013_ as the validation set (6003 sentence pairs) and use _newstest2014_ as the test set (3003 sentence pairs).


## Model Configuration
e use the pytorch based _Transformer_ for all experiments (version 0.40). We use the _transformer_big_ setting, where $d=1024$ and $d_{ff}=4096$. We try v1 and v2 setting with different layers. The dropout ratio is fixed as 0.1 for all settings.

All experiments are conducted on eight M40 GPU. The batchsize is $4096$ tokens per GPU. We set update-freq as 16 to simulate the 128 GPU environment.

Each v1/v2 setting is runned for one time due to the limitation of computation resources.
 
## Inference results
We use beam search with beamwidth 5 and lenght penalty 1.0 to generate candidates. The results are shown below:

| Network Architecture | v1 | v2 |
|-------|--------|---------|
| 6 layers | 43.06 | 42.05 |
| 8 layers | 42.69 | 42.28 |
| 10 layers | 42.73 | 42.26 |
| 12 layers | 42.69 | 42.33 |

Still, using 6-layer network is the best choice.

The BLEU scores reported in existing work are summarized in [BLEU scores in existing works](https://xyc1207.github.io/transformer-revisit/summary/baselines_wmt14_enfr.html)

## Training performance of WMT2014 English$\to$French
<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~xyc1207/16.embed"></iframe>




## BLEU scores of WMT 14 English to German

We work on the v1 setting 

| Network Architecture | BLEU | 
|-------|--------|
| 6 layers | 29.12 | 
| 8 layers | 28.75 | 
| 10 layers | 28.63 | 
| 12 layers | failed |
