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
We use the data provided by [WMT2014](http://www.statmt.org/wmt14/translation-task.html). There are 4.5M training sentence sentence pairs. We concatenate _newstest2012_ and _newstest2013_ as the validation set (6003 sentence pairs) and use _newstest2014_ as the test set (3003 sentence pairs).


## Model Configuration
e use the pytorch based _Transformer_ for all experiments (version 0.40). We use the _transformer_big_ setting, where $d=1024$ and $d_{ff}=4096$. We try v1 and v2 setting with different layers. The dropout ratio is fixed as $0.3$ for all settings.

All experiments are conducted on eight M40 GPU. The batchsize is $4096$ tokens per GPU. We set update-freq as 16 to simulate the 128 GPU environment.

Each v1/v2 setting is runned for one time due to the limitation of computation resources.
 
## Inference results
We use beam search with beamwidth 5 and lenght penalty 1.0 to generate candidates. The results are shown below:

| Network Architecture | v1 | v2 |
|-------|--------|---------|
| 6 layers | 29.12 | 26.11 |
| 8 layers | 28.75 | -- |
| 10 layers | 28.63 | -- |
| 12 layers | fail | -- |

Still, using 6-layer network is the best choice.

