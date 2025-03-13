# PYDIAS

(**Py**thon tools for **i**nversion of **a**erosol **s**ize distributions)

The **Py**thon code encompasses a suite of utilities for inverting aerosol characteristics from classifier data. Code includes tools to evaluate transfer functions of a range of instruments (`tfer`), tools to perform the more traditional unidimensional inversion (e.g., of SMPS data, `odiad`), and tools for bidimensional inversion (from tandem measurements, `bidias`). 

The bidimensional inversion program was originally released in Matlab with [Sipkens et al. (2020a)][1_JAS1] and is designed to invert tandem measurements of aerosol size distributions. Initially, this code was developed for inversion of particle mass analyzer-differential mobility analyzer (PMA-DMA) data to find the two-dimensional mass-mobility distribution. However, the code has since been generalized to other applications, e.g., the inversion of PMA-SP2 data as in Naseri et al. ([2021][naseri21], [2022][naseri22]). 



