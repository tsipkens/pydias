# PYDIAS

(**Py**thon tools for **i**nversion of **a**erosol **s**ize distributions)

The **Py**thon code encompasses a suite of utilities for inverting aerosol characteristics from classifier data. Code includes tools to evaluate transfer functions of a range of instruments (`tfer`), tools to perform the more traditional unidimensional inversion (e.g., of SMPS data, `odiad`), and tools for bidimensional inversion (from tandem measurements, `bidias`). 

The bidimensional inversion program was originally released in Matlab with [Sipkens et al. (2020a)][1_JAS1] and is designed to invert tandem measurements of aerosol size distributions. Initially, this code was developed for inversion of particle mass analyzer-differential mobility analyzer (PMA-DMA) data to find the two-dimensional mass-mobility distribution. However, the code has since been generalized to other applications, e.g., the inversion of PMA-SP2 data as in Naseri et al. ([2021][naseri21], [2022][naseri22]). 


#### References

[Buckley, D. T., Kimoto, S., Lee, M. H., Fukushima, N., & Hogan Jr, C. J. (2017). Technical note: A corrected two dimensional data inversion routine for tandem mobility-mass measurements. *J. Aerosol Sci.* 114, 157-168.][3_Buck]

[Cultrera, A., & Callegaro, L. (2016). A simple algorithm to find the L-curve corner in the regularization of inverse problems. arXiv preprint arXiv:1608.04571.][7_CC_Lcurve]

[Naseri, A., Sipkens, T. A., Rogak, S. N., Olfert, J. S. (2021). An improved inversion method for determining two-dimensional mass distributions of non-refractory materials on refractory black carbon. *Aerosol Sci. Technol.* 55, 104-118.][naseri21]

[Naseri, A., Sipkens, T. A., Rogak, S. N., Olfert, J. S. (2022). Optimized instrument configurations for tandem particle mass analyzer and single particle-soot photometer experiments. *J. Aerosol Sci.* 160, 105897.][naseri22]

[Rawat, V. K., Buckley, D. T., Kimoto, S., Lee, M. H., Fukushima, N., & Hogan Jr, C. J. (2016). Two dimensional size–mass distribution function inversion from differential mobility analyzer–aerosol particle mass analyzer (DMA–APM) measurements. *J. Aerosol Sci.*, 92, 70-82.][rawat]

[Sipkens, T. A., Olfert, J. S., & Rogak, S. N. (2019). MATLAB tools for PMA transfer function evaluation (mat-tfer-pma).][5_code]

[Sipkens, T. A., Hadwin, P. J., Grauer, S. J., & Daun, K. J. (2017). General error model for analysis of laser-induced incandescence signals. *Appl. Opt.* 56, 8436-8445.][6_AO17]

[Sipkens, T. A., Olfert, J. S., & Rogak, S. N. (2020a). Inversion methods to determine two-dimensional aerosol mass-mobility distributions: A critical comparison of established methods. *J. Aerosol Sci.* 140, 105484.][1_JAS1]

[Sipkens, T. A., Olfert, J. S., & Rogak, S. N. (2020b). New approaches to calculate the transfer function of particle mass analyzers. *Aerosol Sci. Technol.* 54, 111-127.][2_AST]

[Sipkens, T. A., Olfert, J. S., & Rogak, S. N. (2020c). Inversion methods to determine two-dimensional aerosol mass-mobility distributions II: Existing and novel Bayesian methods. *J. Aerosol Sci.* 146, 105565][4]

[Stolzenburg, M. R. (2018). A review of transfer theory and characterization of measured performance for differential mobility analyzers. *Aerosol Sci. Technol.* 52, 1194-1218.][Stolz18]

[1_JAS1]: https://doi.org/10.1016/j.jaerosci.2019.105484
[2_AST]: https://doi.org/10.1080/02786826.2019.1680794
[3_Buck]: https://doi.org/10.1016/j.jaerosci.2017.09.012
[4]: https://doi.org/10.1016/j.jaerosci.2020.105565
[5_code]: https://github.com/tsipkens/mat-tfer-pma
[6_AO17]: https://doi.org/10.1364/AO.56.008436
[7_CC_Lcurve]: https://arxiv.org/abs/1608.04571
[naseri21]: https://doi.org/10.1080/02786826.2020.1825615
[naseri22]: https://doi.org/10.1016/j.jaerosci.2021.105897
[rawat]: https://www.sciencedirect.com/science/article/abs/pii/S0021850215300306
[Stolz18]: https://www.tandfonline.com/doi/full/10.1080/02786826.2018.1514101
[code_v11]: https://github.com/tsipkens/mat-2d-aerosol-inversion/releases/tag/v1.1
[code_v3]: https://github.com/tsipkens/mat-2d-aerosol-inversion/releases/tag/v3.0
