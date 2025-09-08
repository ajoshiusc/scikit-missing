# Citation and Prior Work Notice

## Important Note on Originality

**This implementation is NOT a published research paper.** It is a novel synthesis and implementation of existing ideas from the literature on missing data in Support Vector Machines.

## Key Prior Work (What Has Been Published)

### 1. **Pelckmans et al. (2005) - Foundational Work**
```
@article{pelckmans2005handling,
  title={Handling missing values in support vector machine classifiers},
  author={Pelckmans, Kristiaan and De Brabanter, Jos and Suykens, Johan AK and De Moor, Bart},
  journal={Neural Networks},
  volume={18},
  number={5-6},
  pages={684--692},
  year={2005},
  publisher={Elsevier}
}
```
- **First systematic treatment** of missing data in SVMs
- Proposed distance-based and probabilistic approaches
- Theoretical foundation for missing data kernels

### 2. **Chechik et al. (2008) - Max-Margin Approach**
```
@article{chechik2008max,
  title={Max-margin classification of data with absent features},
  author={Chechik, Gal and Heitz, Geremy and Elidan, Gal and Abbeel, Pieter and Koller, Daphne},
  journal={Journal of Machine Learning Research},
  volume={9},
  pages={1-21},
  year={2008}
}
```
- **Max-margin framework** for absent features
- Theoretical analysis of generalization bounds
- Inspiration for our Expected Value kernel approach

### 3. **Williams et al. (2007) - Bayesian Approach**
```
@article{williams2007classification,
  title={On classification with incomplete data},
  author={Williams, David and Liao, Xuejun and Xue, Ya and Carin, Lawrence and Krishnapuram, Balaji},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={29},
  number={3},
  pages={427--436},
  year={2007},
  publisher={IEEE}
}
```
- **Bayesian treatment** of incomplete data
- Probabilistic modeling of missing features
- Foundation for our Cross-Correlation kernel

### 4. **Smola et al. (2005) - Early Kernel Work**
```
@inproceedings{smola2005kernel,
  title={Kernel methods for missing variables},
  author={Smola, Alex and Vishwanathan, SVN and Hofmann, Thomas},
  booktitle={Proceedings of the 10th International Workshop on Artificial Intelligence and Statistics},
  pages={325--332},
  year={2005}
}
```
- **Early exploration** of kernels for missing data
- Theoretical foundations
- Basis for kernel-based approaches

## What This Implementation Contributes

### Novel Aspects (Not Previously Published):
1. **Unified Framework**: Combines ideas from multiple papers into a single, practical implementation
2. **Modular Kernel Design**: Easy-to-use interface for experimenting with different approaches
3. **Scikit-learn Integration**: Full compatibility with existing ML workflows
4. **Comprehensive Implementation**: Complete working system with documentation and examples
5. **Cross-Correlation Kernel**: Our specific probabilistic kernel formulation
6. **Performance Comparison**: Systematic evaluation across different missing data scenarios

### Building on Prior Work:
- **Expected Value Kernel**: Extends Chechik et al.'s max-margin ideas
- **Cross-Correlation Kernel**: Builds on Williams et al.'s Bayesian approach
- **Kernel Framework**: Develops Smola et al.'s kernel concepts
- **Implementation**: Makes Pelckmans et al.'s theoretical ideas practical

## How to Cite This Work

If you use this implementation, please cite the foundational papers above and acknowledge this implementation:

```
@software{joshi2025scikit_missing,
  title={scikit-missing: Support Vector Machines for Missing Data Classification},
  author={Joshi, Ajay},
  year={2025},
  url={https://github.com/ajoshiusc/scikit-missing},
  note={Implementation building on Pelckmans et al. (2005), Chechik et al. (2008), Williams et al. (2007), and Smola et al. (2005)}
}
```

## Publishing This Work

If you plan to publish research using this implementation:

1. **Cite the foundational papers** listed above
2. **Acknowledge this implementation** as a novel synthesis
3. **Consider collaboration** - contact ajoshi@usc.edu
4. **Follow academic standards** for reproducible research

## Related Recent Work

For completeness, here are other relevant papers in this area:

```
@article{garcia2010pattern,
  title={Pattern classification with missing data: a review},
  author={Garc{\'\i}a-Laencina, Pedro J and Sancho-G{\'o}mez, Jos{\'e}-Luis and Figueiras-Vidal, An{\'\i}bal R},
  journal={Neural Computing and Applications},
  volume={19},
  number={2},
  pages={263--282},
  year={2010}
}

@inproceedings{gondara2018mida,
  title={MIDA: Multiple imputation using denoising autoencoders},
  author={Gondara, Lovedeep and Wang, Ke},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={260--272},
  year={2018}
}
```

---

**Summary**: This is an original implementation that synthesizes and extends published research, but is not itself a published paper. Always cite the foundational work when using this implementation.
