# ast2vec Version 0.2.0

Copyright (C) 2021 - Benjamin Paassen, Jessica McBroom  
The University of Sydney

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

## Introduction

ast2vec is a neural network that can translate a Python syntax tree to a
256-dimensional vector and a 256-dimensional vector back to a syntax tree.
If you use ast2vec in academic work, please cite the paper

* Paaßen, B., McBroom, J., Jeffries, B., Koprinska, I., and Yacef, K. (2021).
  Mapping Python Programs to Vectors using Recursive Neural Encodings.
  Journal of Educational Datamining. In press. [Link][paper]

**Note:** This network was updated in June 2021 to account for changes
in the Python grammar since version 3.8. We also slightly changed the
architecture, meaning less parameters, better autoencoding performance,
and an easier interface.

## Quickstart Guide

If you wish to translate a Python program to a vector, you need to apply the
following steps.

```python
# compile the Python code into a syntax tree
src = "print('Hello world!')"
import ast
import python_ast_utils
tree = python_ast_utils.ast_to_tree(ast.parse(src))

# load the ast2vec model
import ast2vec
model = ast2vec.load_model()

# translate to a vector
x = model.encode(tree)
```

The vector `x` is 256-dimensional and represents the syntax tree.

If you wish to decode a vector back into a syntax tree, you need to apply
the following function.

```python
tree = model.decode(x, max_size = 300)
```

The optional `max_size` argument is important to prevent endless loops during
decoding.

The `ast2vec` module also contains functions `encode_trees` and `decode_points`
to encode and decode multiple objects at the same time.

## Tutorial

A more detailed tutorial using an entire mock data set of programs is given in
the `tutorial.ipynb` notebook.

## Background

ast2vec is a recursive tree grammar autoencoder as proposed by [Paaßen, Koprinska, and Yacef (2021)][rtgae].
In more detail, the encoder part of the model
follows the recursive behavior of a bottom-up tree parser of the Python
programming language to generate the overall vector encoding. Consider the
example of the syntax tree `Module(Expr(Call(Name, Str)))` which corresponds
to the program `print('Hello, world!')`. To process this syntax tree we start
with the leaf nodes `Name` and `Str` and encode them as learned 256-dimensional
vectors. Note that every Name and every Str receive the same vector encoding,
irrespective of content. Next, we apply a small neural network for the `Name`
node which takes the two vector encodings for `Name` and `Str` as input and
returns the vector code for the entire subtree `Call(Name, Str)`. This process
continues recursively until we have encoded the entire tree.

For decoding, we apply an inverse mechanism: Given the vector code `x`, we
first apply a classifier `h` which chooses the current syntactic element
(e.g. `Module`). Once we know the syntactic element we apply a separate
decoding function for each child of such a syntactic element, yielding the
vector codes for all children. Then, we apply the same scheme recursively until
no child is left to decode anymore.

The entire network is trained in a variational autoencoding framework
([Kingma and Welling, 2013][Kin2013]). In particular, we train the network to
maximize the probability that the vector code for a training tree gets
decoded back to itself, even after adding a small amount of Gaussian noise.
This scheme has the advantage that we do not need any expert annotation. We
only need syntax trees as training data.

In particular, our training data consisted of 448,992 Python programs recorded
as part of the the [National Computer Science School (NCSS)][NCSS], an
Australian educational outreach programme. The course was delivered by the
[Grok Learning platform][grok]. After compilation, we were left with 86,991
unique abstract syntax trees. We performed training for 130,000 epochs (each
epoch with a mini-batch of 32 trees), which corresponds to roughly one epoch
per program (32 * 130,000 = 416,000).
All training was performed on a consumer-grade laptop with a 2017 Intel core
i7 CPU and took roughly one week of real time.

## Contents

* `ast2vec.pt` : A pytorch model file containing the ast2vec model parameters.
* `ast2vec.py` : A Python module containing convenience functions to use ast2vec.
* `LICENSE.md` : A copy of the [GPLv3][GPLv3] license.
* `mock_dataset` : A directory containing an example dataset of Python code for the `tutorial.ipynb` notebook.
* `python_ast_utils.py` : A Python module containing convenience functions to access Python abstract syntax trees as well as the formal grammar describing the Python programming language.
* `README.md` : This file.
* `recursive_tree_grammar_auto_encoder.py` : A pyTorch implementation of recursive tree grammar autoencoders (Paaßen, Koprinska, and Yacef, 2021).
* `tree.py` : A Python implementation of a recursive tree datastructure for internal use.
* `tree_grammar.py` : A Python implementation of regular tree grammars.
* `tutorial.ipynb` : An ipython notebook illustrating how to use ast2vec to analyze program data.
* `variable_classifier.py` : A [scikit-learn][sklearn] compatible classifier to infer the variable references and function calls in a syntax tree based on training data. Refer to the `tutorial.ipynb` notebook for an example.
* `version_0_1_0` : A directory containing the version as described in the [paper][paper]. This version is only compatible with Python version 3.7 and thus not up to date anymore.

## License

All code enclosed in this repository is licensed under the
[GNU General Public License (Version 3)][GPLv3]. This documentation as well as the
neural network parameters are licensed under [Creative Commons Attribution Share-Alike (CC-BY-SA 4.0)][CCBYSA].

## Dependencies

This library depends on [NumPy][np] for matrix operations, on
[scikit-learn][sklearn] for the support vector machine solver, and on
[pyTorch][torch] for the actual ast2vec model. For the respective licenses,
please refer to their websites.

## Literature

* Paaßen, B., McBroom, J., Jeffries, B., Koprinska, I., and Yacef, K. (2021). Mapping Python Programs to Vectors using Recursive Neural Encodings. Journal of Educational Datamining. in press. [Link][paper]
* Paaßen, B., Koprinska, I., and Yacef, K. (2021). Recursive Tree Grammar Autoencoeders. Submitted to the IEEE Transactions on Neural Networks and Learning Systems. Under Review. [Link][rtgae]
* Kingma, D., and Welling, M. (2013). Auto-Encoding Variational Bayes. Proceedings of the 1st International Conference on Learning Representations (ICLR 2013). [Link][Kin2013]

[NCSS]:https://ncss.edu.au "National Computer Science School (NCSS)"
[grok]:https://groklearning.com/challenge/ "Grok Learning platform"
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html "The GNU General Public License Version 3"
[CCBYSA]:https://creativecommons.org/licenses/by-sa/4.0/ "Creative Commons Attribution 4.0 International (CC BY SA 4.0) License"
[np]: http://numpy.org/ "Numpy"
[sklearn]: https://scikit-learn.org/stable/ "Scikit-learn"
[torch]:https://pytorch.org/ "pyTorch"
[Kin2013]:https://arxiv.org/abs/1312.6114 "Kingma, D., and Welling, M. (2013). Auto-Encoding Variational Bayes. Proceedings of the 1st International Conference on Learning Representations (ICLR 2013)."
[rtgae]:https://arxiv.org/abs/2012.02097 "Paaßen, B., Koprinska, I., and Yacef, K. (2021). Recursive Tree Grammar Autoencoeders. Submitted to the IEEE Transactions on Neural Networks and Learning Systems. Under Review."
[paper]:https://educationaldatamining.org/EDM2021/virtual/static/pdf/EDM21_paper_J499.pdf "Paaßen, B., McBroom, J., Jeffries, B., Koprinska, I., and Yacef, K. (2021). Mapping Python Programs to Vectors using Recursive Neural Encodings. Journal of Educational Datamining. in press."
