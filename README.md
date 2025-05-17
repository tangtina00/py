
```
py
├─ data
│  └─ MNIST
│     └─ raw
│        ├─ t10k-images-idx3-ubyte
│        ├─ t10k-images-idx3-ubyte.gz
│        ├─ t10k-labels-idx1-ubyte
│        ├─ t10k-labels-idx1-ubyte.gz
│        ├─ train-images-idx3-ubyte
│        ├─ train-images-idx3-ubyte.gz
│        ├─ train-labels-idx1-ubyte
│        └─ train-labels-idx1-ubyte.gz
├─ environment.yaml
├─ __pycache__
│  ├─ 乘法层.cpython-311.pyc
│  └─ 加法层.cpython-311.pyc
├─ 公共函数_数据集sourcecode
│  ├─ common
│  │  ├─ functions.py
│  │  ├─ gradient.py
│  │  ├─ layers.py
│  │  ├─ multi_layer_net.py
│  │  ├─ multi_layer_net_extend.py
│  │  ├─ optimizer.py
│  │  ├─ trainer.py
│  │  ├─ util.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ functions.cpython-311.pyc
│  │     ├─ functions.cpython-39.pyc
│  │     ├─ gradient.cpython-311.pyc
│  │     ├─ gradient.cpython-39.pyc
│  │     ├─ layers.cpython-311.pyc
│  │     ├─ layers.cpython-39.pyc
│  │     ├─ multi_layer_net.cpython-311.pyc
│  │     ├─ multi_layer_net.cpython-39.pyc
│  │     ├─ multi_layer_net_extend.cpython-39.pyc
│  │     ├─ optimizer.cpython-311.pyc
│  │     ├─ optimizer.cpython-39.pyc
│  │     ├─ trainer.cpython-39.pyc
│  │     ├─ util.cpython-311.pyc
│  │     ├─ util.cpython-39.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-39.pyc
│  ├─ dataset
│  │  ├─ lena.png
│  │  ├─ lena_gray.png
│  │  ├─ mnist.pkl
│  │  ├─ mnist.py
│  │  ├─ t10k-images-idx3-ubyte.gz
│  │  ├─ t10k-labels-idx1-ubyte.gz
│  │  ├─ train-images-idx3-ubyte.gz
│  │  ├─ train-labels-idx1-ubyte.gz
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ mnist.cpython-311.pyc
│  │     ├─ mnist.cpython-39.pyc
│  │     ├─ __init__.cpython-311.pyc
│  │     └─ __init__.cpython-39.pyc
│  ├─ __init__.py
│  └─ __pycache__
│     └─ __init__.cpython-311.pyc
├─ 卷积层权重的可视化
│  ├─ deep-learning-from-scratch
│  │  ├─ ch01
│  │  │  ├─ hungry.py
│  │  │  ├─ img_show.py
│  │  │  ├─ man.py
│  │  │  ├─ simple_graph.py
│  │  │  ├─ sin_cos_graph.py
│  │  │  └─ sin_graph.py
│  │  ├─ ch02
│  │  │  ├─ and_gate.py
│  │  │  ├─ nand_gate.py
│  │  │  ├─ or_gate.py
│  │  │  └─ xor_gate.py
│  │  ├─ ch03
│  │  │  ├─ mnist_show.py
│  │  │  ├─ neuralnet_mnist.py
│  │  │  ├─ neuralnet_mnist_batch.py
│  │  │  ├─ relu.py
│  │  │  ├─ sample_weight.pkl
│  │  │  ├─ sigmoid.py
│  │  │  ├─ sig_step_compare.py
│  │  │  └─ step_function.py
│  │  ├─ ch04
│  │  │  ├─ gradient_1d.py
│  │  │  ├─ gradient_2d.py
│  │  │  ├─ gradient_method.py
│  │  │  ├─ gradient_simplenet.py
│  │  │  ├─ train_neuralnet.py
│  │  │  └─ two_layer_net.py
│  │  ├─ ch05
│  │  │  ├─ buy_apple.py
│  │  │  ├─ buy_apple_orange.py
│  │  │  ├─ gradient_check.py
│  │  │  ├─ layer_naive.py
│  │  │  ├─ train_neuralnet.py
│  │  │  └─ two_layer_net.py
│  │  ├─ ch06
│  │  │  ├─ batch_norm_gradient_check.py
│  │  │  ├─ batch_norm_test.py
│  │  │  ├─ hyperparameter_optimization.py
│  │  │  ├─ optimizer_compare_mnist.py
│  │  │  ├─ optimizer_compare_naive.py
│  │  │  ├─ overfit_dropout.py
│  │  │  ├─ overfit_weight_decay.py
│  │  │  ├─ weight_init_activation_histogram.py
│  │  │  └─ weight_init_compare.py
│  │  ├─ ch07
│  │  │  ├─ apply_filter.py
│  │  │  ├─ gradient_check.py
│  │  │  ├─ params.pkl
│  │  │  ├─ simple_convnet.py
│  │  │  ├─ train_convnet.py
│  │  │  └─ visualize_filter.py
│  │  ├─ ch08
│  │  │  ├─ awesome_net.py
│  │  │  ├─ deep_convnet.py
│  │  │  ├─ deep_convnet_params.pkl
│  │  │  ├─ half_float_network.py
│  │  │  ├─ misclassified_mnist.py
│  │  │  └─ train_deepnet.py
│  │  ├─ common
│  │  │  ├─ functions.py
│  │  │  ├─ gradient.py
│  │  │  ├─ layers.py
│  │  │  ├─ multi_layer_net.py
│  │  │  ├─ multi_layer_net_extend.py
│  │  │  ├─ optimizer.py
│  │  │  ├─ trainer.py
│  │  │  ├─ util.py
│  │  │  └─ __init__.py
│  │  ├─ dataset
│  │  │  ├─ lena.png
│  │  │  ├─ lena_gray.png
│  │  │  ├─ mnist.py
│  │  │  └─ __init__.py
│  │  ├─ environment.yml
│  │  ├─ LICENSE.md
│  │  ├─ notebooks
│  │  │  ├─ ch01.ipynb
│  │  │  ├─ ch02.ipynb
│  │  │  ├─ ch03.ipynb
│  │  │  ├─ ch04.ipynb
│  │  │  ├─ ch05.ipynb
│  │  │  ├─ ch06.ipynb
│  │  │  ├─ ch07.ipynb
│  │  │  ├─ ch08.ipynb
│  │  │  └─ common.ipynb
│  │  └─ README.md
│  └─ 卷积层权重的可视化.py
├─ 最小值
│  ├─ __pycache__
│  │  ├─ optimizer.cpython-311.pyc
│  │  └─ 最小值.cpython-311.pyc
│  └─ 最小值.py
├─ 正反向传播
│  ├─ __pycache__
│  │  ├─ 乘法层.cpython-311.pyc
│  │  └─ 加法层.cpython-311.pyc
│  ├─ 乘法层.py
│  ├─ 加法层.py
│  ├─ 苹果.py
│  └─ 苹果桔子.py
└─ 门
   ├─ and_gate.py
   ├─ main.py
   ├─ nand_gate.py
   ├─ or_gate.py
   └─ xor_gate.py

```