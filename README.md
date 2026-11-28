# TensorMIST
A deep learning library built from scratch featuring a tensor engine, automatic differentiation, CPU/GPU kernels, and a graph compiler.

### Repository structure
----------
```
TensorMIST
├─ .github
│  └─ workflows
│     └─ ci.yml
├─ .gitignore
├─ CMakeLists.txt
├─ LICENSE
├─ Makefile
├─ README.md
├─ compile_flags.txt
├─ include
│  └─ ops.h
├─ pyproject.toml
├─ src
│  ├─ backend
│  │  └─ ops.c
│  ├─ bindings
│  │  └─ module.cpp
│  ├─ tensormist
│  │  ├─ __init__.py
│  │  ├─ functional.py
│  │  ├─ tensor.py
│  │  └─ validation.py
│  └─ tests
│     ├─ ops_test.c
│     ├─ test_backward.py
│     ├─ test_forward.py
│     ├─ test_package.py
│     └─ test_tensor.py
└─ uv.lock
```
