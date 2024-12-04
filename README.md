# Pascal

Pascal is a software framework written in C that enables hardware-enhanced Bayesian Learning. It contains modules that can carry out operations on tensors, similar to [NumPy Arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html), and other popular tools. Pascal can also carry out reverse-mode automatic differentiation using the `pascal-autodiff` module.


## Requirements
Pascal can run using several backends. By default, Pascal uses `clapack`, which is included in this repository. Therefore you do not need any other requirements, other than a C compiler.

Pascal uses either a `cblas` or a `libgsl` backend to carry out some operations. Thus, these should be installed. The [`Makefile`](Makefile) includes flags that should point to standard locations. If these don't work, please updated the `INCLUDE_FLAGS`, `LDFLAGS` and `FRAMEWORK_FLAGS` variables in the [`configs/default.conf`](configs/default.conf) as needed. Alternatively, you can create a new configuration file (`.conf`) in the `configs` folder and update the `CONFIGURATION` variable in the [`Makefile`](Makefile) (note that if your new configuration is `configs/new-config.conf`, `CONFIGURATION` should be set to `new-config`).

### CBLAS and GSL on MacOS
On MacOS, these are included in the [Accelerate framework](https://developer.apple.com/documentation/accelerate). This is installed with the XCode command-line tools, by running:
```
xcode-select --install
```

Currently, `default.conf` assumes that [`libgsl`](https://www.gnu.org/software/gsl/doc/html/index.html) is installed using [`homebrew`](https://brew.sh) by running:
```
brew install gsl
```

If you install it using other means, please update the `INCLUDE_FLAGS` and `LDFLAGS` in the correct if-else block.

### CBLAS and GSL on Linux
To be written

## Build
To use this package, simply clone this repository. This can be done as:
```
git clone --recursive https://github.com/physical-computation/pascal.git
```

> [!NOTE]
> The `--recursive` is important since this repository uses submodules. This also means that if you use Pascal as a submodule to another repository, you need to run the following after adding the Pascal submodule.
> ```
> git submodule update --init --recursive
> ```

 Then you can build it using
```
make
```

This creates a library file in `libs/`

To test, run:
```
make test
```

If everything worked well, this will print:
```bash
tests/load_data/test; tests/tensor_integration/test; tests/tensor/test; tests/tensor_random_sample/test; tests/tensor_utils/test; tests/autodiff/test; tests/autodiff_primitives/test; tests/tensor_broadcast/test; tests/tensor_iterate/test;
Running tests in load_data:
Completed:       3/3 passed in 596µs
Running tests in pascal_tensor_integrations:
Completed:       15/15 passed in 192µs
Running tests in tensor:
Completed:       48/48 passed in 372µs
Running tests in pascal_tensor_random_sample:
Completed:       2/2 passed in 3µs
Running tests in pascal_tensor_utils:
Completed:       8/8 passed in 10µs
Running tests in autodiff:
Completed:       15/15 passed in 273µs
Running tests in autodiff_primitives:
Completed:       19/19 passed in 106µs
Running tests in pascal_tensor_broadcast:
Completed:       3/3 passed in 12µs
Running tests in pascal_tensor_iterate:
Completed:       6/6 passed in 8µs
```

## Examples and Benchmarks
> [!NOTE]
> Currently, the Makefiles in [`examples/`](examples/) and [`benchmarks/`](benchmarks/) are written to work with `macOS` only.

A basic implementation of a tensor operation using `Pascal` is


The examples and benchmarks use Python to run some analyses and baselines. It is recommended that you use a virtual environment. Install the required packages using
```
pip install -r requirements.txt
```
Please see the `examples/` folder. It includes a few simple examples of using the functionality that Pascal offers. The `Makefile`s in these folders currently only support `macOS`. However, they should be a useful guide for other platforms: you can think of them as compiling your application while including `libpascal.a` (which gets created when you run `make`).

### Running examples
All examples can be executed by running the following from the root directory of this project:
```
make examples
```

Individual examples can be executed using the following syntax:
```
make eg-<example>
```
`<example>` can be replaced by the names of the folders inside of [`examples/`](examples/), such as `nn-simple` and `bnn-simple`.

### Running benchmarks
We currently compare some of our tensor operations against [`NumPy`](www.numpy.org). To run these benchmarks, run
```
make benchmarks
```

## Documentation
There is a set of incomplete documentation in [`docs/`](docs/). If you load [`docs/html/index.html`](docs/html/index.html) locally, you can navigate this documentation on your browser.

Furthermore, documentation comments can be found on some of the function in [`include/pascal.h`](include/pascal.h).
