# PLS-PF-S

This repository contains the supplementary numerical material and MATLAB
implementation associated with the paper

**Subspace methods for min-max problems**

by Morteza Kimiaei, Shima Shabani, and Michael Breuß.

The repository contains the MATLAB implementation of the proposed
line-search projection-based subspace methods (`PLS-S`) and their fixed-step
counterparts (`PF-S`), the learning-based monotone-equation test environment,
the comparison methods used in the numerical experiments, the driver for the
representative learning-based experiments, and the supplementary material.

The repository is organized into four main parts:

```text
PLS-PF-S/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── suppMat/
│   └── suppMat.pdf
│
├── TE/
│   ├── TE.mat
│   └── genTE.m
│
├── solvers/
│   ├── DFSdir.m
│   └── MonotoneSolvers.m
│
└── examples/
    └── driverME.m
```

## 1. Supplementary material

The folder `suppMat/` contains

```text
suppMat/suppMat.pdf
```

The supplementary material provides additional background, structural
analysis, implementation details, and numerical results complementing the
main paper.

In particular, it contains:

- additional background on nonlinear monotone equations and convex--concave
  min--max problems;
- the asymptotic and canonical OGDA-type direction structures;
- the residual--memory representation of the proposed Jacobian-free
  subspace directions;
- the relation with representative OGDA-type methods;
- the complete construction and validation of the learning-based test
  problems;
- detailed comparisons of the two- and three-term `PLS-S` and `PF-S`
  variants; and
- detailed numerical results for the OGDA-type comparison methods.

---

## 2. Monotone-equation test environment

The folder `TE/` contains

```text
TE/TE.mat
TE/genTE.m
```

### 2.1 Test environment

The file `TE.mat` stores the MATLAB structure `TE`, containing the
100 learning-based nonlinear monotone-equation test problems used in the
numerical experiments.

The problems are generated from three application classes:

- **Multi-Agent Reinforcement Learning (MARL)**,
- **Robust Adversarial Learning (RAL)**,
- **Generative Adversarial Networks (GANs)**.

All generated problems are unconstrained nonlinear monotone equations of the
form

```text
E(x) = 0,    x in R^n.
```

The corresponding min--max models have saddle operators of the form

```text
E(x) = [ grad_min Phi ; -grad_max Phi ].
```

The algorithms operate directly on `R^n`. The finite ranges stored with the
problems are used only for generating representative initial points and
validation samples; they are not imposed as constraints on the iterates.

### 2.2 Generating the test environment

The script

```text
TE/genTE.m
```

generates the complete test environment.

The generator uses

```matlab
Nproblems = 100;
rng(42);
```

so that the generated collection is reproducible.

The three application families are

```matlab
families = {'marl','robust_adversarial','gan'};
```

and each family contains four variants with different dimensions and
application-specific parameters.

The generator also performs numerical validation of each instance, including
checks of:

- dimensions;
- finiteness of the operator values;
- the known solution;
- monotonicity on randomly sampled validation pairs;
- the stored Lipschitz constant or global Lipschitz upper bound; and
- the validity of the generated initial point.

### 2.3 Loading `TE.mat`

To load the test environment in MATLAB, use

```matlab
load('TE/TE.mat','TE')
```

The individual problems are stored in

```matlab
TE.problem
```

For example,

```matlab
names = fieldnames(TE.problem);

pname = names{1};

P = TE.problem.(pname);
```

### 2.4 Structure of an individual test problem

Each problem structure contains the main fields

```matlab
P.name
P.variant
P.dim
P.L

P.x
P.xstar

P.funE
P.funf
P.resf

P.initLow
P.initUpp

P.low
P.upp
```

Their meanings are:

```text
P.name       problem name

P.variant    application/variant identifier

P.dim        dimension of the nonlinear monotone equation

P.L          global Lipschitz constant or a valid global upper bound

P.x          initial point used in the numerical experiments

P.xstar      known equilibrium/root used for validation

P.funE       monotone operator E(x)

P.funf       corresponding min--max objective Phi(x)

P.resf       residual merit function 0.5*||E(x)||^2

P.initLow    lower endpoint of the initialization/validation range

P.initUpp    upper endpoint of the initialization/validation range

P.low        -Inf vector representing the unconstrained domain

P.upp        +Inf vector representing the unconstrained domain
```

For example,

```matlab
x0 = P.x;

E0 = P.funE(x0);

f0 = P.resf(x0);

Phi0 = P.funf(x0);

L = P.L;
```

The known solution can be checked by

```matlab
norm(P.funE(P.xstar))
```

which should be close to zero up to numerical precision.

---

## 3. Solvers

The folder `solvers/` contains

```text
solvers/DFSdir.m
solvers/MonotoneSolvers.m
```

The file `DFSdir.m` contains the proposed `PLS-S` and `PF-S` methods, while
`MonotoneSolvers.m` contains the comparison methods.

### 3.1 Proposed PLS-S/PF-S solver

The file

```text
solvers/DFSdir.m
```

implements the proposed projection-based subspace framework for solving

```text
E(x) = 0,    x in R^n.
```

Both the line-search realization `PLS-S` and the fixed-step realization
`PF-S` are implemented in the same code.

The two realizations employ the same Jacobian-free subspace directions and
the same projection mechanism. They differ only in the choice of the step
size.

### 3.2 Jacobian-free subspace directions

The available direction states are

```text
FR
PR
HS
DY
LS
DL
HZ

3PR
3HS

An1
An2
De

A1
A2
A3
A4
A5
A6
A7
A8
```

These correspond to the four direction groups introduced in the paper:

```text
JFS2dir:
    FR, PR, HS, DY, LS, DL, HZ

JFS3dir1:
    Z1, Z2

JFS3dir2:
    An1, An2, De

JFS3dir3:
    A1, ..., A8
```

The internal MATLAB names for the two members of `JFS3dir1` are

```text
3PR  -> Z1
3HS  -> Z2
```

Thus, for example,

```matlab
tune.state = 'FR';
```

selects `PLS-S-FR`, while

```matlab
tune.state = '3HS';
```

selects `PLS-S-Z2`.

### 3.3 Fixed-step PF-S variants

Appending `-F` to a direction state selects the corresponding fixed-step
`PF-S` realization.

For example,

```matlab
tune.state = 'LS';
```

selects

```text
PLS-S-LS
```

whereas

```matlab
tune.state = 'LS-F';
```

selects

```text
PF-S-LS.
```

Similarly,

```matlab
tune.state = '3HS-F';
```

selects

```text
PF-S-Z2.
```

The complete correspondence is therefore

```text
PLS-S-FR      <->  FR
PLS-S-PR      <->  PR
PLS-S-HS      <->  HS
PLS-S-DY      <->  DY
PLS-S-LS      <->  LS
PLS-S-DL      <->  DL
PLS-S-HZ      <->  HZ

PLS-S-Z1      <->  3PR
PLS-S-Z2      <->  3HS

PLS-S-An1     <->  An1
PLS-S-An2     <->  An2
PLS-S-De      <->  De

PLS-S-A1      <->  A1
...
PLS-S-A8      <->  A8
```

and

```text
PF-S-FR       <->  FR-F
PF-S-PR       <->  PR-F
PF-S-HS       <->  HS-F
PF-S-DY       <->  DY-F
PF-S-LS       <->  LS-F
PF-S-DL       <->  DL-F
PF-S-HZ       <->  HZ-F

PF-S-Z1       <->  3PR-F
PF-S-Z2       <->  3HS-F

PF-S-An1      <->  An1-F
PF-S-An2      <->  An2-F
PF-S-De       <->  De-F

PF-S-A1       <->  A1-F
...
PF-S-A8       <->  A8-F
```

### 3.4 Direction safeguard and scaling

The effective subspace direction used by `PLS-S` and `PF-S` is constructed
according to the algorithm in the paper:

```text
candidate JFS direction
        |
        v
zero-direction safeguard
        |
        v
angle-condition check
        |
        v
angle modification, if necessary
        |
        v
scaling
        |
        v
p_l^sc
```

The angle condition is imposed with respect to the current residual
`E(x_l)`.

After the angle condition has been enforced, the resulting nonzero direction
is scaled so that

```text
zeta ||E(x_l)|| <= ||p_l^sc|| <= ||E(x_l)||.
```

Thus, the direction used to generate the trial point is the controlled scaled
direction `p_l^sc`.

### 3.5 PLS-S parameters

For the line-search `PLS-S` implementation, the parameters used in the
reported numerical experiments are

```text
rho         = 1e-2
gamma       = 2
Delta_angle = 0.9
zeta        = 1e-3
mu_init     = 0.48/L
```

where `L` is the Lipschitz constant associated with the corresponding test
problem.

The lower line-search safeguard is

```text
mu_min = min{0.48/L, 0.9/[2(L+1e-2)]}.
```

In `DFSdir.m`, the implementation variables satisfy

```text
eta = rho
ro  = 1/gamma.
```

### 3.6 PF-S parameters

For `PF-S`, the projected line search is not invoked.

The fixed step size used in the numerical experiments is

```text
alpha = min{1, 0.9/(L+1e-2)}.
```

This corresponds to the largest fixed step allowed by the admissibility
condition used in the paper with

```text
rho         = 1e-2
Delta_angle = 0.9.
```

The scaling parameter is

```text
zeta = 1e-3.
```

### 3.7 Running a PLS-S variant

After loading a problem `P`, a line-search variant can be called as follows.

For example, `PLS-S-Z2`:

```matlab
st = struct();

st.prt    = -1;
st.nfmax  = 20000;
st.secmax = 300;
st.fbest  = 0;

tune = struct();

tune.state = '3HS';
tune.mu    = 0.48/P.L;
tune.L     = P.L;

[x,f,info] = DFSdir(P.funE,P.x,st,tune);
```

### 3.8 Running a PF-S variant

For example, `PF-S-Z2`:

```matlab
st = struct();

st.prt    = -1;
st.nfmax  = 20000;
st.secmax = 300;
st.fbest  = 0;

tune = struct();

tune.state = '3HS-F';
tune.L     = P.L;

[x,f,info] = DFSdir(P.funE,P.x,st,tune);
```

If `tune.alpha` is not explicitly supplied, `DFSdir.m` uses the largest
theoretically admissible fixed step

```text
alpha = min{1, Delta_angle/(L+rho)}.
```

### 3.9 Comparison methods

The file

```text
solvers/MonotoneSolvers.m
```

contains stand-alone implementations of the following unconstrained methods:

```text
EG       Extragradient

OGDA     Optimistic Gradient Descent--Ascent

EAG      Extra Anchored Gradient

FOGDA    Fast OGDA
```

The methods operate directly on

```text
E(x) = 0,    x in R^n,
```

without box projection, clipping, or application-specific feasibility
corrections.

The desired method is selected using

```matlab
tune.method
```

For example,

```matlab
tune.method = 'EG';
```

or

```matlab
tune.method = 'OGDA';
```

### 3.10 Step sizes of the comparison methods

For the comparisons reported in the main paper, both EG and OGDA use the
Lipschitz-scaled input step

```text
s = 0.48/L.
```

The implementation additionally contains the safety caps

```text
EG      : s <= 0.99/L
OGDA    : s <= 0.49/L
EAG     : s <= 0.25/L
FOGDA   : s <= 0.10/L.
```

Hence, for the reported EG and OGDA experiments,

```text
s_EG   = 0.48/L
s_OGDA = 0.48/L.
```

A typical OGDA call is

```matlab
tune = struct();

tune.method = 'OGDA';
tune.s      = 0.48/P.L;
tune.L      = P.L;
tune.maxit  = inf;

[x,f,info] = MonotoneSolvers(P.funE,P.x,st,tune);
```

Similarly, EG is selected by

```matlab
tune.method = 'EG';
```

---

## 4. Driver for the representative learning-based experiments

The folder `examples/` contains

```text
examples/driverME.m
```

The script `driverME.m` reproduces the representative learning-based
experiments used to illustrate the numerical behavior of the methods.

It calls the existing solver files

```text
DFSdir.m
MonotoneSolvers.m
```

directly rather than redefining the algorithms.

### 4.1 Representative learning problems

The driver uses one representative problem from each of the three
learning-based application classes:

```text
MARL
Robust adversarial learning
GAN
```

With the current generated `TE.mat`, the selected problems are

```text
ME54_marl_discounted_markov_v4

ME71_robust_adversarial_adversarial_v1

ME65_gan_categorical_logistic_gan_v4
```

These correspond respectively to representative:

```text
MARL : variant 4

RAL  : variant 1

GAN  : variant 4.
```

### 4.2 Methods used by the driver

The current representative comparison uses

```text
PLS-S-Z2
EG
OGDA.
```

Internally, `PLS-S-Z2` is selected in `DFSdir.m` by

```matlab
tune.state = '3HS';
```

### 4.3 Benchmark stopping conditions

The stopping quantity used by the benchmark driver is the residual merit

```text
f(x) = 0.5 ||E(x)||^2.
```

The stopping limits are

```text
f(x) <= 1e-5

nf   <= 20000

sec  <= 300.
```

The stopping mechanism is implemented in the operator-evaluation wrapper
inside `driverME.m`. Consequently, every evaluation of `E(x)` is counted
consistently by the driver.

### 4.4 Quantities recorded by the driver

For every solver and representative problem, the driver records

```text
number of residual evaluations

merit function:
    0.5 ||E(x)||^2

residual norm:
    ||E(x)||

min--max objective error:
    |Phi(x)-Phi(x*)|

distance to the known solution:
    ||x-x*||.
```

The corresponding histories are stored in the MATLAB structure

```matlab
RESULT
```

including fields of the form

```matlab
RESULT.(pname).(fieldSolver).nf

RESULT.(pname).(fieldSolver).f

RESULT.(pname).(fieldSolver).res

RESULT.(pname).(fieldSolver).objerr

RESULT.(pname).(fieldSolver).dist

RESULT.(pname).(fieldSolver).x

RESULT.(pname).(fieldSolver).ffinal

RESULT.(pname).(fieldSolver).info
```

### 4.5 Figures

For each representative learning problem, `driverME.m` produces convergence
figures with the number of residual evaluations on the horizontal axis.

The recorded quantities include the merit-function convergence

```text
log10(0.5 ||E(x)||^2)
```

and the min--max objective error

```text
log10(|Phi(x)-Phi(x*)|).
```

The figures are exported in high-resolution graphical formats for use in the
numerical presentation.

### 4.6 Running the driver

From the repository root, first add the relevant folders to the MATLAB path:

```matlab
addpath('solvers');
addpath('TE');
addpath('examples');
```

Then run

```matlab
driverME
```

The driver loads

```text
TE/TE.mat
```

uses the stored Lipschitz constants and initial points, executes the selected
solvers, records their convergence histories, and produces the representative
figures.

---

## Reproducibility

The learning-based test environment is generated using the fixed random seed

```matlab
rng(42)
```

through

```text
TE/genTE.m.
```

The generated file

```text
TE/TE.mat
```

is included so that the exact generated test collection can be used directly
without regenerating the problems.

To regenerate the test environment, run

```matlab
genTE
```

from the `TE/` folder.

The resulting `TE.mat` contains self-contained function handles for the
operators and objective functions associated with the generated problems.

For complete reproduction of the numerical experiments, the supplied
Lipschitz constant `P.L`, initial point `P.x`, solver parameters, stopping
conditions, and random seed should be left unchanged.

---

## Basic usage

A minimal MATLAB workflow is:

```matlab
addpath('solvers');
addpath('TE');

load('TE/TE.mat','TE');

names = fieldnames(TE.problem);

P = TE.problem.(names{1});
```

To inspect the selected problem:

```matlab
disp(P.name)
disp(P.dim)
disp(P.L)

fprintf('||E(x0)|| = %.6e\n',norm(P.funE(P.x)));
fprintf('||E(x*)|| = %.6e\n',norm(P.funE(P.xstar)));
```

To run `PLS-S-Z2`:

```matlab
st = struct();

st.prt    = -1;
st.nfmax  = 20000;
st.secmax = 300;

tune = struct();

tune.state = '3HS';
tune.mu    = 0.48/P.L;
tune.L     = P.L;

[x,f,info] = DFSdir(P.funE,P.x,st,tune);
```

To run `PF-S-Z2`:

```matlab
tune = struct();

tune.state = '3HS-F';
tune.L     = P.L;

[x,f,info] = DFSdir(P.funE,P.x,st,tune);
```

To run EG:

```matlab
tune = struct();

tune.method = 'EG';
tune.s      = 0.48/P.L;
tune.L      = P.L;
tune.maxit  = inf;

[x,f,info] = MonotoneSolvers(P.funE,P.x,st,tune);
```

To run OGDA:

```matlab
tune.method = 'OGDA';

[x,f,info] = MonotoneSolvers(P.funE,P.x,st,tune);
```

---

## MATLAB requirements

The numerical codes are written in MATLAB.

The repository uses standard MATLAB functionality for:

- matrix and vector operations;
- spectral norms and singular-value decompositions;
- MAT-file storage;
- random-number generation;
- plotting and figure export.

No projection onto box constraints is used by the proposed methods or the
comparison methods; all algorithms are applied directly to the unconstrained
monotone equations.

---

## Citation

If you use the `PLS-S`/`PF-S` implementation, the learning-based test
environment, or the supplementary numerical material, please cite the
associated paper:

**Morteza Kimiaei, Shima Shabani, and Michael Breuß,  
Subspace methods for min-max problems.**

A complete bibliographic entry will be added after publication.

---

## License

This repository is distributed under the MIT License. See

```text
LICENSE
```

for details.
