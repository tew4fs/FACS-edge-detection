﻿# FACS-edge-detection

by Trevor Williams & Amanuel Anteneh

Project based on [FACS Edge Detection Paper](https://direct.mit.edu/isal/proceedings-pdf/ecal2015/27/398/1903827/978-0-262-33027-5-ch071.pdf)



## Project Setup


### Pipenv
Install pipenv:

```bash

pip install pipenv

```



Install dependencies:

```bash

pipenv install

```



Activate virtual environment: 

```bash

pipenv shell

```


Install pre commit hooks:

```bash

pre-commit install

```


### Makefile

#### Installation for Windows
1. Install [Chocolatey](https://chocolatey.org/install#individual)
2. Run 
```bash

choco install make

```

#### Commands
Build the whole app:
```bash

make

```

Run the formatter:
```bash

make format

```

Run the linter:
```bash

make lint

```

Run the app
```bash

make run

```

Run tests:

```bash

make test

```
