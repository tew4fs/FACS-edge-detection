# FACS-edge-detection

by Trevor Williams & Amanuel Anteneh



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