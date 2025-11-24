# mars-steg

The mars-steg project is focused on trying to elicit steganographic behaviour by placing LLMs under an unusual kind of RL optimisation pressure, using a reward function with two elements:

1. A reward for successfully completing a task;
2. A penalty for using an aspect of language in the CoT, which is chosen to be critical to successful task performance.

We hope that, as the chosen aspect of language disappears from the CoT due to the associated penalty, the LLM will naturally learn ways to keep the task-relevant information that this language carries. With the right choice of task and language aspect, we hope to demonstrate that this can result in steganography.

We hope to study a broad range of tasks and language aspects, to increase our chances of observing this emergent behaviour. To do this efficiently, each task should be implemented in a way that is compatible with a single shared RL pipeline. 


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features
- 🔥 Fast and lightweight
- 🔄 Supports multiple file formats
- 🔧 Customizable via settings


## Installation
### Prerequisites
- See [requirements.txt](requirements.txt) for dependencies.

### Steps
```sh
git clone https://github.com/puria-radmard/mars-steg.git
cd mars-steg
pip install -r requirements.txt
```

## Usage
Run the following command to run the training script:
```sh
run ./run_math.sh
```


## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
🚧 _In Construction_

## Acknowledgments
🚧 _In Construction_




