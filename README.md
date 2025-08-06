# Speech-to-Text Translation
Final project for NNAA: a sequence-to-sequence speech-to-text translation model implemented from scratch

## Folder Structure
- `data/`: Preprocessed data and download scripts
- `models/`: Model components like encoder, decoder, and attention
- `scripts/`: Python scripts for training, evaluation, and inference
- `notebooks/`: Jupyter notebooks for experimentation and debugging
- `utils/`: Utility functions for data loading, metrics, etc.
- `results/`: Output predictions, visualizations, and analysis
- `tests/`: Unit tests for model components

## Requirements



## Team Members & Contributions

| Name    | GitHub ID      | Main Contributions                                  |
|---------|----------------|------------------------------------------------------|
| Siyeon Kim  | @sy-grace     |     |
| Anthony   | @az315     |     |
| Nandita    | @       |    |

Main Contributions Example:
- Encoder module, training loop (scripts/train.py)
- Data preprocessing, evaluation (data/, scripts/)
- Decoder module with attention (models/decoder.py)

> This table will be updated continuously as the project progresses.  
> Each member contributes via feature branches and pull requests.

## Usage

```bash
python scripts/train.py      # Train the model
python scripts/evaluate.py   # Evaluate the model
python scripts/predict.py    # Inference on new inputs
```

## Branching Strategy

We follow a simplified Git workflow for team collaboration:

- `main`: Stable release branch. Only the team lead merges into this branch.
- `dev`: Integration branch for collaborative work.
- `feature/<name-task>`: Individual branches for each team member’s task.

### Branch Naming Convention
Use the following format:

```
feature/<your-name>-<task-name>
```

Examples:

- `feature/jiyoon-decoder`
- `feature/minsu-data-loader`
- `feature/sora-attention`

### Workflow

1. Create a new feature branch:
    ```bash
    git checkout -b feature/your-name-task
    ```

2. Push your changes:
    ```bash
    git push origin feature/your-name-task
    ```

3. Create a Pull Request to merge into `dev`.

4. The team reviews and merges `dev` into `main` after stability is confirmed.

## License
This project is licensed under the MIT License.
