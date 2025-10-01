# Speech-to-Text Translation
Final project for NNAA: a sequence-to-sequence speech-to-text translation model implemented from scratch

## Folder Structure
- `data/`: Preprocessed data and download scripts
- `models/`: Model and artifacts generated during training
- `scripts/`: Python scripts for training, evaluation, and inference
- `notebooks/`: Jupyter notebooks for experimentation and debugging
- `utils/`: Utility functions for data loading, metrics, etc.
- `results/`: Output predictions, visualizations, and analysis
- `tests/`: Unit tests for model components

## Requirements
To set up the conda environment, run:

```bash
conda env create -f environment.yml
```
```bash
conda activate stt_translation
```
```bash 
pip install -r requirements.txt
```



## Team Members & Contributions

| Name    | GitHub ID      | Main Contributions                                  |
|---------|----------------|------------------------------------------------------|
| Siyeon Kim  | @sy-grace     |     |
| Anthony   | @az315     |     |
| Nandita    | @       |    |

Main Contributions Example:
- Training loop (scripts/trainer.py)
- Data Loader (scripts/data_loader.py)
- Data preprocessing, evaluation (data/, scripts/)
- Encoder-Decoder module with attention (models/encoder_decoder_transformer.py)

> This table will be updated continuously as the project progresses.  
> Each member contributes via feature branches and pull requests.

## Config
### Setting environment variables to the data directory
```
setx COMMON_VOICE_BASE_PREPROCESSED_DATA_DIR "path/to/data"        # Windows "path/to/data" should be replaced with the actual path in your device
export COMMON_VOICE_BASE_PREPROCESSED_DATA_DIR="path/to/data"      # macOS/Linux
```
## Usage

```bash
python main.py --mode preprocess # Preprocess the data, only needed if the existing preprocessed data is not used
python main.py --mode train      # Train the model, set training subset to use in config.py (e.g., `subset_fraction: float = 0.01`, for 1% of the data)
python main.py --mode evaluate    # Evaluate the model
python main.py --mode infer     # Inference on new audio inputs
```

## Branching Strategy

We follow a simplified Git workflow for team collaboration:

- `main`: Stable release branch. Only the team lead merges into this branch. Use dev for pull requests.
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
