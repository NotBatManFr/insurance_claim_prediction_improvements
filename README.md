# Insurance Claim Prediction App

## Prerequisites

To run this project, you will need:
* **Docker:** For running the production-ready container or executing the CI testing suite locally.
* **Miniconda (or Anaconda):** For local development, fast iterative testing, and IDE support.

## Local Development Setup (Using Conda)

We use Conda for local development to ensure fast, reliable access to pre-compiled data science libraries (like `pandas` and `scikit-learn`).

1.  **Create the Conda Environment:**
    Navigate to the project root and create the environment using the provided file.
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the Environment:**
    ```bash
    conda activate insurance_claim_env
    ```
3.  **Run the Application locally:**
    We provide helper scripts that automatically handle Conda activation for you.
    ```bash
    ./run.sh
    ```
4.  **Run the Test Suite locally:**
    Our test suite ensures 100% passing tests and enforces an 80% coverage quality gate.
    ```bash
    ./run_tests.sh
    ```

## Docker & CI/CD Setup

Our Docker configuration is optimized for production and CI/CD pipelines. Unlike our local setup, the `Dockerfile` installs all dependencies *globally* using Conda, removing the need for virtual environments inside the container.

1.  **Build the Docker Image:**
    ```bash
    docker build -t insurance-app:latest .
    ```
2.  **Run the Application via Docker:**
    ```bash
    docker run insurance-app:latest
    ```
3.  **Run the CI Test Suite via Docker:**
    Our CI pipeline (e.g., GitHub Actions) uses this exact command to verify the container integrity before deployment. You can run it locally to mimic the CI pipeline:
    ```bash
    docker run insurance-app:latest sh -c "coverage run -m pytest && coverage report -m --fail-under=80"
    ```

## Repository Cleanup Note
If you are transitioning to this new setup, you can safely delete `requirements.txt` and `requirements-dev.txt`, as `environment.yml` is now the single source of truth for all dependencies.