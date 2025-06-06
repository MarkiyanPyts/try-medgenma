# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a MedGemma project that uses Google's MedGemma-4B-IT model for medical image analysis through Hugging Face transformers.

## Development Environment

- Python 3.11+ required
- Uses uv package manager (pyproject.toml configuration)
- Main dependencies: Pillow, PyTorch, and Transformers

## Key Commands

### Setup and Dependencies
```bash
# Install dependencies using uv
uv pip install -e .

# Or install directly
pip install pillow torch transformers
```

### Running the Application
```bash
python main.py
```

## Architecture

The project consists of a single main.py file that:
- Uses Hugging Face's pipeline API for image-text-to-text tasks
- Loads the google/medgemma-4b-it model
- Currently configured to analyze a sample candy image URL

## Important Notes

- The model (google/medgemma-4b-it) is specifically designed for medical image analysis
- First run will download the model weights (~8GB)
- GPU recommended for optimal performance