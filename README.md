# Music Duplicate Finder

A Python script for finding duplicate and similar music files using spectrogram analysis with GPU acceleration support.

## Features

- **GPU Acceleration**: Uses CUDA (RTX 4090) for faster processing when available
- **Multi-format Support**: MP3, WAV, FLAC, M4A, AAC, OGG, APE, WMA, ALAC, AIFF
- **Advanced Analysis**: Spectrogram and MFCC feature extraction
- **Dual Detection**: Exact hash duplicates and similar audio content
- **Batch Processing**: Efficient processing of large music libraries

## Usage

```bash
python music_duplicate_finder.py /path/to/music/library
```

## Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `library_path`| Path to music library directory | (required) |
| `--threshold`| Similarity threshold (0.0-1.0) | 0.95 |
| `--output`| Output JSON filename | `music_duplicates.json` |
| `--cpu`| Force CPU mode (disable GPU) | `False` |
| `--batch-size`| Processing batch size | `1000` |

### Examples
**Example 1:** Basic usage
```bash
python music_duplicate_finder.py "D:\Music"
```

**Example 2:** Custom threshold and output:
```bash
python music_duplicate_finder.py "/home/user/music" --threshold 0.9 --output "results.json"
```

**Example 3:** Force CPU mode:
```bash
python music_duplicate_finder.py "/music/library" --cpu
```

## Requirements
```txt
librosa>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
cupy-cuda12x>=12.0.0
cuml-cuda12>=23.8.0
cudf-cuda12>=23.8.0
```

## GPU Requirements (Optional)
    NVIDIA GPU with CUDA support
    cupy, cuml, and cudf packages for CUDA acceleration
