"""
Configuration for the Multi-Agent Bias Detection System.

All vLLM server URLs and model IDs are set here. Override any value
with the corresponding environment variable before running.
"""

import os
from typing import Any, Dict
from pathlib import Path


def get_config() -> Dict[str, Any]:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    dirs = {
        'base':              base_dir,
        'data':              os.path.join(base_dir, 'data'),
        'balanced_datasets': os.path.join(base_dir, 'data', 'balanced_datasets'),
        'models':            os.path.join(base_dir, 'models'),
        'outputs':           os.path.join(base_dir, 'outputs'),
        'outputs_regular':   os.path.join(base_dir, 'outputs', 'ensemble_outputs'),
        'outputs_small':     os.path.join(base_dir, 'outputs', 'ensemble_outputs_small'),
    }

    config = {
        'dirs': dirs,

        # ── vLLM server configuration ──────────────────────────────────────
        # Each model runs as a separate vLLM server instance.
        # Override with environment variables for custom deployments.
        #
        # Quick start (small ensemble):
        #   vllm serve meta-llama/Llama-3.2-3B-Instruct      --port 8001 --api-key token-abc123
        #   vllm serve Qwen/Qwen3-4B                          --port 8002 --api-key token-abc123
        #   vllm serve mistralai/Mistral-7B-Instruct-v0.3     --port 8003 --api-key token-abc123
        #
        # Quick start (regular ensemble):
        #   vllm serve Qwen/Qwen3-14B                         --port 8001 --api-key token-abc123
        #   vllm serve openai/gpt-oss-20b                     --port 8002 --api-key token-abc123
        #   vllm serve mistralai/Mistral-Small-Instruct-2409  --port 8003 --api-key token-abc123
        'vllm': {
            'api_key': os.environ.get('VLLM_API_KEY', 'token-abc123'),

            # Any of the three slots can be swapped for a different model:
            # point the slot's *_URL at your server and set *_MODEL to the
            # served model ID. No code changes are needed — every labeler
            # speaks the same OpenAI-compatible protocol and JSON contract.
            'small_ensemble': {
                'llama32': {
                    'base_url': os.environ.get('VLLM_LLAMA_URL',   'http://localhost:8001/v1'),
                    'model_id': os.environ.get('VLLM_LLAMA_MODEL', 'meta-llama/Llama-3.2-3B-Instruct'),
                },
                'qwen3': {
                    'base_url':        os.environ.get('VLLM_QWEN_URL',    'http://localhost:8002/v1'),
                    'model_id':        os.environ.get('VLLM_QWEN_MODEL',  'Qwen/Qwen3-4B'),
                    'enable_thinking': os.environ.get('VLLM_QWEN_THINKING', '1') not in ('0', 'false', 'False'),
                },
                'mistral': {
                    'base_url': os.environ.get('VLLM_MISTRAL_URL',   'http://localhost:8003/v1'),
                    'model_id': os.environ.get('VLLM_MISTRAL_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3'),
                },
            },

            'regular_ensemble': {
                'qwen3': {
                    'base_url':        os.environ.get('VLLM_QWEN14B_URL',    'http://localhost:8001/v1'),
                    'model_id':        os.environ.get('VLLM_QWEN14B_MODEL',  'Qwen/Qwen3-14B'),
                    'enable_thinking': os.environ.get('VLLM_QWEN14B_THINKING', '1') not in ('0', 'false', 'False'),
                },
                'gptoss': {
                    'base_url': os.environ.get('VLLM_GPTOSS_URL',   'http://localhost:8002/v1'),
                    'model_id': os.environ.get('VLLM_GPTOSS_MODEL', 'openai/gpt-oss-20b'),
                },
                'mistral': {
                    'base_url': os.environ.get('VLLM_MISTRAL22B_URL',   'http://localhost:8003/v1'),
                    'model_id': os.environ.get('VLLM_MISTRAL22B_MODEL', 'mistralai/Mistral-Small-Instruct-2409'),
                },
            },
        },

        # ── Dataset configurations ─────────────────────────────────────────
        'datasets': {
            'baly': {
                'path':            os.path.join(dirs['balanced_datasets'], 'balanced_baly'),
                'has_ground_truth': True,
                'bias_field':      'bias',        # Numeric: 0=Left, 1=Center, 2=Right
            },
            'budak': {
                'path':            os.path.join(dirs['balanced_datasets'], 'balanced_budak'),
                'has_ground_truth': True,
                'bias_field':      'bias_text',   # Text: 'left', 'center', 'right'
            },
            'ad_fontes': {
                'path':            os.path.join(dirs['balanced_datasets'], 'balanced_ad_fontes'),
                'has_ground_truth': True,
                'bias_field':      'Bias',        # Numeric: -42 to +42
            },
            'custom': {
                'path':            os.path.join(dirs['balanced_datasets'], 'custom_100_per_outlet'),
                'has_ground_truth': False,
                'description':     'Custom dataset for outlet-level evaluation',
            },
        },

        # ── File paths ────────────────────────────────────────────────────
        'files': {
            'allsides_ratings': os.path.join(dirs['data'], 'allsides', 'AllSides_Rating.csv'),
        },

        # ── Ensemble runtime settings ──────────────────────────────────────
        'ensemble': {
            'max_discussion_rounds':  8,
            'convergence_threshold':  0.5,
            'discussion_timeout':     1800,   # seconds per article
        },

        'batch_processing': {
            'regular_batch_size': 3,
            'small_batch_size':   8,
            'timeout_per_batch':  7200,       # seconds
        },

        # ── HuggingFace token (for downloading gated models via CLI) ───────
        # Set:  $env:HF_TOKEN="hf_..."   (PowerShell)
        #       export HF_TOKEN="hf_..."  (Linux/macOS)
        'huggingface': {
            'token': os.environ.get('HF_TOKEN', ''),
        },
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return config


if __name__ == "__main__":
    import json
    cfg = get_config()
    print("Configuration loaded successfully!")
    print(json.dumps(cfg, indent=2, default=str))
