# Ouro Setup Guide

This guide will help you set up all the prerequisites needed to run Ouro, the privacy-first local RAG system.

## 1. Hugging Face Setup (Required)

Ouro requires a Hugging Face account to download language models. Follow these simple steps:

### Create a Hugging Face Account

1. Visit [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up with your email or using GitHub/Google
3. Verify your email address

### Generate an Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click the "New token" button
3. Name your token (e.g., "Ouro")
4. Select "Read" for permission level
5. Click "Generate token"
6. Copy your token (it starts with "hf_")

### Log in via Command Line

Open your terminal and run:

```bash
huggingface-cli login
```

When prompted, paste your token and press Enter. 

If the command isn't found, Ouro will help you install it when you first run the application.

## 2. System Requirements

- **Python**: 3.9 or newer (3.10 or 3.11 recommended)
- **Memory**: 
  - Small model: 2-4GB RAM
  - Medium model: 4-6GB RAM
  - Large model: 6-8GB RAM
  - Very Large model: 12-16GB+ RAM
- **Storage**: At least 5GB free space for model downloads

## 3. Troubleshooting

### Issue: "Command not found: huggingface-cli"

Solution: Ouro will install this for you when you run it. If you want to install it manually:

```bash
pip install huggingface_hub
```

### Issue: Authentication Error

Solution: 
- Verify you used the correct token from https://huggingface.co/settings/tokens
- Try logging in manually: `huggingface-cli login`
- Check your internet connection

### Issue: Application Crashes or Runs Out of Memory

Solution:
- Choose a smaller model option (Small or Medium)
- Close other applications to free up memory
- If using Python 3.13, try the compatibility options in README.md

## Next Steps

Once you've completed these setup steps, return to the README.md for instructions on how to run Ouro.

Happy querying!