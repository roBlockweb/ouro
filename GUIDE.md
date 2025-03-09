# Ouro Setup Guide

This guide will help you set up all the prerequisites needed to run Ouro, the privacy-first local RAG system.

## 1. Hugging Face Setup (Recommended)

While Ouro will work without Hugging Face authentication, logging in is **highly recommended** for better model access and to avoid download rate limits. Follow these simple steps:

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
  - M1 Optimized: 4-6GB RAM (Apple Silicon only)
  - Large model: 8-10GB RAM
  - Very Large model: 12-16GB+ RAM
- **Storage**: At least 5GB free space for model downloads

## 3. Performance Optimization

### Apple Silicon (M1/M2 Macs)

Ouro includes special optimizations for Apple Silicon processors. To use them:

```bash
./run.sh --m1
```

This will:
- Use models that perform well on the MPS (Metal Performance Shaders) backend
- Apply memory optimizations specific to M1/M2
- Configure PyTorch for optimal Apple Silicon performance

### Faster Responses

If you want faster responses (at the cost of some quality), use:

```bash
./run.sh --fast
```

Or toggle fast mode inside the application with the `fast_mode` command.

### Memory-Constrained Systems

For systems with limited RAM:

```bash
./run.sh --small
```

This will use the smallest model (1.1B parameters) which requires only about 2-4GB of RAM.

### Combining Optimizations

You can combine optimization flags:

```bash
./run.sh --m1 --fast  # For fastest performance on Apple Silicon
./run.sh --small --fast  # For minimum resource usage
```

## 4. Using Advanced Features

### Conversation Memory

Ouro remembers previous conversation turns to provide more contextual responses. You can:

- Clear memory: `clear_memory` command
- Toggle conversation history: `toggle_history` command
- Configure memory depth: Edit `memory_turns` in config.py or use the `--memory-turns` flag

### Adaptive Learning

Ouro can learn from past conversations:

1. Enable conversation saving with the `save_conversations` option
2. Use the `learn` command to process saved conversations
3. New insights will be added to your knowledge base automatically

### System Information

To see your current configuration and system details:

```
system_info
```

This shows your hardware, current model, and optimization settings.

## 5. Troubleshooting

### Issue: "Command not found: huggingface-cli"

Solution: Ouro will install this for you when you run it. If you want to install it manually:

```bash
pip install huggingface_hub
```

### Issue: Model Download Failures

Solution: 
- While Ouro will now run without Hugging Face authentication, you may encounter download issues without logging in.
- Verify you used the correct token from https://huggingface.co/settings/tokens
- Try logging in manually: `huggingface-cli login`
- Some models require explicit acceptance of terms on the Hugging Face website
- Check your internet connection

### Issue: Application Crashes or Runs Out of Memory

Solution:
- Choose a smaller model option (Small or Medium)
- Close other applications to free up memory
- Use quantization with the `quantize` option in config.py
- Enable fast mode with `fast_mode` command or `--fast` flag

### Issue: Slow Performance on Apple Silicon

Solution:
- Use the M1 Optimized model with `./run.sh --m1`
- Ensure you're using PyTorch 2.0+ which includes MPS optimizations
- Check that your model is using the MPS device with `system_info`

## Next Steps

Once you've completed these setup steps, return to the README.md for instructions on how to run Ouro.

Happy querying!