# ComfyUI-Ideogram4

ComfyUI custom nodes for Ideogram 4.0.

Ideogram 4 only works with structured JSON captions. The usual workflow is:

```text
Magic Prompt -> Generate -> Preview Image or Save Image
Pipeline Loader -> Generate
```

Magic Prompt turns a normal text prompt into the structured caption format.
Generate runs the local Ideogram 4 model.

This repo is the ComfyUI wrapper only. Model weights are downloaded from Hugging
Face, and the core inference package is installed from
`https://github.com/ideogram-oss/ideogram-4`.

## Example Workflow

Download and drag this image into ComfyUI to load the example workflow:

<img width="2048" height="2048" alt="ComfyUI_00002_" src="https://github.com/user-attachments/assets/7c659f00-a1e5-407d-bfc4-10100f942eca" />

## Install

> **Note:** Ideogram 4 requires `torch>=2.11`. Installing the requirements may
> upgrade the PyTorch already in your ComfyUI environment. If you depend on a
> specific CUDA build, install a matching `torch>=2.11` first so the upgrade does
> not replace it with an incompatible build.

### ComfyUI-Manager or Comfy Registry

Install `ComfyUI-Ideogram4` through ComfyUI-Manager or the Comfy Registry. The
Python requirements are installed automatically.

Restart ComfyUI after installation.

### Manual Install

Clone the node into `custom_nodes`:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/ideogram-oss/ComfyUI-Ideogram4.git ComfyUI-Ideogram4
```

Install requirements in the same Python environment that runs ComfyUI:

```bash
cd /path/to/ComfyUI
source venv/bin/activate
pip install -r custom_nodes/ComfyUI-Ideogram4/requirements.txt
```

Start or restart ComfyUI:

```bash
cd /path/to/ComfyUI
source venv/bin/activate   # or however you normally launch ComfyUI
python main.py
```

Open `http://127.0.0.1:8188`.

> **Remote machine** Start with `--listen 0.0.0.0` and forward port `8188` to your
> local machine. ComfyUI has no built-in authentication, so only do this on a
> trusted network or behind a tunnel/VPN — never expose it to the open internet.


## Set API Keys

After ComfyUI starts, set all credentials from one place:

```text
Settings -> Ideogram 4.0 -> API Keys -> Manage API keys...
```

Fill the fields you need:

| Field | Purpose |
| --- | --- |
| `IDEOGRAM_API_KEY` | Required for the default Magic Prompt provider, `ideogram`. |
| `OPENROUTER_API_KEY` | Required for the Magic Prompt provider, `openrouter`. |
| `HF_TOKEN` | Required for the Hugging Face weight repo. |

Only the key for the magic prompt provider you choose is needed. You can use any
node to generate the magic prompt. This node is provided for ease of use.

Keys are saved to `custom_nodes/ComfyUI-Ideogram4/ideogram_config.json`. The file
is git-ignored and is not embedded in workflows or output images. Settings
changes apply on the next run; you do not need to restart ComfyUI.

You can also create the config file by hand:

```json
{
  "IDEOGRAM_API_KEY": "your-ideogram-api-key",
  "OPENROUTER_API_KEY": "your-openrouter-api-key",
  "HF_TOKEN": "your-hugging-face-token"
}
```

For headless or cloud setups, environment variables also work:

```bash
export IDEOGRAM_API_KEY=your-ideogram-api-key
export OPENROUTER_API_KEY=your-openrouter-api-key
export HF_TOKEN=your-hugging-face-token
```

The config file takes priority over environment variables.

### Hugging Face Access

The public weight repos are:

- `https://huggingface.co/ideogram-ai/ideogram-4-nf4`
- `https://huggingface.co/ideogram-ai/ideogram-4-fp8`

If the weights are gated, open the selected Hugging Face repo in a browser while
signed in and accept its terms once.

Create a token at `https://huggingface.co/settings/tokens`.
A read-only token scoped to the Ideogram repos is recommended.

This wrapper loads weights through the core `ideogram4` package using the
Hugging Face cache layout.

By default, Hugging Face stores downloaded weights under:

```text
~/.cache/huggingface
```

To use a different Hugging Face cache location, set `HF_HOME` before starting
ComfyUI:

```bash
export HF_HOME=/path/to/huggingface/cache
```

## Quick Workflow

1. Add `Ideogram 4.0 Magic Prompt`.
2. Type a normal prompt.
3. Leave `magic_prompt_provider` on `ideogram`.
4. Leave `verify_json` on.
5. Add `Ideogram 4.0 Pipeline Loader`.
6. Pick `4.0 NF4` for the first run.
7. Add `Ideogram 4.0 Generate`.
8. Connect `Magic Prompt.expanded_prompt` to `Generate.prompt`.
9. Connect `Pipeline Loader.pipeline` to `Generate.pipeline`.
10. Set the same `width` and `height` on Magic Prompt and Generate.
11. Connect `Generate.image` to `Preview Image.images` or `Save Image.images`.
12. Leave `sampler_preset` on `4.0 Default 20` for the first run.

Use `4.0 Turbo 12` for faster results, or `4.0 Quality 48` for the highest
quality.

The first run may take longer while Hugging Face downloads the weights and the
pipeline loads them into GPU memory.

## Nodes

### Ideogram 4.0 Magic Prompt

Expands a plain prompt into the structured JSON caption format used by Ideogram
4.

Inputs:

- `prompt`: your normal text prompt.
- `width`, `height`: target image size. Defaults to `2048x2048`; supported
  values are multiples of 16 from 256 to 2048, with aspect ratios up to 6:1.
  Use the same values on Generate.
- `magic_prompt_provider`: `ideogram` or `openrouter`. `ideogram` is the
  default.
- `openrouter_model`: only used with `openrouter`; leave blank for `ideogram`.
- `verify_json`: checks the expanded prompt with the core caption verifier.
  Keep this on unless you are debugging raw OpenRouter output.

Output:

- `expanded_prompt`

Notes:

- `ideogram` uses `IDEOGRAM_API_KEY` and calls Ideogram's **free** Magic Prompt API.
- `openrouter` uses `OPENROUTER_API_KEY` and the `openrouter_model` value.
- If Magic Prompt raises, image generation has not started yet. Fix the key,
  provider, model, or JSON issue first.
- You can queue Magic Prompt by itself to inspect the expanded JSON.

### Ideogram 4.0 Pipeline Loader

Loads and caches the Ideogram 4 pipeline.

Inputs:

- `model_weights`: `4.0 NF4` or `4.0 FP8`.

Output:

- `pipeline`

Notes:

- Loading a different weight type clears the cached pipeline first.
- Device, dtype, tokenizer, and internal Hugging Face paths come from the core
  `ideogram4` package.

### Ideogram 4.0 Generate

Runs the image model and returns a Comfy `IMAGE`. Connect it to `Preview Image`
or `Save Image` to view or save the result.

Inputs:

- `pipeline`: connect from `Ideogram 4.0 Pipeline Loader`.
- `prompt`: Magic Prompt output or hand-written Ideogram JSON.
- `width`, `height`: defaults to `2048x2048`; supported values are multiples
  of 16 from 256 to 2048, with aspect ratios up to 6:1.
- `sampler_preset`: `4.0 Quality 48`, `4.0 Default 20`, `4.0 Turbo 12`, or
  `custom`.
- `seed`: generation seed.

Advanced inputs, used only when `sampler_preset` is `custom`:

- `num_steps`
- `guidance_scale`
- `mu`
- `std`

Plain text can be passed directly to Generate, but is not recommended. The
model was trained on structured captions, so use Magic Prompt or JSON for normal
use.

## Sampler Presets

The named sampler presets come from the official
`ideogram4.sampler_configs.PRESETS` registry. Each one bundles a step count, a
per-step CFG guidance schedule, and the logit-normal noise schedule parameters
`mu` and `std`.

Named presets ignore the advanced custom fields. `4.0 Default 20` is the ComfyUI
default.

In the core code, guidance schedules are stored in loop-index order, where index
0 is the final sampling step. The table below writes the same schedules in run
order: main sampling steps first, then the lower-guidance polish steps.

| Comfy preset | Official preset | Steps | CFG schedule | `mu` | `std` | Use |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `4.0 Quality 48` | `V4_QUALITY_48` | 48 | 45 steps @ gw=7, then 3 polish steps @ gw=3 | 0.0 | 1.5 | Best quality |
| `4.0 Default 20` | `V4_DEFAULT_20` | 20 | 18 steps @ gw=7, then 2 polish steps @ gw=3 | 0.0 | 1.75 | Balanced |
| `4.0 Turbo 12` | `V4_TURBO_12` | 12 | 11 steps @ gw=7, then 1 polish step @ gw=3 | 0.5 | 1.75 | Fast generation |
| `custom` | custom | `num_steps` | constant `guidance_scale` | `mu` | `std` | Experiments |

For the named presets, `gw` means guidance weight. The final lower-guidance
polish steps are part of the official preset schedule; they are not an extra
ComfyUI step.

The custom defaults match the `4.0 Default 20` preset closely, but use a constant
guidance scale instead of the preset's per-step schedule:

```text
num_steps=20
guidance_scale=7.0
mu=0.0
std=1.75
```

Generate shows the resolved settings on the node after it runs.

## Troubleshooting

### Nodes do not show up

- Make sure the folder is `ComfyUI/custom_nodes/ComfyUI-Ideogram4`.
- Install requirements in the ComfyUI Python environment.
- Restart ComfyUI after installing or changing custom nodes.
- Check the terminal for `IMPORT FAILED`.

### `ideogram4` cannot be imported

Install the requirements in the ComfyUI environment:

```bash
cd /path/to/ComfyUI
source venv/bin/activate
pip install -r custom_nodes/ComfyUI-Ideogram4/requirements.txt
```

For local development only, you can also point at the core repo directly:

```bash
export IDEOGRAM4_REPO=/path/to/ideogram-4
```

Set it before starting ComfyUI.

### Hugging Face load errors

- Accept the Hugging Face terms for the selected weight repo in a browser.
- Set `HF_TOKEN` in Settings, `ideogram_config.json`, or the environment.
- Make sure that account has access to the selected weights.
- Set `HF_HOME` to a cache location with enough disk space if the default disk
  is too small.
- Watch the Pipeline Loader status text and terminal logs during first load.

### CUDA out of memory

- Try `4.0 NF4`.
- Stop other GPU jobs.
- Restart ComfyUI to clear cached models.

### Magic Prompt errors

- For `ideogram`, set `IDEOGRAM_API_KEY`.
- For `openrouter`, set `OPENROUTER_API_KEY` and enter an `openrouter_model`.
- If `openrouter` produces bad JSON, try a stronger model.
- Turn off `verify_json` only when you intentionally want raw OpenRouter output
  for debugging.

## License

The ComfyUI node code in this repository is licensed under
[Apache-2.0](LICENSE.md).

The Ideogram 4.0 model weights are not covered by this license. The Ideogram
4.0 model weights are distributed under the Ideogram 4.0 Non-Commercial license
on Hugging Face and require accepting the terms on the model page. Downloading 
or running the weights through this node is subject to the license specified
on Hugging Face.
