# VQ-VLA-Code (Stripped Source Release)

> 这是原始 VQ-VLA 项目的“精简版”代码仓库，只包含核心源码与脚本，已经移除所有大型数据集、训练输出、wandb 日志以及权重文件。方便查阅与二次开发，不直接用于复现完整训练。

## 包含内容
- `prismatic/` 与 `action_vqvae/` 等核心模型与工具代码
- 训练 / 微调脚本：`scripts/`, `vla-scripts/`, `train_vae/`
- 基础配置与构建文件：`pyproject.toml`, `Makefile`, `LICENSE`, `.gitignore`
- 示例与辅助工具若体积较小（如 `assets/` 中的流程示意图）

## 不包含内容
- 大型数据集目录：`datasets/`, `data/` 等
- 训练产物与日志：`outputs/`, `wandb/`, `rollouts/`, `eval_logs/`
- 模型权重与检查点：`*.pt`, `*.pth`, `*.safetensors`, `*.ckpt` 等

如需获取完整权重与数据，请参考原始论文与 Hugging Face 资源：
- 论文: https://arxiv.org/abs/2507.01016
- 模型/权重: https://huggingface.co/VQ-VLA

## 安装 (精简版)
```bash
# 建议使用 Python 3.10
conda create -n vqvla python=3.10 -y
conda activate vqvla

# 安装核心依赖 (根据 GPU/CUDA 版本调整 PyTorch)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 克隆本精简仓库
git clone https://github.com/lcj111wl/VQ-VLA-Code.git
cd VQ-VLA-Code
pip install -e .
```
若需训练 / 微调全量模型，请再参考原始仓库的 README 说明补齐数据与权重。

## 目录概览
```text
prismatic/            # 模型与训练相关模块
scripts/              # 额外数据处理与 VQ 训练脚本
vla-scripts/          # 微调、部署、验证脚本
train_vae/            # VQ-VAE 训练入口脚本
assets/               # 轻量图片或示意文件（若保留）
pyproject.toml        # 项目依赖与打包配置
Makefile              # 格式化 / 清理辅助命令
LICENSE               # MIT 许可
.gitignore            # 忽略大数据与权重模式
```

## 快速测试
```bash
python -c "import prismatic, importlib; print('Loaded prismatic version OK')"
```

## 贡献
欢迎提交 issue 或 PR 来改进该精简版；若涉及新增大型数据或权重，请优先使用外部存储与下载脚本方式而非直接提交。

## 许可证
MIT License，详见 `LICENSE`。

## 引用
如使用本代码，请引用原始论文：
```bibtex
@inproceedings{wang25vqvla,
  title={VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers},
  author={Yating Wang and Haoyi Zhu and Mingyu Liu and Jiange Yang and Hao-Shu Fang and Tong He},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
