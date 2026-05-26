---
name: 图像文本检索技能
description: 使用任意 CLIP 检查点（checkpoint）构建图像嵌入（embedding）索引；支持按文本查询与按图像查询
version: 1.0.0
phase: 4
lesson: 18
tags: [CLIP, 检索, FAISS, 零样本]
---

# 图像文本检索器（Image-Text Retriever）

利用 CLIP 嵌入（embedding）将图像文件夹转换为可搜索的索引。

## 适用场景

- 在内部目录上构建零样本（zero-shot）图像搜索功能。
- 通过嵌入距离对高度相似的图像进行去重。
- 无需标注数据集即可快速构建“查找相似图像”组件。

## 输入参数

- `image_folder`：图像文件所在的目录。
- `clip_model`：HuggingFace 模型 ID，例如 `openai/clip-vit-base-patch32` 或 `google/siglip-base-patch16-224`。
- `index_type`：索引类型，可选 flat（扁平） | IVF | HNSW。
- `embedding_dim`：由模型自动推断得出。

## 操作步骤

1. 加载 CLIP 模型及其预处理器（preprocessor）。
2. 对文件夹中的所有图像进行批量编码。将嵌入向量保存为形状为 (N, D) 的 float32 数组及对应的文件名列表。
3. 基于嵌入向量构建 FAISS 索引。对 L2 归一化（L2-normalised）后的向量使用内积（inner-product）计算余弦相似度（cosine similarity）。
4. 暴露两个查询接口：
   - `search_by_text(text, k)` — 对文本进行嵌入编码并执行搜索。
   - `search_by_image(image_path, k)` — 对图像进行嵌入编码并执行搜索。

## 输出模板

import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import faiss


class ImageTextRetriever:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim
        self.index = None
        self.filenames = []

    @torch.no_grad()
    def _encode_images(self, paths, batch=16):
        embs = []
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = self.processor(images=imgs, return_tensors="pt")
            out = self.model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            embs.append(out.cpu().numpy())
        return np.concatenate(embs).astype(np.float32)

    @torch.no_grad()
    def _encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        out = self.model.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.cpu().numpy().astype(np.float32)

    def build_index(self, folder, index_type="flat"):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        self.filenames = sorted(files)
        embs = self._encode_images(self.filenames)
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(256, max(4, len(embs) // 32))
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embs)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search_by_text(self, text, k=5):
        q = self._encode_text([text])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]

    def search_by_image(self, image_path, k=5):
        q = self._encode_images([image_path])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]

## 报告输出

[retriever]
  model:          <name>
  num_images:     <int>
  dim:            <int>
  index_type:     flat | IVF | HNSW
  index_size_mb:  <float>

## 规则

- 在构建索引前，始终对嵌入向量（embeddings）进行 L2 归一化（L2-normalization）；FAISS 对归一化向量计算的内积（inner product）等同于余弦相似度（cosine similarity）。
- 对于少于 10 万张图像，`IndexFlatIP`（精确搜索）是最简单且最快的选择。
- 对于 10 万至 1000 万张图像，`IndexIVFFlat` 是标准的权衡方案。
- 对于超过 1000 万张图像，请使用 HNSW 或乘积量化（product quantization）变体。
- 绝不要在每次查询时重建索引；只需计算一次嵌入，即可进行多次搜索。