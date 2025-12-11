# 显存使用分析与优化方案

## 显存占用分析

在执行 `python3 start_code.py --device 3 --sub_device 3 --method mend --model qwen --ds vqa` 时，显存占用主要来自以下几个部分：

### 1. 模型本身（最大部分，约14GB）
- **位置**：`BaseTrainer.__init__` -> `get_model()` 
- **大小**：qwen2-vl-7b在bfloat16下约14GB
- **设备**：全部在`config.device`上

### 2. edited_model（约14-20GB，峰值）
- **位置**：`MEND.edit()` -> `monkeypatch()` 
- **问题**：使用higher库的monkeypatch会保留完整计算图，显存占用约为原模型的1.5-2倍
- **设备**：当前全部在`config.device`上

### 3. 多次前向传播的激活值（约2-5GB）
- **位置**：`MultimodalTrainer.edit_step()`
- **前向传播次数**：7次
  1. `base_outputs = self.model(batch["loc"])`
  2. `base_image_outputs = self.model(batch["loc_image"])`
  3. `post_edit_outputs = edited_model(batch["edit_outer"])`
  4. `inner_edit_outputs = edited_model(batch["edit_inner"])`
  5. `post_image_edit_outputs = edited_model(batch["edit_outer_image"])`
  6. `post_base_outputs = edited_model(batch["loc"])`
  7. `post_image_base_outputs = edited_model(batch["loc_image"])`
- **问题**：训练模式下会保存梯度，激活值占用显存

### 4. 梯度计算（约2-3GB）
- **位置**：`MEND.edit()` -> `loss.backward()`
- **问题**：需要保存所有中间激活的梯度

### 5. original_model副本（如果train_base=True，约14GB）
- **位置**：`BaseTrainer.__init__`
- **当前状态**：代码中`train_base=False`，不创建副本

## 优化方案

### 方案1：将edited_model移到sub_device（推荐）
- **优点**：直接利用第二张卡，显存减半
- **实现**：修改`MEND.edit()`，将edited_model的计算移到sub_device

### 方案2：模型分片（device_map）
- **优点**：自动分配模型到多卡
- **缺点**：需要修改模型加载逻辑，可能影响训练

### 方案3：减少激活值占用
- 使用gradient checkpointing
- 及时清理中间变量
- 将部分前向传播移到sub_device

### 方案4：混合策略（最佳）
- edited_model在sub_device
- 部分前向传播在sub_device
- 及时清理不需要的中间变量

