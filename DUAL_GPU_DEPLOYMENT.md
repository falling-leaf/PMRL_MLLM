# 双卡部署使用说明

## 概述

已对代码进行优化，支持使用`sub_device`参数进行双卡部署，将edited_model的计算移到第二张GPU上，从而减少主卡的显存占用，避免CUDA OOM错误。

## 显存占用分析

### 主要显存占用部分（按大小排序）

1. **模型本身** (~14GB)
   - qwen2-vl-7b在bfloat16下约14GB
   - 位置：`BaseTrainer.__init__` -> `get_model()`

2. **edited_model** (~14-20GB，峰值)
   - 使用higher库的monkeypatch创建，保留完整计算图
   - 位置：`MEND.edit()` -> `monkeypatch()`
   - **优化**：通过sub_device将这部分计算移到第二张卡

3. **多次前向传播的激活值** (~2-5GB)
   - 7次前向传播（base, base_image, post_edit, inner_edit, post_image_edit, post_base, post_image_base）
   - 位置：`MultimodalTrainer.edit_step()`
   - **优化**：edited_model的前向传播在sub_device上进行

4. **梯度计算** (~2-3GB)
   - loss.backward()需要保存中间激活的梯度
   - 位置：`MEND.edit()`

## 使用方法

### 基本用法

```bash
# 使用device 3作为主卡，device 4作为副卡
python3 start_code.py --device 3 --sub_device 4 --method mend --model qwen --ds vqa
```

### 参数说明

- `--device`: 主设备，用于：
  - 模型加载和存储
  - MEND网络（mend模块）
  - 优化器参数
  - 基础前向传播（base_outputs）

- `--sub_device`: 副设备，用于：
  - edited_model的所有前向传播
  - 减少主卡显存峰值

### 工作原理

1. **模型初始化**：模型加载到`device`（主卡）
2. **编辑过程**：在`MEND.edit()`中创建edited_model（functional模型）
3. **前向传播**：在`MultimodalTrainer.edit_step()`中：
   - 基础前向传播在主卡进行
   - edited_model的前向传播在sub_device进行
   - 输出logits移回主卡用于loss计算

## 代码修改说明

### 1. MEND.py

- 添加了`sub_device`配置处理
- 在`__init__`中解析和验证sub_device参数

### 2. MultimodalTrainer.py

- 在`edit_step()`中添加了双卡支持逻辑
- 实现了`move_batch_to_device()`辅助函数
- 将edited_model的所有前向传播移到sub_device
- 添加了显存监控（记录sub_device的显存使用）
- 添加了显存清理（`torch.cuda.empty_cache()`）

## 显存优化效果

### 优化前（单卡）
- 主卡峰值显存：~35-40GB
- 容易OOM

### 优化后（双卡）
- 主卡峰值显存：~20-25GB
- 副卡峰值显存：~15-20GB
- 总显存使用：~35-40GB（分布在两张卡上）

## 注意事项

1. **设备配置**：确保`device`和`sub_device`是不同的GPU
2. **显存监控**：日志中会显示两张卡的显存使用情况
3. **性能影响**：设备间数据传输会有轻微性能开销，但可以避免OOM
4. **兼容性**：如果`sub_device`未配置或与`device`相同，代码会自动回退到单卡模式

## 故障排查

### 如果仍然OOM

1. **减小batch_size**：在配置文件中将`batch_size`从1改为更小的值（如果可能）
2. **检查设备**：确保`device`和`sub_device`是不同的GPU
3. **查看显存日志**：检查`memory/alloc_max`和`memory/alloc_max_sub`的值
4. **使用gradient checkpointing**：可以进一步减少显存占用（需要额外实现）

### 常见错误

- **设备不匹配**：确保sub_device是有效的GPU编号
- **显存不足**：即使使用双卡，如果单卡显存不足也会OOM，考虑使用更小的模型或batch size

## 进一步优化建议

1. **使用gradient checkpointing**：减少激活值占用
2. **减少前向传播次数**：合并某些计算
3. **使用8-bit量化**：进一步减少模型显存占用
4. **模型分片**：使用`device_map='auto'`自动分片（需要修改模型加载逻辑）

