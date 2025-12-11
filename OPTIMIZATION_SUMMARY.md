# 显存优化总结

## 问题分析

执行 `python3 start_code.py --device 3 --sub_device 3 --method mend --model qwen --ds vqa` 时出现CUDA OOM错误。

### 显存占用最大的部分（按大小排序）

1. **edited_model（峰值约20GB）** ⭐ **最大瓶颈**
   - 使用higher库的monkeypatch创建functional模型
   - 保留完整计算图，显存占用约为原模型的1.5-2倍
   - 位置：`MEND.edit()` -> `monkeypatch()`

2. **模型本身（约14GB）**
   - qwen2-vl-7b在bfloat16下约14GB
   - 位置：`BaseTrainer.__init__` -> `get_model()`

3. **多次前向传播的激活值（约2-5GB）**
   - 7次前向传播产生的激活值
   - 位置：`MultimodalTrainer.edit_step()`

4. **梯度计算（约2-3GB）**
   - loss.backward()需要保存中间激活的梯度
   - 位置：`MEND.edit()`

## 优化方案

### 核心策略：双卡部署

将edited_model的所有前向传播移到`sub_device`（第二张GPU），从而：
- 主卡（device）只保留模型本身和MEND网络
- 副卡（sub_device）处理edited_model的计算
- 显存峰值从单卡的~40GB降低到双卡的~20GB+~20GB

### 实现细节

#### 1. MEND.py 修改
- 添加`sub_device`配置解析
- 在`__init__`中处理sub_device参数

#### 2. MultimodalTrainer.py 修改
- 检测sub_device配置
- 实现`move_batch_to_device()`辅助函数
- 将edited_model的4次前向传播移到sub_device：
  - `post_edit_outputs = edited_model(batch["edit_outer"])`
  - `inner_edit_outputs = edited_model(batch["edit_inner"])`
  - `post_image_edit_outputs = edited_model(batch["edit_outer_image"])`
  - `post_base_outputs = edited_model(batch["loc"])`
  - `post_image_base_outputs = edited_model(batch["loc_image"])`
- 输出logits移回主卡用于loss计算
- 添加显存监控和清理

## 使用方法

### 基本命令

```bash
# 使用device 3作为主卡，device 4作为副卡
python3 start_code.py --device 3 --sub_device 4 --method mend --model qwen --ds vqa
```

### 重要提示

⚠️ **注意**：`--device`和`--sub_device`必须是**不同的GPU编号**！

- ✅ 正确：`--device 3 --sub_device 4`
- ❌ 错误：`--device 3 --sub_device 3`（不会启用双卡优化）

### 显存分配

- **主卡（device）**：
  - 模型参数：~14GB
  - MEND网络：~少量
  - 优化器状态：~少量
  - 基础前向传播激活：~2GB
  - **总计：~20GB**

- **副卡（sub_device）**：
  - edited_model计算图：~15-20GB
  - edited_model前向传播激活：~2-3GB
  - **总计：~20GB**

## 预期效果

### 优化前（单卡）
```
主卡显存：~40GB（峰值）
结果：CUDA OOM ❌
```

### 优化后（双卡）
```
主卡显存：~20GB（峰值）
副卡显存：~20GB（峰值）
结果：正常运行 ✅
```

## 代码修改文件

1. `easyeditor/trainer/algs/MEND.py`
   - 添加sub_device配置处理

2. `easyeditor/trainer/MultimodalTrainer.py`
   - 实现双卡部署逻辑
   - 添加显存监控和清理

## 进一步优化建议

如果仍然遇到显存问题，可以考虑：

1. **减小batch_size**：在配置文件中将`batch_size`从1改为更小的值
2. **使用gradient checkpointing**：减少激活值占用（需要额外实现）
3. **使用8-bit量化**：进一步减少模型显存占用
4. **模型分片**：使用`device_map='auto'`自动分片（需要修改模型加载逻辑）

## 验证方法

运行后检查日志中的显存使用情况：
- `memory/alloc_max`：主卡峰值显存
- `memory/alloc_max_sub`：副卡峰值显存（如果启用双卡）

如果两个值都存在且合理，说明双卡部署成功。

