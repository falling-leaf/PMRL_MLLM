
import argparse
import torch
from easyeditor import MENDMultimodalTrainingHparams, WISEMultimodalHyperParams, MENDMultimodalHparams
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MultimodalEditor, MultimodalTrainer
from examples.run_adsedit import get_data
# 读取数据集的索引json
caption_train_path = '/data/jjsu/easyedit/MMEdit/editing-data/caption/caption_train_edit.json'
caption_eval_path = '/data/jjsu/easyedit/MMEdit/editing-data/caption/caption_eval_edit.json'
vqa_train_path = '/data/jjsu/easyedit/MMEdit/editing-data/vqa/vqa_train.json'
vqa_eval_path = '/data/jjsu/easyedit/MMEdit/editing-data/vqa/vqa_eval.json'

# 模型和数据路径
model_path = '/model/jjsu/Model/'
data_path = '/data/jjsu/easyedit/MMEdit/'

def apply_mend_method(args):
    # 加载训练配置
    if args.model == 'blip2':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2.yaml')
        training_hparams.name = model_path + 'opt-2.7b'
        training_hparams.tokenizer_name = model_path + 'opt-2.7b'
        training_hparams.qformer_checkpoint = model_path + 'blip2_pretrained_opt2.7b.pth'
        training_hparams.state_dict_file = model_path + 'eva_vit_g.pth'
    elif args.model == 'qwen':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/qwen2vl-7b.yaml')
    elif args.model == 'minigpt4':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4.yaml')
        training_hparams.name = model_path + 'Vicuna'
        training_hparams.tokenizer_name = model_path + 'Vicuna'
        training_hparams.qformer_checkpoint = model_path + 'blip2_pretrained_flant5xxl.pth'
        training_hparams.state_dict_file = model_path + 'eva_vit_g.pth'
        training_hparams.pretrained_ckpt = model_path + 'pretrained_minigpt4_7b.pth'
    else:
        raise ValueError(f"Unknown model configuration: {args.model}")
    training_hparams.coco_image = data_path
    training_hparams.rephrase_image = data_path
    # 设置设备
    training_hparams.device = "cuda:" + args.device
    training_hparams.sub_device = "cuda:" + args.sub_device
    
    # 根据数据集类型选择相应的数据集类
    if args.ds == 'caption':
        train_ds = CaptionDataset(caption_train_path, config=training_hparams)
        eval_ds = CaptionDataset(caption_eval_path, config=training_hparams)
    elif args.ds == 'vqa':
        train_ds = VQADataset(vqa_train_path, config=training_hparams)
        eval_ds = VQADataset(vqa_eval_path, config=training_hparams)
    else:
        raise ValueError(f"Unknown dataset type: {args.ds}")
    
    # 创建训练器并运行
    trainer = MultimodalTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

def apply_wise_method(args):
    # 加载训练配置
    if args.model == 'blip2':
        raise ValueError(f"Unknown model configuration: {args.model}")
    elif args.model == 'minigpt4':
        raise ValueError(f"Unknown model configuration: {args.model}")
    elif args.model == 'qwen':
        raise ValueError(f"Unknown model configuration: {args.model}")
    elif args.model == 'llava':
        hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/llavaov-7b.yaml')
        hparams.model_name = model_path + "llava-onevision-qwen2-7b-ov-hf"
        hparams.dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown model configuration: {args.model}")
    hparams.coco_image = data_path
    hparams.rephrase_image = data_path
    # 设置设备
    hparams.device = int(args.device)
    hparams.sub_device = int(args.sub_device)

    if args.ds == 'caption':
        train_ds = CaptionDataset(caption_train_path, config=hparams)
    elif args.ds == 'vqa':
        train_ds = VQADataset(vqa_train_path, config=hparams)
    else:
        raise ValueError(f"Unknown dataset type: {args.ds}")
    
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        train_ds,
        keep_original_weight=False,
        verbose=True
    )
    acc = 0
    gen = 0
    t_loc = 0
    i_loc = 0
    for case in metrics:
        acc += case["post"]["rewrite_acc"].item()
        gen += case["post"]["image_rephrase_acc"].item()
        t_loc += case["post"]["locality_acc"].item()
        i_loc += case["post"]["multimodal_locality_acc"].item()
    print("-------------------- Final Results -------------------")
    print(f"Rewrite Acc: {acc/len(metrics)}, Rephrase Acc: {gen/len(metrics)}, Text Loc Acc: {t_loc/len(metrics)}, Image Loc Acc: {i_loc/len(metrics)}")


def main():
    # python3 start_code.py --device 7 --sub_device 7 --method mend --model blip2 --ds caption
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='MEND Multimodal Training Script')
    
    # 添加设备参数
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for training (e.g., cuda:0, cpu)')

    parser.add_argument('--sub_device', type=str, default='0',
                       help='Sub-device to use for training (e.g., cuda:0, cpu)')

    parser.add_argument('--method', type=str, default='mend',
                       help='Editing method to use (default: mend)')
    
    # 添加配置文件路径参数
    parser.add_argument('--model', type=str, default='./hparams/TRAINING/MEND/blip2.yaml',
                       help='Path to training configuration YAML file')
    
    # 添加数据集类型参数
    parser.add_argument('--ds', type=str, default='caption', choices=['caption', 'vqa'],
                       help='Type of dataset to use: caption or vqa')
    
    # 解析参数
    args = parser.parse_args()

    if args.method == 'mend':
        apply_mend_method(args)
    elif args.method == 'wise':
        apply_wise_method(args)
    else:
        raise ValueError(f"Unknown editing method: {args.method}")

if __name__ == "__main__":
    main()