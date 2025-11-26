
import argparse
from easyeditor import MENDMultimodalTrainingHparams, MENDMultimodalHparams, CaptionDataset, VQADataset, MultimodalTrainer

caption_train_path = '/data/jjsu/easyedit/MMEdit/editing-data/caption/caption_train_edit.json'
caption_eval_path = '/data/jjsu/easyedit/MMEdit/editing-data/caption/caption_eval_edit.json'
vqa_train_path = '/data/jjsu/easyedit/MMEdit/editing-data/vqa/vqa_train.json'
vqa_eval_path = '/data/jjsu/easyedit/MMEdit/editing-data/vqa/vqa_eval.json'
model_path = '/model/jjsu/Model/'
data_path = '/data/jjsu/easyedit/MMEdit/'

def apply_mend_method(args):
    # 加载训练配置
    if args.model == 'blip2':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2.yaml')
    elif args.model == 'qwen':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/qwen2vl-7b.yaml')
    elif args.model == 'minigpt4':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/minigpt4.yaml')
    else:
        raise ValueError(f"Unknown model configuration: {args.model}")
    training_hparams.name = model_path + 'opt-2.7b'
    training_hparams.tokenizer_name = model_path + 'opt-2.7b'
    training_hparams.qformer_checkpoint = model_path + 'blip2_pretrained_opt2.7b.pth'
    # 下面这条逻辑被写死了
    # training_hparams.qformer_name_or_path = model_path + 'bert-base-uncased'
    training_hparams.state_dict_file = model_path + 'eva_vit_g.pth'
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
    else:
        raise ValueError(f"Unknown editing method: {args.method}")

if __name__ == "__main__":
    main()