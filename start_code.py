
import argparse
import os
import torch
import csv
from easyeditor import MENDMultimodalTrainingHparams, WISEMultimodalHyperParams, MENDMultimodalHparams
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MultimodalEditor, MultimodalTrainer
from examples.run_adsedit import get_data
# 各种模型配置速记
# modelscope download --model Qwen/Qwen2-VL-7B --local_dir ./qwen2-vl-7b
# 
# # 读取数据集的索引json
# caption_train_path = '/data/jjsu/easyedit/MMEdit/editing-data/caption/caption_train_edit.json'
# caption_eval_path = '/data/jjsu/easyedit/MMEdit/editing-data/caption/caption_eval_edit.json'
# vqa_train_path = '/data/jjsu/easyedit/MMEdit/editing-data/vqa/vqa_train.json'
# vqa_eval_path = '/data/jjsu/easyedit/MMEdit/editing-data/vqa/vqa_eval.json'

# # 模型和数据路径
# model_path = '/model/jjsu/Model/'
# data_path = '/data/jjsu/easyedit/MMEdit/'
# result_path = '/data/jjsu/easyedit/MMEdit/results/'

# 读取数据集的索引json
caption_train_path = '/root/autodl-tmp/editing-data/caption/caption_train_edit.json'
caption_eval_path = '/root/autodl-tmp/editing-data/caption/caption_eval_edit.json'
vqa_train_path = '/root/autodl-tmp/editing-data/vqa/vqa_train.json'
vqa_eval_path = '/root/autodl-tmp/editing-data/vqa/vqa_eval.json'

# 模型和数据路径
model_path = '/root/autodl-tmp/model/'
data_path = '/root/autodl-tmp/MMEdit_images/'
result_path = '/root/autodl-tmp/results/'

# python3 start_code.py --device 0 --sub_device 0 --method wise --model blip2 --ds caption --pmrl_tau_alignment 8 --pmrl_scale 25 --num_rephrase 3
# easyedit python3 start_code.py --device 0 --sub_device 0 --method wise --model blip2 --ds caption
# easyedit_1 python3 start_code.py --device 6 --sub_device 6 --method vqa --model blip2 --ds caption
# easyedit_2 python3 start_code.py --device 0 --sub_device 0 --method mend --model blip2 --ds vqa
# easyedit_3 python3 start_code.py --device 0 --sub_device 0 --method wise --model minigpt4 --ds caption
# easyedit_4 python3 start_code.py --device 6 --sub_device 6 --method wise --model minigpt4 --ds vqa
# easyedit_5 python3 start_code.py --device 7 --sub_device 7 --method wise --model qwen --ds caption
# python3 start_code.py --device 7 --sub_device 7 --method mend --model llava --ds caption(显存问题)
def apply_mend_method(args):
    # 加载训练配置
    if args.model == 'blip2':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/blip2.yaml')
        training_hparams.name = model_path + 'opt-2.7b'
        training_hparams.tokenizer_name = model_path + 'opt-2.7b'
        training_hparams.qformer_checkpoint = model_path + 'blip2_pretrained_opt2.7b.pth'
        training_hparams.state_dict_file = model_path + 'eva_vit_g.pth'
        training_hparams.qformer_name_or_path = model_path + 'bert-base-uncased'
    elif args.model == 'qwen':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/qwen2vl-7b.yaml')
        training_hparams.name = model_path + 'qwen2-vl-7b'
        training_hparams.dtype = torch.bfloat16
    elif args.model == 'llava':
        training_hparams = MENDMultimodalTrainingHparams.from_hparams('./hparams/TRAINING/MEND/llavaov-7b.yaml')
        training_hparams.name = model_path + 'llava-onevision-qwen2-7b-ov-hf'
        training_hparams.dtype = torch.bfloat16
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
    training_hparams.results_dir = result_path
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
        hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/blip2.yaml')
        hparams.name = model_path + 'opt-2.7b'
        hparams.tokenizer_name = model_path + "opt-2.7b"
        hparams.qformer_checkpoint = model_path + 'blip2_pretrained_opt2.7b.pth'
        hparams.state_dict_file = model_path + 'eva_vit_g.pth'
        hparams.qformer_name_or_path = model_path + 'bert-base-uncased'
    elif args.model == 'minigpt4':
        hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/minigpt4.yaml')
        hparams.name = model_path + 'Vicuna'
        hparams.tokenizer_name = model_path + 'Vicuna'
        hparams.qformer_checkpoint = model_path + 'blip2_pretrained_flant5xxl.pth'
        hparams.state_dict_file = model_path + 'eva_vit_g.pth'
        hparams.pretrained_ckpt = model_path + 'pretrained_minigpt4_7b.pth'
    elif args.model == 'qwen':
        hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/qwen2vl-7b.yaml')
        hparams.model_name = model_path + "qwen2-vl-7b"
        hparams.name = model_path + "qwen2-vl-7b"
        hparams.dtype = torch.bfloat16
    elif args.model == 'llava':
        hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/llavaov-7b.yaml')
        hparams.model_name = model_path + "llava-onevision-qwen2-7b-ov-hf"
        hparams.name = model_path + "llava-onevision-qwen2-7b-ov-hf"
        hparams.dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown model configuration: {args.model}")
    hparams.coco_image = data_path
    hparams.rephrase_image = data_path
    # 设置设备
    hparams.device = int(args.device)
    hparams.sub_device = int(args.sub_device)
    if args.using_extra:
        hparams.using_extra = True
        hparams.using_dropout = False
        hparams.using_LAP = True
    else:
        hparams.using_extra = False
        hparams.using_dropout = False
        hparams.using_LAP = False
    hparams.pmrl_tau_alignment = args.pmrl_tau_alignment
    hparams.pmrl_tau_regularization = 0.1
    hparams.pmrl_scale = args.pmrl_scale
    hparams.num_rephrase = args.num_rephrase
    hparams.using_imageembedding = args.using_imageembedding

    # hparams.sequential_edit = True



    if args.ds == 'caption':
        train_ds = CaptionDataset(caption_train_path, config=hparams, size=100)
    elif args.ds == 'vqa':
        # train_ds = VQADataset(vqa_train_path, config=hparams, size=100)
        train_ds = VQADataset(vqa_train_path, config=hparams, size=50)
    else:
        raise ValueError(f"Unknown dataset type: {args.ds}")
    
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        train_ds,
        keep_original_weight=False,
        verbose=True
    )
    acc = 0
    t_gen = 0
    gen = 0
    t_loc = 0
    i_loc = 0
    # List to store gen values for each case
    gen_values_list = []
    for case in metrics:
        acc += case["post"]["rewrite_acc"].item()
        t_gen += case["post"]["rephrase_acc"].item()
        case_gen_value = case["post"]["image_rephrase_acc"].item()
        gen += case_gen_value
        gen_values_list.append(case_gen_value)  # Store individual gen value
        t_loc += case["post"]["locality_acc"].item()
        i_loc += case["post"]["multimodal_locality_acc"].item()
    
    # Record gen values of each case to a txt file
    gen_values_filename = f"gen_values_{args.method}_{args.model}_{args.ds}.txt"
    gen_values_filepath = os.path.join(result_path, gen_values_filename)
    os.makedirs(result_path, exist_ok=True)
    with open(gen_values_filepath, 'w') as f:
        for idx, gen_val in enumerate(gen_values_list):
            f.write(f"Case {idx}: {gen_val}\n")
    
    rewrite_acc = acc/len(metrics) * 100
    t_gen_acc = t_gen/len(metrics) * 100
    rephrase_acc = gen/len(metrics) * 100
    text_loc_acc = t_loc/len(metrics) * 100
    image_loc_acc = i_loc/len(metrics) * 100
    
    # Print original console output
    print("-------------------- Final Results -------------------")
    print(f"Rewrite Acc: {rewrite_acc}, Text Gen Acc: {t_gen_acc}, Rephrase Acc: {rephrase_acc}, Text Loc Acc: {text_loc_acc}, Image Loc Acc: {image_loc_acc}")
    
    # Create single CSV log file path
    csv_filename = "results.csv"
    csv_filepath = os.path.join(result_path, csv_filename)
    
    # Write results to single CSV file in result_path
    os.makedirs(result_path, exist_ok=True)  # Ensure directory exists
    file_exists = os.path.exists(csv_filepath)
    with open(csv_filepath, 'a', newline='') as csvfile:  # Using append mode to add to existing file
        fieldnames = ['method', 'model', 'dataset', 'acc','t-gen', 'i-gen', 't-loc', 'i-loc', 'PMRL_scale', 'PMRL_tau', 'num_rephrase', 'using_imageembedding']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file doesn't exist yet
        if not file_exists:
            writer.writeheader()
        
        # Write the results row
        writer.writerow({
            'method': args.method,
            'model': args.model,
            'dataset': args.ds,
            'acc': rewrite_acc,
            't-gen': t_gen_acc,
            'i-gen': rephrase_acc,
            't-loc': text_loc_acc,
            'i-loc': image_loc_acc,
            'PMRL_scale': args.pmrl_scale,
            'PMRL_tau': args.pmrl_tau_alignment,
            'num_rephrase': args.num_rephrase,
            'using_imageembedding': args.using_imageembedding
        })


def main():
    # python3 start_code.py --device 4 --sub_device 4 --method mend --model blip2 --ds caption
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
    
    # 添加pmrl_tau_alignment和pmrl_scale参数
    parser.add_argument('--pmrl_tau_alignment', type=float, default=0.5,
                       help='PMRL tau alignment parameter (default: 0.5)')
    
    parser.add_argument('--pmrl_scale', type=float, default=0.5,
                       help='PMRL scale parameter (default: 0.5)')

    parser.add_argument('--num_rephrase', type=int, default=2,
                       help='Number of rephrases to use (default: 6)')

    parser.add_argument('--using_imageembedding', action='store_true',
                       help='Use image embedding (default: False)')

    parser.add_argument('--using_extra', action='store_true',
                       help='Use extra (default: False)')
    
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