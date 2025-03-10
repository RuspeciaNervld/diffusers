import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import open_clip
from PIL import Image
from torchvision import transforms

class CLIPPeftModel(nn.Module):
    def __init__(self, 
                 clip_model_name='ViT-B/32',
                 checkpoint_path=None,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 lora_trainable_modules=["all","in_proj","out_proj"],
                 lora_config=None,
                 device="cuda",
                 dtype=torch.bfloat16):
        super().__init__()
        # 初始化基础模型
        self.clip_model_name = clip_model_name
        self.clip_model, self.preprocess_train, self.preprocess_eval = open_clip.create_model_and_transforms(clip_model_name)
        self.clip_model.to(device=device, dtype=dtype)
        
        # 加载微调权重
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.clip_model.load_state_dict(state_dict['CLIP'])

        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
            
        # 添加投影层
        self.clip_model.visual.projection = nn.Linear(512, 768).to(device=device, dtype=dtype)
        
        # 修正LoRA配置
        if lora_config is None:
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_trainable_modules,  # 新增模块选择方法
                lora_dropout=lora_dropout,
                modules_to_save=["projection"]  # 确保投影层可训练
            )
        else:
            self.lora_config = lora_config
        
        
        # 生成PEFT模型时应包含完整结构
        self.peft_model = get_peft_model(
            nn.Sequential(
                self.clip_model.visual,
                self.clip_model.visual.projection  # 确保投影层被包含
            ), 
            self.lora_config
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        """主前向传播方法"""
        # 预处理（兼容不同输入维度）
        if isinstance(x, torch.Tensor):  # 处理张量输入
            if x.ndim == 3:
                x = x.unsqueeze(0)

        # print(x.shape)

        if self.training:
            batch_preprocess = self.create_batch_transform(is_train=True, image_mean=None, image_std=None)
        else:
            batch_preprocess = self.create_batch_transform(is_train=False, image_mean=None, image_std=None)


        batch_preprocess = batch_preprocess.to('cuda')

        
        # 执行CLIP视觉编码
        if self.training:
            # print("clip training")
            processed_images = batch_preprocess(x)
            # print(processed_images.shape)
            visual_features = self.peft_model(processed_images)
        else:
            # print("clip eval")
            with torch.no_grad():
                processed_images = batch_preprocess(x)
                visual_features = self.peft_model(processed_images)
        visual_features = visual_features.unsqueeze(1)  # 序列长度设为 1
        # 标准化输出（参考CLIP官方实现）
        return visual_features / visual_features.norm(dim=-1, keepdim=True)
        
    def encode_image(self, x):
        return self(x)  # L2归一化[1](@ref)

    def train_mode(self, enable_projection=True):
        """增强训练模式控制"""
        self.peft_model.train()
        
    def eval_mode(self):
        """统一评估模式"""
        self.peft_model.eval()
        
    def save_pretrained(self, save_path):
        """分离基础模型与微调组件"""
        torch.save({
            'lora_config': self.lora_config,  # 配置参数
            'peft_state_dict': self.peft_model.state_dict(),  # LoRA+投影层
            'clip_model_name': self.clip_model_name,
        }, save_path)
        
    @classmethod
    def from_pretrained(cls, 
                    base_model_path,  # 基础模型路径
                    finetune_path,    # 微调参数路径
                    device="cuda",
                    dtype=torch.bfloat16,
                    **kwargs):
        finetune_checkpoint = torch.load(finetune_path)
        
        clip_model_name = 'ViT-B/32'
        if "clip_model_name" in finetune_checkpoint:
            clip_model_name = finetune_checkpoint['clip_model_name']

        lora_config = None
        if "lora_config" in finetune_checkpoint:
            lora_config = finetune_checkpoint['lora_config']

        # 初始化微调框架
        instance = cls(
            clip_model_name=clip_model_name,
            checkpoint_path=base_model_path,
            lora_config=lora_config,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
        # 加载微调参数
        instance.peft_model.load_state_dict(finetune_checkpoint['peft_state_dict'])
        
        return instance
    

    def create_batch_transform(self, is_train, image_mean, image_std):
        return BatchImageTransform(
            image_size=self.clip_model.visual.image_size,
            mean=image_mean or self.clip_model.visual.image_mean,
            std=image_std or self.clip_model.visual.image_std,
            is_train=is_train
        )
    
    def show_trainable_params(self):
        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

from torchvision import transforms as T
class BatchImageTransform(torch.nn.Module):
    def __init__(self, image_size, mean, std, is_train):
        super().__init__()
        # 批量友好的变换组合
        self.transforms = torch.nn.Sequential(
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size) if not is_train else T.RandomResizedCrop(image_size),
            T.Normalize(mean=mean, std=std)  # 自动支持四维输入
        )

    def forward(self, x):
        return self.transforms(x)