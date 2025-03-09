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
                 device="cuda",
                 dtype=torch.bfloat16):
        super().__init__()
        # 初始化基础模型
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
        self.projection = nn.Linear(512, 768).to(device=device, dtype=dtype)
        
        # 修正LoRA应用方式
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self._get_target_modules(lora_trainable_modules),  # 新增模块选择方法
            lora_dropout=lora_dropout,
            modules_to_save=["projection"]  # 确保投影层可训练
        )
        
        # 重新组织模型结构
        self.peft_model = get_peft_model(
            self.clip_model.visual,  # 直接应用视觉模型
            self.lora_config
        )
        self.peft_model.to(device=device, dtype=dtype)

        # 分离投影层（保持独立训练）
        self.projection = nn.Linear(512, 768).to(device=device, dtype=dtype)

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
            print("clip training")
            processed_images = batch_preprocess(x)
            # print(processed_images.shape)
            visual_features = self.peft_model(processed_images)
        else:
            print("clip eval")
            with torch.no_grad():
                processed_images = batch_preprocess(x)
                visual_features = self.peft_model(processed_images)

        # print(visual_features.shape)
            
        # 应用LoRA和投影
        visual_features = self.projection(visual_features)
        
        # 标准化输出（参考CLIP官方实现）
        return visual_features / visual_features.norm(dim=-1, keepdim=True)
        
    def encode_image(self, x):
        return self(x)  # L2归一化[1](@ref)

    def train_mode(self, enable_projection=True):
        """增强训练模式控制"""
        self.peft_model.train()
        if enable_projection:
            self.projection.train()
        else:
            self.projection.eval()
        
    def eval_mode(self):
        """统一评估模式"""
        self.peft_model.eval()
        self.projection.eval()
        
    def save_pretrained(self, save_path):
        """保存完整模型"""
        torch.save({
            'clip_state_dict': self.clip_model.state_dict(),
            'projection_state_dict': self.projection.state_dict(),
            'lora_config': self.lora_config,
            'peft_state_dict': self.peft_model.state_dict()
        }, save_path)
        
    @classmethod
    def from_pretrained(cls, load_path, **kwargs):
        """加载预训练模型"""
        checkpoint = torch.load(load_path, map_location='cpu')
        instance = cls(**kwargs)
        instance.clip_model.load_state_dict(checkpoint['clip_state_dict'])
        instance.projection.load_state_dict(checkpoint['projection_state_dict'])
        instance.peft_model.load_state_dict(checkpoint['peft_state_dict'])
        return instance.to(checkpoint['device'])
    
    def _get_target_modules(self, module_names):
        """动态解析目标模块"""
        valid_modules = []
        for name, module in self.clip_model.visual.named_modules():
            if "in_proj" in name or "out_proj" in name:
                valid_modules.append(name)
        return valid_modules

    def create_batch_transform(self, is_train, image_mean, image_std):
        return BatchImageTransform(
            image_size=self.clip_model.visual.image_size,
            mean=image_mean or self.clip_model.visual.image_mean,
            std=image_std or self.clip_model.visual.image_std,
            is_train=is_train
        )

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