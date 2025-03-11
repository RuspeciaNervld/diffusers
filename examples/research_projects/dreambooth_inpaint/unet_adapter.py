import torch
import torch.nn as nn
from accelerate import load_checkpoint_in_model
import torch.nn.functional as F

class SkipAttnProcessor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # 这里的梯度不更新
        # self.requires_grad_(False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        return hidden_states
    
class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        **kwargs
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def init_adapter(unet, 
                cross_attn_cls=SkipAttnProcessor,
                self_attn_cls=None,
                cross_attn_dim=None, 
                **kwargs):
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
            else:
                # retain the original attn processor
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
        else:
            if not name.startswith("up_blocks"):
                attn_procs[name] = cross_attn_cls(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)
            else:
                attn_procs[name] = AttnProcessor2_0(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, **kwargs)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules

# ! 加载catvton的attention权重
def get_trainable_module(unet, trainable_module_name):
    if trainable_module_name == "unet":
        return unet
    elif trainable_module_name == "transformer":
        trainable_modules = torch.nn.ModuleList()
        for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
            if hasattr(blocks, "attentions"):
                trainable_modules.append(blocks.attentions)
            else:
                for block in blocks:
                    if hasattr(block, "attentions"):
                        trainable_modules.append(block.attentions)
        return trainable_modules
    elif trainable_module_name == "attention":
        attn_blocks = torch.nn.ModuleList()
        for name, param in unet.named_modules():
            if "attn1" in name:
                attn_blocks.append(param)
        for name, param in unet.named_modules():
            if name.startswith("up_blocks") and "attn2" in name:
                print("加入一个上采样交叉注意力")
                attn_blocks.append(param)
        return attn_blocks
    else:
        raise ValueError(f"Unknown trainable_module_name: {trainable_module_name}")



def adapt_unet_with_catvton_attn(unet, catvton_attn_path=None, trainable_modules="attention") -> torch.nn.ModuleList:
    init_adapter(unet, cross_attn_cls=SkipAttnProcessor)  

    attn_modules = torch.nn.ModuleList()
    for module_name in trainable_modules.split(";"):
        module = get_trainable_module(unet, module_name.strip())
        if isinstance(module, torch.nn.ModuleList):
            attn_modules.extend(module)
        else:
            attn_modules.append(module)

    if catvton_attn_path:
        # 打印attn_modules中每个模块的名称
        # for i, module in enumerate(attn_modules):
            # print(f"Module {i+1}: {module}")
        # 打印catvton_attn_path里面的model.safetensors中每个模块的名
        # import safetensors.torch
        # state_dict = safetensors.torch.load_file(args.catvton_attn_path + "/model.safetensors")
        # for key in state_dict.keys():
        #     print(f"{key}")
        load_checkpoint_in_model(attn_modules, catvton_attn_path)
        print("成功加载catvton的attention权重")
    else:
        print("没有提供catvton的attention权重，继续使用默认的attention权重")

    return attn_modules

    #! 到此成功加载catvton的attention权重，但是他把交叉注意力层关闭了


# def adapt_unet_with_catvton_and_my_attn(
#     unet, 
#     catvton_path=None, 
#     my_weights_path=None,
#     trainable_modules="attention",
#     condition_dim=512
# ):
#     """双权重加载适配函数"""
#     # 步骤1：初始化双处理器结构
#     init_adapter_v2(unet)
    
#     # 模块分离
#     catvton_modules = []
#     my_modules = []
    
#     # 分离自注意力和交叉注意力模块
#     for name, module in unet.named_modules():
#         if "attn1" in name and any([n in name for n in ["mid_block", "up_blocks"]]):
#             catvton_modules.append(module)
#         elif "attn2" in name and any([n in name for n in ["mid_block", "up_blocks"]]):
#             my_modules.append(module)
    
#     # 步骤2：加载CatVton权重到自注意力层
#     if catvton_path:
#         # 创建权重映射表
#         catvton_state = torch.load(catvton_path)
#         mapped_state = {}
#         for k, v in catvton_state.items():
#             new_key = k.replace("attn_layers", "attn1.processor.self_attn_proj")
#             mapped_state[new_key] = v
#         for mod in catvton_modules:
#             mod.load_state_dict(mapped_state, strict=False)
    
#     # 步骤3：加载自定义权重到交叉注意力层
#     if my_weights_path:
#         my_state = torch.load(my_weights_path)
#         for mod in my_modules:
#             # 调整键名匹配
#             mod.cross_attn_proj.load_state_dict({
#                 k.replace("cross_attn.", ""): v 
#                 for k, v in my_state.items()
#             }, strict=True)
    
#     # 配置可训练参数
#     trainable_params = []
#     if "self_attn" in trainable_modules:
#         trainable_params += [p for mod in catvton_modules for p in mod.parameters()]
#     if "cross_attn" in trainable_modules:
#         trainable_params += [p for mod in my_modules for p in mod.parameters()]
    
#     return torch.nn.ModuleList(trainable_params)