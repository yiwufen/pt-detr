import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers.models.detr.modeling_detr import DetrEncoderLayer, DetrDecoderLayer
def ball_detr():
    """
    加载在气球数据集上微调的DETR模型和处理器。

    该函数加载一个预训练的DETR（Detection Transformer）模型及其对应的图像处理器，
    两者都专门针对检测图像中的气球进行了微调。

    返回:
        model: 在气球数据集上微调的DETR模型。
        processor: 在气球数据集上微调的DETR模型的图像处理器。
    """
    model = DetrForObjectDetection.from_pretrained("yiyiyiwufeng/detr-finetuned-balloon-v2", id2label={0:"balloon"})
    processor = DetrImageProcessor.from_pretrained("yiyiyiwufeng/detr-finetuned-balloon-v2")
    return model, processor

import torch.nn as nn

class PromptTunedEncoderLayer(DetrEncoderLayer):
    def __init__(self, config):
        super().__init__(config)

        # 初始化提示词嵌入为可学习参数
        prompt_length = 10  # 根据需要设置提示词的长度
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_length, config.d_model))
        # self.prompt_object_queries = nn.Parameter(torch.randn(1, prompt_length, config.d_model))  # 新增提示词对象查询

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        # 扩展提示词嵌入的批次维度
        batch_size = hidden_states.size(0)
        prompt_embeddings = self.prompt_embeddings.expand(batch_size, -1, -1)

        # 连接提示词嵌入和隐藏状态
        hidden_states = torch.cat([prompt_embeddings, hidden_states], dim=1)

        # 更新 object_queries
        prompt_object_queries = torch.zeros_like(self.prompt_embeddings)
        prompt_object_queries = prompt_object_queries.expand(batch_size, -1, -1)
        if object_queries is not None:
            object_queries = torch.cat([prompt_object_queries, object_queries], dim=1)  # [batch_size, 960, d_model]
        else:
            object_queries = prompt_object_queries  # [batch_size, 10, d_model]


        # 更新注意力掩码
        if attention_mask is not None:
            prompt_len = self.prompt_embeddings.size(1)

            # 创建一个新的注意力掩码，大小为 [batch_size, 1, seq_len + prompt_len, seq_len + prompt_len]
            new_seq_len = hidden_states.size(1)
            new_attention_mask = torch.zeros(
                (batch_size, 1, new_seq_len, new_seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            
            # 将原始的注意力掩码复制到新的位置
            new_attention_mask[:, :, prompt_len:, prompt_len:] = attention_mask
            
            # 如果需要，可以在这里添加其他规则来设置提示词部分的掩码
            # 例如，假设提示词之间不需要遮蔽自己或其它提示词，保持默认的零值
            
            attention_mask = new_attention_mask

        # 调用父类的 forward 方法
        extended_output = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )
        # 移除提示词嵌入部分
        if isinstance(extended_output, tuple):
            hidden_state = extended_output[0][:, prompt_len:]
            other_outputs = extended_output[1:]
            output = (hidden_state,) + other_outputs
        else:
            output = extended_output[:, prompt_len:]
        
        return output

from typing import Optional

class PromptTunedDecoderLayer(DetrDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        # 初始化提示词嵌入为可学习参数
        prompt_length = 10  # 根据需要设置提示词的长度
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_length, config.d_model))
        # self.prompt_object_queries = nn.Parameter(torch.randn(1, prompt_length, config.d_model))  # 新增提示词对象查询
        # self.prompt_position_embeddings = nn.Parameter(torch.randn(1, prompt_length, config.d_model))
        # self.prompt_encoder_hidden = nn.Parameter(torch.randn(1, prompt_length, config.d_model))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 扩展提示词嵌入的批次维度
        batch_size = hidden_states.size(0)
        prompt_embeddings = self.prompt_embeddings.expand(batch_size, -1, -1)

        # 连接提示词嵌入和隐藏状态
        hidden_states = torch.cat([prompt_embeddings, hidden_states], dim=1)

        # 更新 object_queries
        prompt_object_queries = torch.zeros_like(self.prompt_embeddings)
        prompt_object_queries = prompt_object_queries.expand(batch_size, -1, -1)
        if object_queries is not None:
            object_queries = torch.cat([prompt_object_queries, object_queries], dim=1)  # [batch_size, 960, d_model]
        else:
            object_queries = prompt_object_queries  # [batch_size, 10, d_model]

        # 更新注意力掩码
        if attention_mask is not None:
            prompt_len = self.prompt_embeddings.size(1)
            # 创建一个新的注意力掩码，大小为 [batch_size, 1, seq_len + prompt_len, seq_len + prompt_len]
            new_seq_len = hidden_states.size(1)
            new_attention_mask = torch.zeros(
                (batch_size, 1, new_seq_len, new_seq_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            
            # 将原始的注意力掩码复制到新的位置
            new_attention_mask[:, :, prompt_len:, prompt_len:] = attention_mask
            
            # 如果需要，可以在这里添加其他规则来设置提示词部分的掩码
            # 例如，假设提示词之间不需要遮蔽自己或其它提示词，保持默认的零值
            
            attention_mask = new_attention_mask
        
        # 更新 encoder_attention_mask
        if encoder_attention_mask is not None:
            prompt_len = self.prompt_embeddings.size(1)  # 通常为10
            batch_size = encoder_hidden_states.size(0)
            orig_query_seq_len = encoder_attention_mask.size(-2)  # 原始查询序列长度，例如100
            orig_key_seq_len = encoder_attention_mask.size(-1)    # 原始键序列长度，例如950
            new_query_seq_len = orig_query_seq_len + prompt_len  # 新查询序列长度，例如110
            new_key_seq_len = orig_key_seq_len + prompt_len      # 新键序列长度，例如960

            # 创建新的注意力掩码，初始化为零（不遮蔽）
            new_attention_mask = torch.zeros(
                (batch_size, 1, new_query_seq_len, new_key_seq_len),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device
            )

            # 将原始的注意力掩码复制到新的位置
            new_attention_mask[:, :, prompt_len:, prompt_len:] = encoder_attention_mask

            # 设置提示词之间的注意力掩码规则
            # 例如，提示词之间不需要遮蔽，保持为0
            new_attention_mask[:, :, :prompt_len, :prompt_len] = 0

            # 如果提示词需要与原始序列交互，可以根据需求设置
            # 例如，允许提示词关注原始序列
            new_attention_mask[:, :, :prompt_len, prompt_len:] = 0  # 提示词关注原始序列
            new_attention_mask[:, :, prompt_len:, :prompt_len] = 0  # 原始序列关注提示词

            encoder_attention_mask = new_attention_mask

        # 处理 query_position_embeddings
        if query_position_embeddings is not None:
            prompt_position_embeddings = torch.zeros_like(self.prompt_embeddings)
            prompt_position_embeddings = prompt_position_embeddings.expand(batch_size, -1, -1)
            query_position_embeddings = torch.cat([prompt_position_embeddings, query_position_embeddings], dim=1)

        # 处理 encoder_hidden_states
        if encoder_hidden_states is not None:
            prompt_encoder_hidden = torch.zeros_like(self.prompt_embeddings)
            prompt_encoder_hidden = prompt_encoder_hidden.expand(batch_size, -1, -1)
            encoder_hidden_states = torch.cat([prompt_encoder_hidden, encoder_hidden_states], dim=1)

        # 调用父类的 forward 方法，保持输入输出一致
        extended_output = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        # 移除提示词嵌入部分
        if isinstance(extended_output, tuple):
            hidden_state = extended_output[0][:, prompt_len:]
            other_outputs = extended_output[1:]
            output = (hidden_state,) + other_outputs
        else:
            output = extended_output[:, prompt_len:]
        
        return output

def pt_model():
    model = DetrForObjectDetection.from_pretrained("yiyiyiwufeng/detr-finetuned-balloon-v2", id2label={0:"balloon"})

    # 替换编码器层并复制参数
    for i in range(len(model.model.encoder.layers)):
        old_layer = model.model.encoder.layers[i]
        new_layer = PromptTunedEncoderLayer(model.config)
        old_state_dict = old_layer.state_dict()
        new_layer.load_state_dict(old_state_dict, strict=False)
        
        # 验证键的一致性
        new_state_dict = new_layer.state_dict()
        missing_keys = set(old_state_dict.keys()) - set(new_state_dict.keys())
        unexpected_keys = set(new_state_dict.keys()) - set(old_state_dict.keys())
        
        if missing_keys:
            print(f"Encoder Layer {i} 缺少键: {missing_keys}")
        if unexpected_keys:
            print(f"Encoder Layer {i} 存在未预期的键: {unexpected_keys}")
        
        model.model.encoder.layers[i] = new_layer

    # 替换解码器层并复制参数
    for i in range(len(model.model.decoder.layers)):
        old_layer = model.model.decoder.layers[i]
        new_layer = PromptTunedDecoderLayer(model.config)
        old_state_dict = old_layer.state_dict()
        new_layer.load_state_dict(old_state_dict, strict=False)
        
        # 验证键的一致性
        new_state_dict = new_layer.state_dict()
        missing_keys = set(old_state_dict.keys()) - set(new_state_dict.keys())
        unexpected_keys = set(new_state_dict.keys()) - set(old_state_dict.keys())
        
        # if missing_keys:
        #     print(f"Decoder Layer {i} 缺少键: {missing_keys}")
        # if unexpected_keys:
        #     print(f"Decoder Layer {i} 存在未预期的键: {unexpected_keys}")
        
        model.model.decoder.layers[i] = new_layer
    
    return model

def freeze_model(model):
    # 冻结模型中的所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻编码器层中的所有提示词嵌入相关参数
    for layer in model.model.encoder.layers:
        if hasattr(layer, 'prompt_embeddings'):
            layer.prompt_embeddings.requires_grad = True
        if hasattr(layer, 'prompt_object_queries'):
            layer.prompt_object_queries.requires_grad = True

    # 解冻解码器层中的所有提示词嵌入相关参数
    for layer in model.model.decoder.layers:
        if hasattr(layer, 'prompt_embeddings'):
            layer.prompt_embeddings.requires_grad = True
        if hasattr(layer, 'prompt_object_queries'):
            layer.prompt_object_queries.requires_grad = True
        if hasattr(layer, 'prompt_position_embeddings'):
            layer.prompt_position_embeddings.requires_grad = True
        if hasattr(layer, 'prompt_encoder_hidden'):
            layer.prompt_encoder_hidden.requires_grad = True

    # # 查看参数的冻结状态
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")

    return model

# model = pt_model()



