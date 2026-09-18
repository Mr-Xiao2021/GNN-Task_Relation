import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
try:
    from transformers import BitsAndBytesConfig
    from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
except ImportError:
    BitsAndBytesConfig = None
    LlamaForCausalLM = None
    LlamaTokenizer = None
    AutoTokenizer = None
    AutoModel = None

try:
    from accelerate.hooks import remove_hook_from_module
except ImportError:
    def remove_hook_from_module(module, recurse=True):
        return module

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

try:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

from gp.nn.layer.pyg import RGCNEdgeConv
from gp.nn.models.GNN import MultiLayerMessagePassing
from gp.nn.models.util_model import MLP
from gp.utils.utils import load_pretrained_state

LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120}


class TextClassModel(torch.nn.Module):
    def __init__(self, model, outdim, task_dim, emb=None):
        super().__init__()
        self.model = model
        if emb is not None:
            self.emb = torch.nn.Parameter(emb.clone())

        self.mlp = MLP([2 * outdim, 2 * outdim, outdim, task_dim])

    def forward(self, g):
        emb = self.model(g)
        class_emb = emb[g.target_node_mask]
        att_emb = class_emb.repeat_interleave(len(self.emb), dim=0)
        att_emb = torch.cat(
            [att_emb, self.emb.repeat(len(class_emb), 1)], dim=-1
        )
        res = self.mlp(att_emb).view(-1, len(self.emb))
        return res


class AdaPoolClassModel(torch.nn.Module):
    def __init__(self, model, outdim, task_dim, emb=None):
        super().__init__()
        self.model = model
        if emb is not None:
            self.emb = torch.nn.Parameter(emb.clone())

        self.mlp = MLP([2 * outdim, 2 * outdim, outdim, task_dim])

    def forward(self, g):
        emb = self.model(g)
        float_mask = g.target_node_mask.to(torch.float)
        target_emb = float_mask.view(-1, 1) * emb
        n_count = global_add_pool(float_mask, g.batch, g.num_graphs)
        class_emb = global_add_pool(target_emb, g.batch, g.num_graphs)
        class_emb = class_emb / n_count.view(-1, 1)
        rep_class_emb = class_emb.repeat_interleave(g.num_classes, dim=0)
        res = self.mlp(
            torch.cat([rep_class_emb, g.x[g.true_nodes_mask]], dim=-1)
        )
        return res


class SingleHeadAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))
        self.Wk = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wk)
        self.Wq = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class BinGraphModel(torch.nn.Module):
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)# 投影 x 和 edge_attr

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = self.model(g) # GNN，得到所有节点表示
        class_emb = emb[g.true_nodes_mask] # 每张图取num_class个类别节点
        res = self.mlp(class_emb) # 每个类别节点输出 1 个分数
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False



class BinGraphAttModel(torch.nn.Module):
    """
    GNN model that use a single layer attention to pool final node representation across
    layers.
    """
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        self.att = SingleHeadAtt(outdim)
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)
        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = torch.stack(self.model(g), dim=1) # [N, L, D]
        query = g.x.unsqueeze(1) # [N, 1, D]
        # 注意力：在同一个节点的不同 GNN 层之间计算。
        emb = self.att(emb, query, emb)[0].squeeze() # key, query, value=> [N, D]

        class_emb = emb[g.true_nodes_mask] # [C, D]
        res = self.mlp(class_emb) # [C,1]
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.att.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)


class LLMModel(torch.nn.Module):
    """
    Large language model from transformers.
    If peft is ture, use lora with pre-defined parameter setting for efficient fine-tuning.
    quantization is set to 4bit and should be used in the most of the case to avoid OOM.
    """
    def __init__(
        self,
        llm_name,
        quantization=True,
        peft=True,
        cache_dir="cache_data/model",
        max_length=500,
        adapter_path=None,
        peft_trainable=True,
    ):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        if AutoModel is None:
            raise ModuleNotFoundError(
                "Text encoding requires transformers. Install transformers in the active environment."
            )
        if quantization and bnb is None:
            raise ModuleNotFoundError(
                "LLM quantization requires bitsandbytes. Install it or set llm_quantization=False."
            )
        if peft and get_peft_model is None:
            raise ModuleNotFoundError(
                "LLM fine-tuning requires peft. Install it or set llm_peft=False."
            )
        if adapter_path is not None and not peft:
            raise ValueError("Loading a PEFT adapter requires llm_peft=True.")
        self.llm_name = llm_name
        self.quantization = quantization
        self.peft = peft
        self.adapter_path = adapter_path

        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        if adapter_path is not None:
            if quantization and peft_trainable:
                model = prepare_model_for_kbit_training(model)
            self.model = PeftModel.from_pretrained(
                model,
                adapter_path,
                is_trainable=peft_trainable,
            )
        elif peft:
            self.model = self.get_lora_perf(model)
        else:
            self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def find_all_linear_names(self, model):
        """
        find all module for LoRA fine-tuning.
        """
        cls = bnb.nn.Linear4bit if self.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        return bnb_config

    def get_lora_perf(self, model):
        """
        LoRA configuration.
        """
        target_modules = self.find_all_linear_names(model)
        config = LoraConfig(
            target_modules=target_modules,
            r=16,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            lora_dropout=0.2,  # dropout probability for layers
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, config)

        return model

    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        if self.quantization:
            bnb_config = self.create_bnb_config()
            model = ModelClass.from_pretrained(model_name,
                                               quantization_config=bnb_config,
                                               #attn_implementation="flash_attention_2",
                                               #torch_type=torch.bfloat16,
                                               cache_dir=self.cache_dir)
        else:
            model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = remove_hook_from_module(model, recurse=True)
        model.config.use_cache = False
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True)
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        # if self.llm_name in ["BERT", "ST", "e5"]:
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

        # else:
        #     return outputs[text_tokens["input_ids"] == 2] # llama2 EOS token

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens["attention_mask"]


class EagerSentenceEncoder:
    """Tokenize and encode graph text when a PyG batch reaches the model."""

    def _init_eager_text_encoder(
        self,
        cache_dir,
        peft,
        quantization,
        train_text_encoder, #是否将分词器也加入训练
        adapter_path,
        max_length,
        text_batch_size,
    ):
        if train_text_encoder and quantization and not peft:
            raise ValueError(
                "Training a quantized base LLM requires PEFT. "
                "Set llm_peft=True or llm_quantization=False."
            )
        self.train_text_encoder = train_text_encoder
        self.text_batch_size = text_batch_size
        self.llm_model = LLMModel(
            self.llm_name,
            quantization=quantization,
            peft=peft,
            cache_dir=cache_dir,
            max_length=max_length,
            adapter_path=adapter_path,
            peft_trainable=train_text_encoder,
        )
        if not self.train_text_encoder:
            self.llm_model.requires_grad_(False)
            self.llm_model.eval()

    def _encode_texts(self, texts):
        num_texts = len(texts)
        if num_texts == 0:
            raise ValueError("Cannot encode an empty text batch")
        batch_size = self.text_batch_size if self.text_batch_size > 0 else num_texts
        device = next(self.llm_model.parameters()).device
        outputs = []
        for start in range(0, num_texts, batch_size):
            end = start + batch_size
            text_batch = texts[start:end]
            if not isinstance(text_batch, list):
                text_batch = text_batch.tolist()
            token_batch = self.llm_model.tokenizer(
                text_batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.llm_model.max_length,
            )
            token_batch = {key: value.to(device) for key, value in token_batch.items()}
            if not self.train_text_encoder:
                # Lightning calls train() on the parent module every epoch.
                self.llm_model.eval()
                output, _ = self.llm_model.encode(token_batch, pooling=True)
            else:
                output = self.llm_model(token_batch)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _as_text_list(values):
        if isinstance(values, np.ndarray):
            values = values.reshape(-1).tolist()
        else:
            values = list(values)
        return [value.decode() if isinstance(value, bytes) else str(value) for value in values]

    def _prepare_graph_texts(self, g):
        node_texts = self._as_text_list(g.x)
        edge_texts = self._as_text_list(g.edge_attr)
        unique_texts = []
        text_mapping = []
        text_to_index = {}

        for texts in (node_texts, edge_texts):
            for text in texts:
                text_index = text_to_index.get(text)
                if text_index is None:
                    text_index = len(unique_texts)
                    text_to_index[text] = text_index
                    unique_texts.append(text)
                text_mapping.append(text_index)

        return unique_texts, text_mapping, len(node_texts)

    def _restore_graph_text_features(self, g, text_features, text_mapping, num_nodes):
        text_features = text_features.to(self.llm_proj.weight.dtype)
        expected_texts = num_nodes + g.num_edges
        if len(text_mapping) != expected_texts:
            raise ValueError(
                f"Expected {expected_texts} node/edge text mappings, "
                f"but received {len(text_mapping)}."
            )
        text_mapping = torch.as_tensor(
            text_mapping, dtype=torch.long, device=text_features.device
        )
        text_features = text_features[text_mapping]
        # 拆回节点和边特征
        g.x = text_features[:num_nodes]
        g.edge_attr = text_features[num_nodes:]
        return g

    def _encode_graph_texts(self, g):
        unique_texts, text_mapping, num_nodes = self._prepare_graph_texts(g)
        text_features = self._encode_texts(unique_texts)
        return self._restore_graph_text_features(
            g, text_features, text_mapping, num_nodes
        )

    def forward(self, g):
        g = self._encode_graph_texts(g)
        return super().forward(g)

    def save_peft_adapter(self, output_dir):
        if not self.llm_model.peft:
            raise ValueError("save_peft_adapter requires llm_peft=True.")
        self.llm_model.model.save_pretrained(output_dir)
        self.llm_model.tokenizer.save_pretrained(output_dir)


class BinGraphLLMModel(EagerSentenceEncoder, BinGraphModel):
    def __init__(
        self,
        cache_dir="cache_data/model",
        peft=False,
        quantization=False,
        train_text_encoder=False,
        adapter_path=None,
        max_length=500,
        text_batch_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_eager_text_encoder(
            cache_dir,
            peft,
            quantization,
            train_text_encoder,
            adapter_path,
            max_length,
            text_batch_size,
        )


class BinGraphAttLLMModel(EagerSentenceEncoder, BinGraphAttModel):
    def __init__(
        self,
        cache_dir="cache_data/model",
        peft=False,
        quantization=False,
        train_text_encoder=False,
        adapter_path=None,
        max_length=500,
        text_batch_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_eager_text_encoder(
            cache_dir,
            peft,
            quantization,
            train_text_encoder,
            adapter_path,
            max_length,
            text_batch_size,
        )



class TransformerModel(nn.Module):
    """Transformer encoder model using Pytorch.
    Args:
        input_dim (int): Input dimension of the model.
        num_layers (int): Number of transformer layer.
        hidden_dim (int): Hidden dimension in transformer model.
        num_heads (int): Number of head in each transformer layer.

    """

    def __init__(
        self, input_dim: int, num_layers: int, hidden_dim: int, num_heads: int
    ):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_dim, num_heads, hidden_dim, batch_first=True
            ),
            num_layers,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
    ) -> Tensor:
        encoded = self.encoder(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask
        )
        return encoded


class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )

