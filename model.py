import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
import random
import time
import scipy.stats
from utils import set_dataset, _preprocess2, _preprocess3, convert_models_to_fp32, save_metrics_csv_and_plot
import clip as _clip


def build_prompts(attrs, levels, templates=None):
    if templates is None:
        templates = ["{attr}, {level}"]
    phrases_per_class = []
    for a in attrs:
        for l in levels:
            phrases_per_class.append([t.format(attr=a, level=l) for t in templates])
    if len(templates) == 1:
        return [ps[0] for ps in phrases_per_class]
    return phrases_per_class


def freeze_text_encoder(clip_model):
    clip_model.token_embedding.requires_grad_(False)
    clip_model.positional_embedding.requires_grad_(False)
    for p in clip_model.transformer.parameters():
        p.requires_grad_(False)
    clip_model.ln_final.weight.requires_grad_(False)
    clip_model.ln_final.bias.requires_grad_(False)
    clip_model.text_projection.requires_grad_(False)
    clip_model.logit_scale.requires_grad_(False)


class CoOpTextPrompt(nn.Module):
    def __init__(self, clip_model, class_phrases, n_ctx=4):
        super().__init__()
        self._clip = weakref.ref(clip_model)
        # self.model = clip_model
        self.n_ctx = n_ctx
        self.dtype = clip_model.dtype

        if isinstance(class_phrases[0],str):
            self.num_classes = len(class_phrases)
            self.num_templates = 1
            flat_phrases = class_phrases
            template_idx = torch.zeros(self.num_classes, dtype=torch.long)
        else:
            self.num_classes = len(class_phrases) # C
            self.num_templates = len(class_phrases[0]) # K
            flat_phrases = [p for ps in class_phrases for p in ps]
            template_idx = torch.arange(self.num_templates).repeat_interleave(self.num_classes)
            template_idx = torch.arange(self.num_templates).repeat(self.num_classes)

        self.register_buffer("template_idx", template_idx)
        self.register_buffer("is_multi", torch.tensor(int(self.num_templates>1)))
        tokenized = _clip.tokenize(flat_phrases)  # [C*K, 77]
        self.register_buffer("tokenized", tokenized)

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx = nn.Parameter(torch.randn(self.num_templates,n_ctx, ctx_dim, dtype=self.dtype) * 0.02)  # trainable

        with torch.no_grad():
            emb = clip_model.token_embedding(tokenized).to(self.dtype)  # [18, 77, D], D is text hidden dim
        self.register_buffer("token_prefix", emb[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", emb[:, 1:, :])  # content + [EOS] + padding [18, 76, D]
        self.register_buffer("eot_idx", tokenized.argmax(dim=-1))  # [18]

    def forward(self):
        CK = self.tokenized.size(0)
        ctx = self.ctx[self.template_idx] #if self.num_templates > 1 else self.ctx  # [C*K, n_ctx, D]
        model = self._clip()
        # C = self.tokenized.size(0)  # number of classes
        # ctx = self.ctx.unsqueeze(0).expand(C, -1, -1).to(self.dtype)  # [C, n_ctx, D]
        x = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)  # [C, 1+n_ctx+76, D]
        x = x[:, :77, :]  # in case n_ctx is large

        x = x + model.positional_embedding[:x.size(1), :].to(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x).to(self.dtype)

        eot = (self.eot_idx + self.n_ctx).clamp(max=x.size(1) - 1)  # [C]
        # text_features = x[torch.arange(C), eot] @ model.text_projection  # [C, D]
        # text_features = F.normalize(text_features, dim=-1)
        # return text_features
        txt = x[torch.arange(CK), eot, :] @ model.text_projection  # [C*K, D]
        txt = F.normalize(txt, dim=-1)

        if self.is_multi.item()==1:
            txt = txt.view(self.num_classes, self.num_templates, -1).mean(dim=1)  # [C, D]
            txt = F.normalize(txt, dim=-1)
        return txt  # [C, D]


class ClipPromptClassifier(nn.Module):
    def __init__(self, clip_model, class_phrases, n_ctx=8):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.text_prompt = CoOpTextPrompt(clip_model, class_phrases, n_ctx=n_ctx)
        self.log_alpha = nn.Parameter(torch.tensor(0.0)) # learnable log temperature
        self.bias = nn.Parameter(torch.zeros(len(class_phrases))) # learnable bias
        # freeze_text_encoder(self.model)

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)  # [B, D]
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.text_prompt()  # [C, D]

        # logit_scale = self.clip_model.logit_scale.exp()
        sims = image_features @ text_features.t()  # [B, C]
        logits = torch.exp(self.log_alpha) * sims + self.bias  # [B, C]
        return logits
