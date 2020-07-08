# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import AutoModel, AutoModelForPreTraining
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling_bert import (
    BertConfig,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
)

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.utils.modeling import get_optimizer_parameters_for_bert


class MMFTTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MMFTImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(5, config.hidden_size)

        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))

        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MMFTransformerBase(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = MMFTTextEmbeddings(config)
        self.img_embeddings = MMFTImageEmbeddings(config, img_dim)
        auto_model = AutoModel.from_pretrained(
            config.bert_model_name,
            config=config,
            cache_dir=os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(-1)
            ),
        )
        self.encoder = auto_model.encoder
        self.pooler = BertPooler(config)
        self.fixed_head_masks = [None for _ in range(len(self.encoder.layer))]
        self.init_weights()

    def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat, img_type_embeddings)
        return output

    def _compute_img_txt_embeddings(
        self,
        input_ids,
        position_ids,
        img_feat,
        img_pos_feat,
        gather_index,
        txt_type_ids=None,
        img_type_ids=None,
    ):
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(img_feat, img_pos_feat, img_type_ids)
        embedding_output = torch.cat([txt_emb, img_emb], dim=1)
        return embedding_output

    def forward(
        self,
        input_ids,
        position_ids,
        img_feat,
        img_pos_feat,
        attention_mask,
        gather_index=None,
        output_all_encoded_layers=True,
        txt_type_ids=None,
        img_type_ids=None,
    ):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_type_ids
            )
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids
            )
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids,
                position_ids,
                img_feat,
                img_pos_feat,
                gather_index,
                txt_type_ids,
                img_type_ids,
            )

        # BERT based Encoder
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, self.fixed_head_masks,
        )
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        pooled_output = self.pooler(encoded_layers[-1])
        return encoded_layers[-1], pooled_output


class MMFTransformerForPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        self.bert = MMFTransformerBase.from_pretrained(
            self.config.bert_model_name,
            config=self.bert_config,
            cache_dir=os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(-1)
            ),
            img_dim=2048,
        )
        # self.bert.embeddings = MMFTEmbeddings(self.bert.config)

        self.vocab_size = self.bert.config.vocab_size

        bert_masked_lm = AutoModelForPreTraining.from_pretrained(
            self.config.bert_model_name,
            cache_dir=os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(-1)
            ),
        )
        self.cls = copy.deepcopy(bert_masked_lm.cls)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()
                self.cls.apply(self.bert._init_weights)

            self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we
            are cloning them instead.
        """
        self.bert._tie_or_clone_weights(
            self.cls.predictions.decoder, self.bert.embeddings.word_embeddings
        )

    def forward(
        self,
        dataset_name,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        gather_index=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
    ):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            attention_mask,
            gather_index,
        )

        output_dict = {}

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )

        # MLM Loss
        if masked_lm_labels is not None:
            output_dict["logits"] = prediction_scores
            masked_lm_loss = self.loss_fct(
                prediction_scores.contiguous().view(-1, self.vocab_size),
                masked_lm_labels.contiguous().view(-1),
            )
            output_dict["masked_lm_loss"] = masked_lm_loss

        return output_dict


class MMFTransformerForClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert_model_name = getattr(self.config, "bert_model_name", None)
        self.bert_config = BertConfig.from_dict(
            OmegaConf.to_container(self.config, resolve=True)
        )
        self.bert = MMFTransformerBase.from_pretrained(
            self.config.bert_model_name,
            config=self.bert_config,
            cache_dir=os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_{}".format(-1)
            ),
            img_dim=2048,
        )

        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, self.config.num_labels),
        )

        self.init_weights()

    def init_weights(self):
        if self.config.random_initialize is False:
            if self.bert_model_name is None:
                # No pretrained model, init weights
                self.bert.init_weights()

            # Classifier needs to be initialized always as it is task specific
            self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        dataset_name,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        position_embeddings_visual=None,
        visual_embeddings_type=None,
        image_text_alignment=None,
        masked_lm_labels=None,
    ):
        sequence_output, pooled_output, attention_weights = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            visual_embeddings,
            position_embeddings_visual,
            visual_embeddings_type,
            image_text_alignment,
        )

        output_dict = {}

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.contiguous().view(-1, self.num_labels)
        output_dict["scores"] = reshaped_logits

        return output_dict


@registry.register_model("mmf_transformer")
class MMFTransformer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/models/mmf_transformer/defaults.yaml"

    def build(self):
        if self.config.training_head_type == "pretraining":
            self.model = MMFTransformerForPretraining(self.config)
        else:
            self.model = MMFTransformerForClassification(self.config)

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def get_optimizer_parameters(self, config):
        return get_optimizer_parameters_for_bert(self.model, config)

    def update_sample_list_based_on_head(self, sample_list):
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids

        image_info = getattr(sample_list, "image_info_0", {})
        image_dim_variable = getattr(image_info, "max_features", None)
        image_feat_variable = getattr(sample_list, "image_feature_0", None)
        bbox = np.array(getattr(image_info, "bbox", None), dtype=np.float32)
        image_w = np.array(getattr(image_info, "image_width", None), dtype=np.float32)
        image_h = np.array(getattr(image_info, "image_height", None), dtype=np.float32)
        image_location = np.zeros((bbox.shape[0], bbox.shape[1], 5), dtype=np.float32)
        image_location[:, :, :4] = bbox
        image_location[:, :, 4] = (
            (image_location[:, :, 3] - image_location[:, :, 1])
            * (image_location[:, :, 2] - image_location[:, :, 0])
            / (image_w * image_h)[:, None]
        )
        image_location[:, :, 0] = image_location[:, :, 0] / image_w[:, None]
        image_location[:, :, 1] = image_location[:, :, 1] / image_h[:, None]
        image_location[:, :, 2] = image_location[:, :, 2] / image_w[:, None]
        image_location[:, :, 3] = image_location[:, :, 3] / image_h[:, None]
        image_loc_variable = torch.tensor(image_location, dtype=torch.float).cuda()

        sample_list.visual_embeddings = image_feat_variable
        sample_list.position_embeddings_visual = image_loc_variable
        sample_list.image_dim = image_dim_variable
        sample_list.input_ids = bert_input_ids
        sample_list.input_mask = bert_input_mask
        sample_list.token_type_ids = bert_input_type_ids
        return sample_list

    def add_custom_params(self, sample_list):
        visual_embeddings = getattr(sample_list, "visual_embeddings", None)
        image_dim = getattr(sample_list, "image_dim", None)
        # pretraining labels
        sample_list.masked_lm_labels = getattr(sample_list, "lm_label_ids", None)
        # image_feat_variable = batch x ( num_choice x ) image_feature_length x dim
        # Prepare Mask
        if visual_embeddings is not None and image_dim is not None:
            image_mask = (
                torch.arange(visual_embeddings.size(-2))
                .expand(*visual_embeddings.size()[:-1])
                .cuda()
            )
            if len(image_dim.size()) < len(image_mask.size()):
                image_dim = image_dim.unsqueeze(-1)
                assert len(image_dim.size()) == len(image_mask.size())
            image_mask = image_mask < image_dim
            sample_list.image_mask = image_mask.long()
        else:
            sample_list.image_mask = None

        attention_mask = torch.cat(
            (sample_list.input_mask, sample_list.image_mask), dim=-1
        )
        if sample_list.masked_lm_labels is not None:
            assert sample_list.masked_lm_labels.size(-1) == sample_list.input_mask.size(
                -1
            )
            new_lm_labels = torch.ones_like(attention_mask) * -1
            size_masked_lm_labels = sample_list.masked_lm_labels.size()
            assert len(size_masked_lm_labels) == 2
            new_lm_labels[
                : size_masked_lm_labels[0], : size_masked_lm_labels[1]
            ] = sample_list.masked_lm_labels
            sample_list.masked_lm_labels = new_lm_labels

        sample_list.attention_mask = attention_mask

        return sample_list

    def forward(self, sample_list):
        sample_list = self.update_sample_list_based_on_head(sample_list)
        sample_list = self.add_custom_params(sample_list)

        output_dict = self.model(
            sample_list.dataset_name,
            sample_list.input_ids,
            sample_list.attention_mask,
            sample_list.token_type_ids,
            sample_list.visual_embeddings,
            sample_list.position_embeddings_visual,
            masked_lm_labels=sample_list.masked_lm_labels,
        )

        if "pretraining" in self.config.training_head_type:
            loss_key = "{}/{}".format(
                sample_list.dataset_name, sample_list.dataset_type
            )
            output_dict["losses"] = {}
            output_dict["losses"][loss_key + "/masked_lm_loss"] = output_dict.pop(
                "masked_lm_loss"
            )
        return output_dict
