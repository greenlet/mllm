import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, BertTokenizer, BertGenerationEncoder, BertGenerationDecoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

from mllm.model.embgen_bert import EncoderEmbDecoderModel, EncEmbExpansionType
from mllm.train.utils import QnaQuesInp, QnaBatch, qna_loss


def run_eed_model_on_batch_old(model: EncoderEmbDecoderModel, batch: QnaBatch) -> torch.Tensor:
    ctxs_toks, other_toks = batch.gen_tensors()
    ctxs_mask = (ctxs_toks > 0).to(batch.device)
    ctx_enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=ctxs_toks, attention_mask=ctxs_mask)
    ctx_lhs = ctx_enc_out.last_hidden_state

    if batch.ques_inp == QnaQuesInp.Enc:
        q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l = other_toks
        loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
        n_ans = len(a_toks_l)
        for q_toks, a_toks, a_att_mask, a_tgt_mask in zip(q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l):
            q_toks = q_toks.unsqueeze(0)
            q_mask = (q_toks > 0).to(batch.device)
            q_enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=q_toks, attention_mask=q_mask)
            ctxq_lhs = torch.concatenate([ctx_lhs, q_enc_out.last_hidden_state], dim=0)
            ctxq_emb = model.run_expansion(ctxq_lhs)
            a_toks = a_toks.repeat(len(a_att_mask), 1)
            a_toks_inp = torch.tril(a_toks)
            a_toks_inp[a_tgt_mask] = batch.tkz.mask_token_id
            a_dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
                input_ids=a_toks_inp, attention_mask=a_att_mask, encoder_hidden_states=ctxq_emb, use_cache=False,
            )
            l = qna_loss(a_dec_out.logits, a_toks, a_tgt_mask)
            loss = loss + l
        loss = loss / n_ans
        return loss

    if batch.ques_inp == QnaQuesInp.Dec:
        ctx_emb = model.run_expansion(ctx_enc_out.last_hidden_state)
        qa_toks_l, qa_att_masks_l, qa_tgt_masks_l = other_toks
        loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
        n_qas = len(qa_toks_l)
        for ind in range(n_qas):
            qa_toks, qa_att_mask, qa_tgt_mask = qa_toks_l[ind].unsqueeze(0), qa_att_masks_l[ind], qa_tgt_masks_l[ind]
            qa_toks = qa_toks.repeat(len(qa_att_mask), 1)
            qa_toks_inp = qa_toks * qa_att_mask
            qa_toks_inp[qa_tgt_mask] = batch.tkz.mask_token_id
            dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
                input_ids=qa_toks_inp, attention_mask=qa_att_mask, encoder_hidden_states=ctx_emb, use_cache=False,
            )
            l = qna_loss(dec_out.logits, qa_toks, qa_tgt_mask)
            loss = loss + l
        loss = loss / n_qas
        return loss

    raise Exception(f'Question input type {batch.ques_inp} is not supported.')


def run_eed_model_on_batch(model: EncoderEmbDecoderModel, batch: QnaBatch) -> torch.Tensor:
    ctxs_toks, other_toks = batch.gen_tensors()
    ctxs_mask = (ctxs_toks > 0).to(batch.device)
    ctx_enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=ctxs_toks, attention_mask=ctxs_mask)
    ctx_lhs = ctx_enc_out.last_hidden_state

    if batch.ques_inp == QnaQuesInp.Enc:
        q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l = other_toks
        loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
        n_ans = len(a_toks_l)
        for q_toks, a_toks, a_att_mask, a_tgt_mask in zip(q_toks_l, a_toks_l, a_att_masks_l, a_tgt_masks_l):
            q_toks = q_toks.unsqueeze(0)
            q_mask = (q_toks > 0).to(batch.device)
            q_enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=q_toks, attention_mask=q_mask)
            ctxq_lhs = torch.concatenate([ctx_lhs, q_enc_out.last_hidden_state], dim=0)
            ctxq_emb = model.run_expansion(ctxq_lhs)
            if a_toks[0] != batch.tkz.cls_token_id:
                a_toks = F.pad(a_toks, (1, 0), 'constant', batch.tkz.cls_token_id)
            a_toks_inp = a_toks.unsqueeze(0)
            a_dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
                input_ids=a_toks_inp, encoder_hidden_states=ctxq_emb,
            )

            logits = a_dec_out.logits.view(-1, model.decoder.config.vocab_size)[:-1]
            labels = a_toks_inp[0][1:]
            l = F.cross_entropy(logits, labels)

            loss = loss + l
        loss = loss / n_ans
        return loss

    # if batch.ques_inp == QnaQuesInp.Dec:
    #     ctx_emb = model.run_expansion(ctx_enc_out.last_hidden_state)
    #     qa_toks_l, qa_att_masks_l, qa_tgt_masks_l = other_toks
    #     loss = torch.tensor(0, dtype=torch.float32, device=batch.device)
    #     n_qas = len(qa_toks_l)
    #     for ind in range(n_qas):
    #         qa_toks, qa_att_mask, qa_tgt_mask = qa_toks_l[ind].unsqueeze(0), qa_att_masks_l[ind], qa_tgt_masks_l[ind]
    #         qa_toks = qa_toks.repeat(len(qa_att_mask), 1)
    #         qa_toks_inp = qa_toks * qa_att_mask
    #         qa_toks_inp[qa_tgt_mask] = batch.tkz.mask_token_id
    #         dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
    #             input_ids=qa_toks_inp, attention_mask=qa_att_mask, encoder_hidden_states=ctx_emb, use_cache=False,
    #         )
    #         l = qna_loss(dec_out.logits, qa_toks, qa_tgt_mask)
    #         loss = loss + l
    #     loss = loss / n_qas
    #     return loss

    raise Exception(f'Question input type {batch.ques_inp} is not supported.')


def get_eed_bert_model(inp_len: int, ques_inp: QnaQuesInp, enc_emb_exp_type: EncEmbExpansionType, enc_emb_exp_bias: bool,
                       batch_size: int, device: torch.device) -> tuple[PreTrainedTokenizer, EncoderEmbDecoderModel]:
    # model_name = 'google-bert/bert-base-uncased'
    model_name = 'bert-base-uncased'
    tkz = BertTokenizer.from_pretrained(model_name)
    print(tkz)
    enc_model: BertGenerationEncoder = BertGenerationEncoder.from_pretrained(model_name, bos_token_id=101, eos_token_id=102)
    # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
    dec_model: BertGenerationDecoder = BertGenerationDecoder.from_pretrained(
        model_name, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102,
        tie_word_embeddings=False,
    )
    enc_inp_batch_size = batch_size
    if ques_inp == QnaQuesInp.Enc:
        enc_inp_batch_size += 1
    model = EncoderEmbDecoderModel(
        encoder=enc_model, decoder=dec_model, enc_emb_exp_type=enc_emb_exp_type, enc_emb_exp_bias=enc_emb_exp_bias,
        enc_inp_len=inp_len, enc_inp_batch_size=enc_inp_batch_size,
    ).to(device)
    return tkz, model


# docs_toks_aug: [n_batch, seq_len]
# docs_toks_tgt: [n_tgt]
def run_eed_model_on_masked_input_old(model: EncoderEmbDecoderModel, tkz: PreTrainedTokenizer, docs_toks_aug: torch.Tensor, docs_toks_tgt: torch.Tensor) -> torch.Tensor:
    assert tkz.pad_token_id == 0, f'pad_token_id = {tkz.pad_token_id}'
    device = docs_toks_aug.device
    enc_toks_inp = docs_toks_aug
    enc_toks_inp_mask = (enc_toks_inp > 0).to(device)
    # [n_batch, seq_len] -> [n_batch, seq_len, d_model]
    enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=enc_toks_inp, attention_mask=enc_toks_inp_mask)
    enc_lhs = enc_out.last_hidden_state
    # model.config.enc_emb_exp_type = EncEmbExpansionType.Emb: [n_batch, seq_len, d_model] -> [n_batch, d_model]
    # model.config.enc_emb_exp_type = EncEmbExpansionType.Mat: [n_batch, seq_len, d_model] -> [seq_len, d_model]
    # [n_batch, seq_len, d_model] -> [dec_seq_len, d_model]
    enc_emb = model.run_expansion(enc_lhs)

    n_tgt = len(docs_toks_tgt)
    # [n_tgt] -> [n_tgt, n_tgt]
    dec_toks = docs_toks_tgt.repeat(n_tgt, 1)
    dec_toks_inp = torch.tril(dec_toks)
    dec_tgt_mask = torch.eye(n_tgt, dtype=torch.bool, device=device)
    dec_toks_inp[dec_tgt_mask] = tkz.mask_token_id
    dec_toks_inp_mask = (dec_toks_inp > 0).to(device)
    dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
        input_ids=dec_toks_inp, attention_mask=dec_toks_inp_mask, encoder_hidden_states=enc_emb, use_cache=False,
    )
    loss = qna_loss(dec_out.logits, dec_toks, dec_tgt_mask)
    return loss


# docs_toks_aug: [n_batch, seq_len]
# docs_toks_tgt: [n_tgt]
def run_eed_model_on_masked_input(model: EncoderEmbDecoderModel, tkz: PreTrainedTokenizer, docs_toks_aug: torch.Tensor, docs_toks_tgt: torch.Tensor) -> torch.Tensor:
    assert tkz.pad_token_id == 0, f'pad_token_id = {tkz.pad_token_id}'
    if docs_toks_tgt[0] != tkz.cls_token_id:
        docs_toks_tgt = F.pad(docs_toks_tgt, (1, 0), 'constant', tkz.cls_token_id)
    device = docs_toks_aug.device
    enc_toks_inp = docs_toks_aug
    enc_toks_inp_mask = (enc_toks_inp > 0).to(device)
    # [n_batch, seq_len] -> [n_batch, seq_len, d_model]
    enc_out: BaseModelOutputWithPastAndCrossAttentions = model.encoder(input_ids=enc_toks_inp, attention_mask=enc_toks_inp_mask)
    enc_lhs = enc_out.last_hidden_state
    # model.config.enc_emb_exp_type = EncEmbExpansionType.Emb: [n_batch, seq_len, d_model] -> [n_batch, d_model]
    # model.config.enc_emb_exp_type = EncEmbExpansionType.Mat: [n_batch, seq_len, d_model] -> [seq_len, d_model]
    # [n_batch, seq_len, d_model] -> [1, dec_seq_len, d_model]
    enc_emb = model.run_expansion(enc_lhs)

    n_tgt = len(docs_toks_tgt)
    # [n_tgt] -> [1, n_tgt]
    doc_toks_tgt = docs_toks_tgt.unsqueeze(0)
    dec_out: CausalLMOutputWithCrossAttentions = model.decoder(
        input_ids=doc_toks_tgt, encoder_hidden_states=enc_emb,
    )
    logits = dec_out.logits.view(-1, model.decoder.config.vocab_size)[:-1]
    labels = doc_toks_tgt[0][1:]
    loss = F.cross_entropy(logits, labels)
    return loss

