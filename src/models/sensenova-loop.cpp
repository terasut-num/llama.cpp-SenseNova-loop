#include "models.h"

// SenseNova-Loop: a Qwen3.5 hybrid trunk that runs its middle layers several times. Inside the looped
// layers the residual is two streams, and each sublayer has a mHC router that mixes them.
//
// Each visit of a layer is a block of its own, so each visit has its own KV cache and recurrent state.
// The GGUF files in use come in three layouts:
//   - sensenova_loop + mhc.layer_kinds: one block per visit, repeated visits can omit their tensors
//   - sensenova_loop + loop_start/loop_end: one block per visit, all visits have tensors
//   - qwen35 + loop_start/loop_end: one block per physical layer

static bool sensenova_loop_enters(uint32_t kind) { return kind == 1 || kind == 3; }
static bool sensenova_loop_leaves(uint32_t kind) { return kind == 2 || kind == 3; }

void llama_model_sensenova_loop::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);

    // gated delta net
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    if (hparams.n_layer_nextn != 0) {
        throw std::runtime_error("sensenova_loop: MTP layers are not supported");
    }

    const uint32_t n_block = hparams.n_layer_all;

    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    const bool has_recr = ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, n_block, false);

    std::fill(hparams.mhc_layer_kinds.begin(), hparams.mhc_layer_kinds.end(), 0);

    uint32_t loop_start = 0;
    uint32_t n_phys     = n_block;

    layer_phys.clear();

    if (ml.get_key(LLM_KV_LOOP_START, loop_start, false)) {
        uint32_t loop_end = 0;
        uint32_t n_loops  = 0;
        ml.get_key(LLM_KV_LOOP_END, loop_end);
        if (!ml.get_key(LLM_KV_NUM_LOOPS, n_loops, false)) {
            ml.get_key(LLM_KV_LOOP_COUNT, n_loops);
        }
        ml.get_key(LLM_KV_PHYSICAL_LAYER_COUNT, n_phys, false);

        if (loop_start >= loop_end || loop_end > n_phys || n_loops < 1) {
            throw std::runtime_error(format("sensenova_loop: invalid loop %u..%u x %u over %u layers", loop_start, loop_end, n_loops, n_phys));
        }

        for (uint32_t p = 0; p < loop_start; ++p) {
            layer_phys.push_back(p);
        }
        for (uint32_t i = 0; i < n_loops; ++i) {
            for (uint32_t p = loop_start; p < loop_end; ++p) {
                layer_phys.push_back(p);
            }
        }
        for (uint32_t p = loop_end; p < n_phys; ++p) {
            layer_phys.push_back(p);
        }

        const uint32_t n_visit = layer_phys.size();
        if (n_visit > LLAMA_MAX_LAYERS) {
            throw std::runtime_error(format("sensenova_loop: %u layer visits exceed the limit of %d", n_visit, LLAMA_MAX_LAYERS));
        }

        if (n_block == n_visit) {
            phys_names = false;
        } else if (n_block == n_phys) {
            phys_names = true;
        } else {
            throw std::runtime_error(format("sensenova_loop: block count %u matches neither %u layers nor %u visits", n_block, n_phys, n_visit));
        }

        if (!ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.n_mhc_streams, false)) {
            ml.get_key(LLM_KV_MHC_STREAMS_COUNT, hparams.n_mhc_streams, false);
        }
        if (!ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.n_mhc_iters, false)) {
            ml.get_key(LLM_KV_MHC_ITERATIONS, hparams.n_mhc_iters, false);
        }
        ml.get_key(LLM_KV_MHC_PRE_SCALE, hparams.f_mhc_pre_scale, false);

        // the routers cover the looped layers, each pass enters and leaves the streams
        if (hparams.n_mhc_streams > 0) {
            const auto inside = [&](uint32_t p) { return p >= loop_start && p < loop_end; };
            for (uint32_t v = 0; v < n_visit; ++v) {
                const uint32_t p = layer_phys[v];
                if (!inside(p)) {
                    continue;
                }
                const bool enters = v == 0           || !inside(layer_phys[v - 1]) || layer_phys[v - 1] >= p;
                const bool leaves = v + 1 == n_visit || !inside(layer_phys[v + 1]) || layer_phys[v + 1] <= p;
                hparams.mhc_layer_kinds[v] = enters ? (leaves ? 3 : 1) : (leaves ? 2 : 4);
            }
        }

        if (phys_names) {
            // the per-layer hparams were read for the physical layers, lay them out per visit
            const auto per_visit = [&](std::array<uint32_t, LLAMA_MAX_LAYERS> & arr) {
                const auto src = arr;
                for (uint32_t v = 0; v < n_visit; ++v) {
                    arr[v] = src[layer_phys[v]];
                }
            };
            per_visit(hparams.n_head_arr);
            per_visit(hparams.n_head_kv_arr);
            per_visit(hparams.n_ff_arr);
            per_visit(hparams.is_recr_impl);
            hparams.n_layer_all = n_visit;
        }
    } else {
        ml.get_key(LLM_KV_MHC_STREAMS, hparams.n_mhc_streams, false);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.n_mhc_iters, false);
        if (hparams.n_mhc_streams > 0) {
            ml.get_key_or_arr(LLM_KV_MHC_LAYER_KINDS, hparams.mhc_layer_kinds, n_block, true);
        }

        // the first pass through the streams is the looped block, the passes after it repeat it
        int first = -1;
        int len   = 0;
        for (int v = 0; v < (int) n_block && len == 0; ++v) {
            if (first < 0 && sensenova_loop_enters(hparams.mhc_layer_kinds[v])) {
                first = v;
            }
            if (first >= 0 && sensenova_loop_leaves(hparams.mhc_layer_kinds[v])) {
                len = v - first + 1;
            }
        }
        int n_pass = len > 0 ? 1 : 0;
        while (len > 0 && first + (n_pass + 1)*len <= (int) n_block && sensenova_loop_enters(hparams.mhc_layer_kinds[first + n_pass*len])) {
            ++n_pass;
        }

        for (int v = 0; v < (int) n_block; ++v) {
            if (n_pass < 2 || v < first + len) {
                layer_phys.push_back(v);
            } else if (v < first + n_pass*len) {
                layer_phys.push_back(first + (v - first) % len);
            } else {
                layer_phys.push_back(v - (n_pass - 1)*len);
            }
        }
        n_phys = layer_phys.back() + 1;
    }

    if (!has_recr) {
        for (uint32_t v = 0; v < hparams.n_layer_all; ++v) {
            hparams.is_recr_impl[v] = (layer_phys[v] + 1) % full_attn_interval != 0;
        }
    }

    if (hparams.n_mhc_streams != 0 && hparams.n_mhc_streams != 2) {
        throw std::runtime_error(format("sensenova_loop: %u mHC streams are not supported, only 2", hparams.n_mhc_streams));
    }
    if (hparams.n_mhc_iters < 1) {
        throw std::runtime_error("sensenova_loop: invalid number of sinkhorn iterations");
    }

    LLAMA_LOG_INFO("%s: %u layers run as %u blocks, %u mHC streams, %u sinkhorn iterations\n", __func__,
            n_phys, hparams.n_layer_all, hparams.n_mhc_streams, hparams.n_mhc_iters);

    type = hparams.n_embd == 4096 ? LLM_TYPE_9B : LLM_TYPE_UNKNOWN;
}

void llama_model_sensenova_loop::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t head_v_dim = hparams.ssm_d_state;
    const int64_t n_k_heads  = hparams.ssm_n_group;
    const int64_t n_v_heads  = hparams.ssm_dt_rank;
    const int64_t key_dim    = head_k_dim * n_k_heads;
    const int64_t value_dim  = head_v_dim * n_v_heads;
    const int64_t conv_dim   = key_dim * 2 + value_dim;

    const int64_t n_str    = hparams.n_mhc_streams;
    const int64_t n_logits = 2*n_str + n_str*n_str;

    const auto has_tensor = [&](const LLM_TN_IMPL & name) {
        return ml.get_tensor_meta(name.str().c_str()) != nullptr;
    };

    // block of the first visit of each physical layer
    std::vector<int> first_block(*std::max_element(layer_phys.begin(), layer_phys.end()) + 1, -1);

    for (int il = 0; il < n_layer; ++il) {
        const int phys = layer_phys[il];
        const int bid  = phys_names ? phys : il;

        // a repeated visit without tensors of its own runs the weights of the first visit
        if (first_block[phys] >= 0 && (phys_names || !has_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il)))) {
            layers[il] = layers[first_block[phys]];
            continue;
        }
        if (first_block[phys] < 0) {
            first_block[phys] = il;
        }

        auto & layer = layers[il];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", bid), { n_embd }, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", bid), { n_embd }, 0);

        if (!hparams.is_recr(il)) {
            create_tensor_qkv(layer, bid, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", bid), { n_embd_head_k * n_head, n_embd }, 0);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", bid), { n_embd_head_k }, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", bid), { n_embd_head_k }, 0);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", bid), { n_embd, key_dim * 2 + value_dim }, 0);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", bid), { n_embd, value_dim }, 0);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", bid), { hparams.ssm_d_conv, conv_dim }, 0);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   bid), { hparams.ssm_dt_rank }, 0);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         bid), { hparams.ssm_dt_rank }, 0);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", bid), { n_embd, n_v_heads }, 0);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", bid), { n_embd, n_v_heads }, 0);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", bid), { head_v_dim }, 0);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", bid), { value_dim, n_embd }, 0);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", bid), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", bid), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", bid), { n_embd, n_ff }, 0);

        if (hparams.mhc_layer_kinds[il] != 0) {
            // the notfakerv layout stores the routers under the hc_*_fn / hc_*_base names
            if (has_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", bid))) {
                layer.mhc_attn_w = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,   "weight", bid), { n_str*n_embd, n_logits }, 0);
                layer.mhc_attn_b = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "bias",   bid), { n_logits }, 0);
                layer.mhc_ffn_w  = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,    "weight", bid), { n_str*n_embd, n_logits }, 0);
                layer.mhc_ffn_b  = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,  "bias",   bid), { n_logits }, 0);
            } else {
                layer.mhc_attn_w = create_tensor(tn(LLM_TENSOR_MHC_ATTN_W, bid), { n_str*n_embd, n_logits }, 0);
                layer.mhc_attn_b = create_tensor(tn(LLM_TENSOR_MHC_ATTN_B, bid), { n_logits }, 0);
                layer.mhc_ffn_w  = create_tensor(tn(LLM_TENSOR_MHC_FFN_W,  bid), { n_str*n_embd, n_logits }, 0);
                layer.mhc_ffn_b  = create_tensor(tn(LLM_TENSOR_MHC_FFN_B,  bid), { n_logits }, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_sensenova_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_sensenova_loop::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // inside the mHC range the residual is two streams: the two halves of the last merge output
    ggml_tensor * str_a = nullptr;
    ggml_tensor * str_b = nullptr;

    // route the streams into the sublayer input, run the sublayer, merge its output back into the streams
    auto build_mhc_sublayer = [&](ggml_tensor * w, ggml_tensor * bias, ggml_tensor * norm_w, bool is_attn, int il) {
        const int64_t d = str_a->ne[0];

        ggml_tensor * packed = ggml_mhc_route(ctx0, str_a, str_b, w, bias, hparams.n_mhc_iters);
        cb(packed, is_attn ? "mhc_attn_route" : "mhc_ffn_route", il);

        ggml_tensor * x = ggml_view_2d(ctx0, packed, d, n_tokens, packed->nb[1], 0);
        if (hparams.f_mhc_pre_scale != 1.0f) {
            x = ggml_scale(ctx0, ggml_cont(ctx0, x), hparams.f_mhc_pre_scale);
        }

        x = build_norm(x, norm_w, nullptr, LLM_NORM_RMS, il);
        cb(x, is_attn ? "attn_norm" : "attn_post_norm", il);

        if (!is_attn) {
            x = build_layer_ffn(x, il);
        } else if (hparams.is_recr(il)) {
            x = build_layer_attn_linear(inp->get_recr(), x, il);
        } else {
            x = build_layer_attn(inp->get_attn(), x, inp_pos, sections, il);
        }

        ggml_tensor * merged = ggml_mhc_merge(ctx0, str_a, str_b, x, packed);
        cb(merged, is_attn ? "mhc_attn_merge" : "mhc_ffn_merge", il);

        str_a = ggml_view_2d(ctx0, merged, d, n_tokens, merged->nb[1], 0);
        str_b = ggml_view_2d(ctx0, merged, d, n_tokens, merged->nb[1], d*merged->nb[0]);
    };

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const uint32_t mhc_kind = hparams.n_mhc_streams > 0 ? hparams.mhc_layer_kinds[il] : 0;

        if (mhc_kind == 0) {
            res->t_layer_inp[il] = inpL;

            ggml_tensor * inpSA = inpL;

            cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            ggml_build_forward_expand(gf, cur);

            if (hparams.is_recr(il)) {
                cur = build_layer_attn_linear(inp->get_recr(), cur, il);
            } else {
                cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
            }

            if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
                cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, inpSA);
            cb(cur, "attn_residual", il);

            ggml_tensor * ffn_residual = cur;
            ggml_tensor * attn_post_norm = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(attn_post_norm, "attn_post_norm", il);

            cur = build_layer_ffn(attn_post_norm, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_residual);
            cb(cur, "post_ffn", il);

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
            continue;
        }

        // entering the mHC range: both streams start as the single residual
        if (sensenova_loop_enters(mhc_kind)) {
            str_a = inpL;
            str_b = inpL;
        }
        GGML_ASSERT(str_a != nullptr && "sensenova_loop: mHC block without streams");

        // inside the streams the first one stands in for the layer input
        res->t_layer_inp[il] = str_a;

        build_mhc_sublayer(layer.mhc_attn_w, layer.mhc_attn_b, layer.attn_norm,      true,  il);
        build_mhc_sublayer(layer.mhc_ffn_w,  layer.mhc_ffn_b,  layer.attn_post_norm, false, il);

        // leaving the mHC range: the residual is the mean of the streams
        if (sensenova_loop_leaves(mhc_kind)) {
            cur = ggml_scale(ctx0, ggml_add(ctx0, str_a, str_b), 0.5f);
            cb(cur, "mhc_leave", il);

            if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }

            cur = build_cvec(cur, il);
            cb(cur, "l_out", il);

            inpL  = cur;
            str_a = nullptr;
            str_b = nullptr;
        }
    }
    GGML_ASSERT(str_a == nullptr && "sensenova_loop: the last block must leave the mHC streams");

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_sensenova_loop::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_sensenova_loop::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_sensenova_loop::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_sensenova_loop::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_sensenova_loop::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}
