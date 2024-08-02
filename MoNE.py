import torch
import torch.nn as nn
from torch.nn import functional as F, init
from einops import rearrange
import math

def nested_expert_token_selection(raw_router_prob, x):
    B = x.shape[0] # Number of elements in the batch
    T = x.shape[1] # Number of tokens in the sequence
    E = raw_router_prob.shape[-1] # Number of nested experts

    # Clone the router probabilities so they can be used for the selection process but not impacted outside this metho
    router_prob = raw_router_prob.clone()

    expert_inputs = []
    sorted_router_probs = []
    # Each nested expert should choose which patch it wants to process so all nested experts are used equally
    for expert_ind in range(E):
        # Paper describes Capacity Distribution Across Experts for dynamic number of nested experts in order to hit a specified inference FLOP requirement.
        # This is done offline and is an area of improvement for this code
        # For an MVP, say each nested expert processes the same number of tokens
        num_tokens_to_expert = T // E

        _, expert_requests = torch.topk(router_prob[:, :, expert_ind], num_tokens_to_expert, dim=1) # NOTE TODO

        # NOTE torch does not support batched index selection
        # Loop over each element of the batch and select the requested tokens for that particular element and then concatenate them back into a single batch
        out = torch.cat([torch.index_select(x[i], 0, expert_requests[i]).unsqueeze(0) for i in range(B)])
        expert_inputs.append(out)

        # Since we need to keep an associated router probability with each token and this is inherently moving tokens around, keep track of the sorted router probability score for later layers
        sorted_router_prob = torch.cat([torch.index_select(raw_router_prob[i], 0, expert_requests[i]).unsqueeze(0) for i in range(B)])
        sorted_router_probs.append(sorted_router_prob)

        # Remove the requsted selected tokens from the pool of available tokens to choose from so the same token is not processed by multiple nested experts
        for i in range(B):
            router_prob[i, expert_requests[i], :] = 0.

    return expert_inputs, sorted_router_probs

class NEMHSA(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()

        dropout = 0.1
        self.num_heads = 8
        self.model_dim = model_dim

        self.norm = nn.LayerNorm(model_dim)

        self.q_w = nn.Parameter(torch.zeros(model_dim, model_dim))
        self.q_b = nn.Parameter(torch.zeros(model_dim))

        self.k_w = nn.Parameter(torch.zeros(model_dim, model_dim))
        self.k_b = nn.Parameter(torch.zeros(model_dim))

        self.v_w = nn.Parameter(torch.zeros(model_dim, model_dim))
        self.v_b = nn.Parameter(torch.zeros(model_dim))

        self.o_w = nn.Parameter(torch.zeros(model_dim, model_dim))
        self.o_b = nn.Parameter(torch.zeros(model_dim))

        self.norm = nn.LayerNorm(model_dim)
        self.scale = model_dim** -0.5
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.k_w, math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.k_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.k_b, -bound, bound)

        init.kaiming_uniform_(self.q_w, math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.q_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.q_b, -bound, bound)

        init.kaiming_uniform_(self.v_w, math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.v_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.v_b, -bound, bound)

        init.kaiming_uniform_(self.o_w, math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.o_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.o_b, -bound, bound)

    def forward(self, x, router_prob):
        # Use the computed router probabilities to route each token to a given nested expert (tokens are still full dimensional here)
        routed_inputs, sorted_router_probs = nested_expert_token_selection(router_prob, x)

        # To still apply token mixing across all tokens, use the nested experts to project each token to its full dimension
        # (slight compute reduction here over vanilla ViT since the qkv projections are reduced for the last n-1 experts)
        qs = []
        ks = []
        vs = []
        new_probs = []
        for expert_ind, expert_input in enumerate(routed_inputs):
            # Pre-norm ViT variant
            expert_input = self.norm(expert_input)

            # Extract the inputs for the specific expert
            m = self.model_dim // 2**(expert_ind)
            extracted_input = expert_input[:, :, 0:m]

            ne_q_w = self.q_w[:, 0:m]
            q = F.linear(extracted_input, ne_q_w, self.q_b)

            ne_k_w = self.k_w[:, 0:m]
            k = F.linear(extracted_input, ne_k_w, self.k_b)

            ne_v_w = self.v_w[:, 0:m]
            v = F.linear(extracted_input, ne_v_w, self.v_b)

            qs.append(q)
            ks.append(k)
            vs.append(v)
            new_probs.append(sorted_router_probs[expert_ind])
        q = torch.cat(qs, dim=1)
        k = torch.cat(ks, dim=1)
        v = torch.cat(vs, dim=1)
        new_probs = torch.cat(new_probs, dim=1)

        # Break apart for multi-headed SA
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # Apply standard multi-headed self attention
        attn = q @ k.transpose(-1, -2) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        attn_out = attn @ v
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
   
        # Route each mixed token to it's appropriate nested expert
        routed_outputs, sorted_router_probs = nested_expert_token_selection(new_probs, attn_out)

        # Perform final projection using nested experts
        out_projs = []
        for expert_ind, expert_output in enumerate(routed_outputs):
            # Extract the outputs for the specific expert
            m = self.model_dim // 2**(expert_ind)
            extracted_output = expert_output[:, :, 0:m]

            ne_o_w = self.o_w[0:m, 0:m]
            ne_o_b = self.o_b[0:m]
            out = F.linear(extracted_output, ne_o_w, ne_o_b)
            
            out_projs.append(out)

        # Residual connection
        routed_inputs, sorted_router_probs = nested_expert_token_selection(new_probs, x)
        final_outs = []
        final_probs = []
        for expert_output, expert_input, experts_router_prob in zip(out_projs, routed_inputs, sorted_router_probs):
            # Pad the tensor so that the residual connection can be applied
            pad_amount = expert_input.shape[-1] - expert_output.shape[-1]
            if pad_amount:
                expert_output = F.pad(expert_output, [0, pad_amount], value=0.)
            out = expert_input + expert_output
            final_outs.append(out)
            final_probs.append(experts_router_prob)

        final_outs = torch.cat(final_outs, dim=1)
        final_probs = torch.cat(final_probs, dim=1)

        return final_outs, final_probs

class NEMLP(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()

        # MoNE hardcodes their inner MLP dim to 4x the model dim
        self.inner_dim = model_dim * 4
        self.model_dim = model_dim

        dropout = 0.1

        self.norm = nn.LayerNorm(model_dim)
        self.l1_w = nn.Parameter(torch.zeros(self.inner_dim, self.model_dim))
        self.l1_b = nn.Parameter(torch.zeros(self.inner_dim))
        self.drop1 = nn.Dropout(dropout)

        self.l2_w = nn.Parameter(torch.zeros(self.model_dim, self.inner_dim))
        self.l2_b = nn.Parameter(torch.zeros(model_dim))
        self.drop2 = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        init.kaiming_uniform_(self.l1_w, math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.l1_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.l1_b, -bound, bound)

        init.kaiming_uniform_(self.l2_w, math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.l2_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.l2_b, -bound, bound)

    def forward(self, x, router_prob, alpha):
        # Use the computed router probabilities to route each token to a given nested expert (tokens are still full dimensional here)
        routed_inputs, sorted_router_probs = nested_expert_token_selection(router_prob, x)

        outer_projs = []
        outer_probs = []
        for expert_ind, expert_input in enumerate(routed_inputs):
            expert_input = self.norm(expert_input)
        
            # Extract the outputs for the specific expert
            m = self.model_dim // 2**(expert_ind)
            extracted_input = expert_input[:, :, 0:m]

            # Nested layer 1 projection
            ne_l1_w = self.l1_w[:, 0:m]
            inner_proj = F.linear(extracted_input, ne_l1_w, self.l1_b)

            inner_proj = F.gelu(inner_proj)
            inner_proj = self.drop1(inner_proj)

            # Nested layer 2 projection
            ne_l2_w = self.l2_w[0:m]
            ne_l2_b = self.l2_b[0:m]
            outer_proj = F.linear(inner_proj, ne_l2_w, ne_l2_b)

            outer_proj = self.drop2(outer_proj)

            outer_projs.append(outer_proj)
            outer_probs.append(sorted_router_probs[expert_ind])

        # Residual connection
        final_outs = []
        final_probs = []
        for expert_ind, (expert_output, expert_input, experts_router_prob) in enumerate(zip(outer_projs, routed_inputs, outer_probs)):
            # Pad the tensor so that the residual connection can be applied
            pad_amount = expert_input.shape[-1] - expert_output.shape[-1]
            if pad_amount:
                expert_output = F.pad(expert_output, [0, pad_amount], value=0.)

            scaling_factor = (alpha * experts_router_prob[:, :, expert_ind] + 1)
            scaling_factor = scaling_factor[:, :, None]
            out = expert_input + scaling_factor * expert_output
            final_outs.append(out)
            final_probs.append(experts_router_prob)

        final_outs = torch.cat(final_outs, dim=1)
        final_probs = torch.cat(final_probs, dim=1)

        return final_outs, final_probs


class MoNELayer(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()

        self.nested_expert_mhsa = NEMHSA(model_dim, num_experts)

        self.nested_expert_mlp = NEMLP(model_dim, num_experts)

    def forward(self, x, router_prob, alpha):
        x, router_prob = self.nested_expert_mhsa(x, router_prob)

        x, router_prob = self.nested_expert_mlp(x, router_prob, alpha)

        return x, router_prob

class ClassificationHead(nn.Module):
    def __init__(self, model_dim, num_classes):
        super().__init__()

        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # ViTs class token is stupid
        x = torch.mean(x, dim=1)

        x = self.head(x)

        return x

class MoNE(nn.Module):
    def __init__(self, img_size):
        super().__init__()

        self.model_dim = 256
        patch_size = 4
        self.num_experts = 4
        num_layers = 8
        num_classes = 10
        num_patches = (img_size // patch_size)**2

        self.patch = nn.Conv2d(3, self.model_dim, patch_size, stride=patch_size)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_patches, self.model_dim))

        self.router = nn.Linear(self.model_dim, self.num_experts)

        self.layers = nn.ModuleList([MoNELayer(self.model_dim, self.num_experts) for _ in range(num_layers)])

        self.head = ClassificationHead(self.model_dim, num_classes)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Patchification stem, converts input image batch to batch of embedded tokens
        x = self.patch(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.positional_embeddings

        # Router, computes router probability. Probability of each token going to one of the E number of nested experts (batch, number of tokens, number of nested experts)
        router_prob = F.softmax(self.router(x), dim=-1)

        # Run through each nested expert layer
        for layer in self.layers:
            x, router_prob = layer(x, router_prob, self.alpha)

        # Compute classification scores
        output = self.head(x)

        return output


