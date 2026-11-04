"""Local/gateway LLM-calling primitives shared by label_neurons.py and
predict_neurons_binary.py: model loading and one generate() call. The caller supplies its own
prompts and parses the response; gateway enforces a schema via response_model, local does not."""
from __future__ import annotations


class LocalLLM:
    """A locally-loaded Hugging Face causal LM (e.g. a Gemma checkpoint), generating on device."""

    # Callers must clamp concurrency to 1 for this backend.
    supports_concurrency = False

    def __init__(self, model: str, device: str | None, dtype: str = "bfloat16"):
        import os
        # Must precede CUDA context creation.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map=device or "auto")

    def generate(self, system: str, user: str, max_new_tokens: int = 400) -> str:
        """Raw decoded continuation for one system+user turn (the prompt tokens are not echoed
        back)."""
        return self.generate_batch(system, [user], max_new_tokens)[0]

    def generate_batch(self, system: str, users: list[str], max_new_tokens: int = 400) -> list[str]:
        """One decoded continuation per user turn, in one forward pass, left-padded. A batch is not
        bit-identical to the same prompts run singly, so keep batch_size fixed across compared
        runs.
        """
        convs = [[{"role": "system", "content": system}, {"role": "user", "content": u}]
                 for u in users]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # return_dict=True for a mapping on every transformers version.
        inputs = self.tokenizer.apply_chat_template(
            convs, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True,
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tokenizer.pad_token_id)
        n_prompt = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(o[n_prompt:], skip_special_tokens=True) for o in out]


class GatewayLLM:
    """An OpenAI-compatible chat endpoint via langchain-openai, with structured output enforced at the
    tool-call layer against response_model (a pydantic BaseModel)."""

    supports_concurrency = True  # independent HTTP requests

    def __init__(self, model: str, base_url: str, api_key: str, response_model):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)
        self.llm = llm.with_structured_output(response_model)

    def generate(self, system: str, user: str):
        """A validated response_model instance for one system+user turn."""
        from langchain_core.messages import HumanMessage, SystemMessage
        return self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])


def build_llm(backend: str, model: str, response_model, device: str | None = None,
             dtype: str = "bfloat16", base_url: str | None = None, api_key: str | None = None):
    """LocalLLM or GatewayLLM for --llm-backend {local,gateway}. Gateway's base_url/api_key fall back
    to $LITELLM_PROXY_API_BASE/$LITELLM_PROXY_API_KEY; raises ValueError if a gateway run ends up
    with no base_url."""
    import os
    if backend == "local":
        return LocalLLM(model, device, dtype)
    base_url = base_url or os.environ.get("LITELLM_PROXY_API_BASE")
    api_key = api_key or os.environ.get("LITELLM_PROXY_API_KEY")
    if not base_url:
        raise ValueError("--llm-backend gateway needs --llm-base-url or $LITELLM_PROXY_API_BASE "
                         "(refusing to fall back to the public OpenAI endpoint)")
    if not api_key:
        # The OpenAI client insists on a non-empty string.
        print("no API key given — sending a placeholder (fine for a keyless self-hosted "
              "server; a managed proxy will reject it)")
        api_key = "EMPTY"
    return GatewayLLM(model, base_url, api_key, response_model)
