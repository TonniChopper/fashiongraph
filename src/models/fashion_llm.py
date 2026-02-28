"""Fashion LLM — LLaMA 3.1 8B + LoRA + RAG + CLIP + GNN fusion."""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from src.rag.retriever import FashionRetriever
from src.models.clip_encoder import FashionCLIPEncoder
from src.models.temporal_gnn import TemporalFashionGNN

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class FashionContext:
    """All context required by the LLM to produce an answer.

    Attributes:
        query: User question in natural language.
        rag_chunks: Retrieved documents from ChromaDB.
        trend_scores: Mapping of fashion element to its GNN trend score.
        image_description: Optional visual context produced by CLIP.
    """

    query: str
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    trend_scores: dict[str, float] = field(default_factory=dict)
    image_description: str | None = None


class FashionLLM:
    """LLaMA 3.1 8B with LoRA, RAG, and multimodal context.

    Attributes:
        model: LLaMA model with a LoRA adapter.
        tokenizer: LLaMA tokenizer.
        retriever: ChromaDB semantic retriever.
        clip_encoder: Optional CLIP encoder for visual context.
        gnn: Optional Temporal GNN for trend score forecasting.
    """

    MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def __init__(
        self,
        retriever: FashionRetriever,
        clip_encoder: FashionCLIPEncoder | None = None,
        gnn: TemporalFashionGNN | None = None,
        lora_path: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        """Initializes the FashionLLM.

        Args:
            retriever: Semantic retriever backed by ChromaDB.
            clip_encoder: Optional CLIP encoder for image understanding.
            gnn: Optional Temporal GNN for trend forecasting.
            lora_path: Path to a pretrained LoRA adapter. If ``None``, a
                fresh adapter is initialised.
            load_in_4bit: Whether to quantise the base model to 4-bit.

        Raises:
            RuntimeError: If the base model or LoRA adapter cannot be loaded.
        """
        self.retriever: FashionRetriever = retriever
        self.clip_encoder: FashionCLIPEncoder | None = clip_encoder
        self.gnn: TemporalFashionGNN | None = gnn

        bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        logger.info("Loading base model '%s'...", self.MODEL_ID)
        try:
            self.tokenizer: PreTrainedTokenizerBase = (
                AutoTokenizer.from_pretrained(self.MODEL_ID)
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception as exc:
            logger.error("Failed to load base model '%s': %s", self.MODEL_ID, exc)
            raise RuntimeError(
                f"Could not load base model '{self.MODEL_ID}'"
            ) from exc

        if lora_path:
            try:
                self.model: PeftModel = PeftModel.from_pretrained(
                    base_model, lora_path
                )
                logger.info("LoRA adapter loaded from '%s'.", lora_path)
            except Exception as exc:
                logger.error(
                    "Failed to load LoRA adapter from '%s': %s", lora_path, exc
                )
                raise RuntimeError(
                    f"Could not load LoRA adapter from '{lora_path}'"
                ) from exc
        else:
            lora_config: LoraConfig = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
            logger.info("Fresh LoRA adapter initialised.")

        self.model.eval()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_prompt(self, ctx: FashionContext) -> str:
        """Builds the full LLaMA chat prompt from a ``FashionContext``.

        The prompt follows the LLaMA 3.1 Instruct template with system,
        user, and assistant header tokens.

        Args:
            ctx: Aggregated context (RAG chunks, trend scores, visual info).

        Returns:
            Formatted prompt string ready for tokenisation.
        """
        rag_text: str = "\n".join(
            f"- [{r['metadata'].get('element', 'unknown')} "
            f"{r['metadata'].get('season', '')}]: {r['document']}"
            for r in ctx.rag_chunks
        )

        trend_text: str = "\n".join(
            f"- {elem}: trend_score={score:.2f}"
            for elem, score in ctx.trend_scores.items()
        )

        image_section: str = (
            f"\nVisual context: {ctx.image_description}"
            if ctx.image_description else ""
        )

        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are FashionGraph AI — an expert fashion trend analyst. "
            f"Use the provided trend data and cultural context to give "
            f"precise, actionable insights.<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"## RAG Context (expert annotations)\n{rag_text}\n\n"
            f"## Trend Scores (GNN forecast)\n{trend_text}"
            f"{image_section}\n\n"
            f"## Question\n{ctx.query}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        images: torch.Tensor | None = None,
        trend_scores: dict[str, float] | None = None,
        n_rag: int = 5,
        max_new_tokens: int = 512,
    ) -> str:
        """Generates a fashion-insight answer for the given query.

        Retrieves RAG context, optionally computes visual similarity via
        CLIP, and runs LLaMA generation.

        Args:
            query: Natural-language question.
            images: Optional batch of preprocessed images ``(B, C, H, W)``.
            trend_scores: Optional dict of element → GNN trend scores.
            n_rag: Number of RAG chunks to retrieve.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The model's generated answer as a string.

        Raises:
            RuntimeError: If RAG retrieval or model generation fails.
        """
        # --- RAG retrieval ---
        try:
            rag_chunks: list[dict[str, Any]] = self.retriever.retrieve(
                query, n_results=n_rag
            )
        except Exception as exc:
            logger.error("RAG retrieval failed for query '%s': %s", query, exc)
            raise RuntimeError("RAG retrieval failed.") from exc

        # --- CLIP visual similarity (optional) ---
        image_description: str | None = None
        if images is not None and self.clip_encoder is not None:
            try:
                txt_emb: torch.Tensor = self.clip_encoder.encode_text([query])
                img_emb: torch.Tensor = self.clip_encoder.encode_image(images)
                sim: torch.Tensor = (img_emb @ txt_emb.T).squeeze()
                if sim.dim() == 0:
                    image_description = f"Visual-text similarity: {sim.item():.3f}"
                else:
                    scores: list[str] = [f"{s:.3f}" for s in sim.tolist()]
                    image_description = f"Visual-text similarities: {scores}"
            except Exception as exc:
                logger.warning("CLIP encoding failed, skipping visual context: %s", exc)

        ctx: FashionContext = FashionContext(
            query=query,
            rag_chunks=rag_chunks,
            trend_scores=trend_scores or {},
            image_description=image_description,
        )

        prompt: str = self._build_prompt(ctx)
        logger.debug("Prompt length: %d characters.", len(prompt))

        try:
            inputs: dict[str, torch.Tensor] = self.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids: torch.Tensor = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as exc:
            logger.error("Model generation failed: %s", exc)
            raise RuntimeError("Model generation failed.") from exc

        new_tokens: torch.Tensor = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer: str = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        logger.info("Generated answer with %d tokens.", len(new_tokens))
        return answer

    def save_lora(self, path: str) -> None:
        """Saves the LoRA adapter and tokenizer to *path*.

        Args:
            path: Directory to write the adapter weights and tokenizer files.

        Raises:
            OSError: If the files cannot be written.
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info("LoRA adapter and tokenizer saved to '%s'.", path)
        except Exception as exc:
            logger.error("Failed to save LoRA adapter to '%s': %s", path, exc)
            raise OSError(f"Could not save LoRA adapter to '{path}'") from exc

    def load_lora(self, path: str) -> None:
        """Loads a LoRA adapter from *path* into the current model.

        Args:
            path: Directory containing a previously saved LoRA adapter.

        Raises:
            RuntimeError: If the adapter cannot be loaded.
        """
        try:
            self.model = PeftModel.from_pretrained(
                self.model.base_model.model, path
            )
            self.model.eval()
            logger.info("LoRA adapter loaded from '%s'.", path)
        except Exception as exc:
            logger.error("Failed to load LoRA adapter from '%s': %s", path, exc)
            raise RuntimeError(
                f"Could not load LoRA adapter from '{path}'"
            ) from exc
