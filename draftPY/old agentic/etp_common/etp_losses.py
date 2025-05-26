import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_reconstruction_loss(
    reconstructed_embeddings: torch.Tensor, 
    original_embeddings: torch.Tensor, 
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Calculates the reconstruction loss between original and reconstructed embeddings.

    Args:
        reconstructed_embeddings: Tensor of embeddings output by the ETP Sphere's decoder.
        original_embeddings: Tensor of the original source embeddings.
        loss_type: String, either "mse" (for F.mse_loss) or "cosine" 
                   (for 1 - F.cosine_similarity(reconstructed, original).mean()).

    Returns:
        The calculated reconstruction loss.
    """
    if reconstructed_embeddings.shape != original_embeddings.shape:
        logging.warning(f"Shape mismatch in reconstruction loss: "
                        f"Reconstructed shape {reconstructed_embeddings.shape}, "
                        f"Original shape {original_embeddings.shape}. This might lead to errors.")

    if loss_type == "mse":
        return F.mse_loss(reconstructed_embeddings, original_embeddings)
    elif loss_type == "cosine":
        # F.cosine_similarity computes similarity along a dimension.
        # For batch of embeddings (N, D), we want (N) similarities.
        if reconstructed_embeddings.ndim == 1: # Single vector case
             cosine_sim = F.cosine_similarity(reconstructed_embeddings.unsqueeze(0), original_embeddings.unsqueeze(0), dim=1)
        else: # Batch of vectors
             cosine_sim = F.cosine_similarity(reconstructed_embeddings, original_embeddings, dim=1)
        return (1 - cosine_sim).mean() # Mean of (1 - similarity) for each item in batch
    else:
        logging.error(f"Unsupported reconstruction loss type: {loss_type}. Defaulting to MSE.")
        return F.mse_loss(reconstructed_embeddings, original_embeddings)


def _pairwise_similarity(batch: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Helper to compute pairwise similarity matrix."""
    if batch.ndim != 2:
        raise ValueError(f"Input batch must be 2D (N, D), but got shape {batch.shape}")
    N, D = batch.shape
    if N == 0:
        return torch.empty((0, 0), device=batch.device, dtype=batch.dtype)
    if N == 1: # Handle single item batch to avoid issues with pairwise comparison logic
        if metric == "cosine":
            return torch.ones((1,1), device=batch.device, dtype=batch.dtype) # Self-similarity is 1
        elif metric == "dot":
            return (batch @ batch.T)


    if metric == "cosine":
        # Normalize rows for cosine similarity
        batch_normalized = F.normalize(batch, p=2, dim=1)
        return batch_normalized @ batch_normalized.T
    elif metric == "dot":
        return batch @ batch.T
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")

def _normalize_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Normalizes a matrix to zero mean and unit variance."""
    if matrix.numel() == 0 : # Handle empty matrix
        return matrix
    if matrix.numel() == 1: # Handle single element matrix (std would be 0)
        return matrix - matrix.mean() # Center it, variance is 0

    mean = matrix.mean()
    std = matrix.std()
    if std < 1e-8: # Avoid division by zero or very small std
        return matrix - mean # Just center it
    return (matrix - mean) / std


def calculate_vector_space_preservation_loss(
    source_batch: torch.Tensor, 
    wubu_latent_batch: torch.Tensor, 
    similarity_metric: str = "cosine", 
    normalize_similarity_matrices: bool = True
) -> torch.Tensor:
    """
    Calculates the Vector Space Preservation (VSP) loss.

    Args:
        source_batch: Batch of source embeddings (e.g., shape (N, D_source)).
        wubu_latent_batch: Batch of corresponding WuBu latent tangent vectors (e.g., shape (N, D_wubu_latent)).
        similarity_metric: String, "cosine" or "dot".
        normalize_similarity_matrices: If True, normalize similarity matrices before MSE.

    Returns:
        The VSP loss.
    """
    if source_batch.shape[0] != wubu_latent_batch.shape[0]:
        logging.error(f"Batch size mismatch in VSP loss: "
                      f"Source batch N={source_batch.shape[0]}, "
                      f"WuBu latent batch N={wubu_latent_batch.shape[0]}. Cannot compute VSP loss.")
        return torch.tensor(0.0, device=source_batch.device, requires_grad=True) # Return a scalar zero tensor

    if source_batch.shape[0] <= 1: # VSP loss is not well-defined for batch size < 2
        # logging.info("VSP loss not computed for batch size <= 1.")
        return torch.tensor(0.0, device=source_batch.device, requires_grad=True)


    sim_source = _pairwise_similarity(source_batch, metric=similarity_metric)
    sim_wubu = _pairwise_similarity(wubu_latent_batch, metric=similarity_metric)

    if normalize_similarity_matrices:
        sim_source = _normalize_matrix(sim_source)
        sim_wubu = _normalize_matrix(sim_wubu)

    # The roadmap suggests sum_{i != j}. MSE over the full matrix includes the diagonal.
    # For simplicity and as per instruction ("full matrix MSE is simpler"), we use full MSE.
    # If i != j is strictly needed, one would create a mask.
    return F.mse_loss(sim_source, sim_wubu)


def calculate_adversarial_latent_alignment_loss_discriminator(
    d_output_source_A: torch.Tensor, 
    d_output_source_B_detached: torch.Tensor, 
    gan_loss_type: str = "bce"
) -> torch.Tensor:
    """
    Calculates GAN loss for the discriminator.

    Args:
        d_output_source_A: Logits from D for latent vectors from Corpus_A ("real").
        d_output_source_B_detached: Logits from D for latent vectors from Corpus_B ("fake", detached).
        gan_loss_type: String, e.g., "bce".

    Returns:
        The discriminator's GAN loss.
    """
    if gan_loss_type == "bce":
        loss_real = F.binary_cross_entropy_with_logits(
            d_output_source_A, torch.ones_like(d_output_source_A)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            d_output_source_B_detached, torch.zeros_like(d_output_source_B_detached)
        )
        return (loss_real + loss_fake) / 2
    else:
        logging.error(f"Unsupported GAN loss type for discriminator: {gan_loss_type}. Defaulting to BCE.")
        # Fallback to BCE logic
        loss_real = F.binary_cross_entropy_with_logits(
            d_output_source_A, torch.ones_like(d_output_source_A)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            d_output_source_B_detached, torch.zeros_like(d_output_source_B_detached)
        )
        return (loss_real + loss_fake) / 2


def calculate_adversarial_latent_alignment_loss_generator(
    d_output_source_A_for_generator: torch.Tensor,
    d_output_source_B_for_generator: torch.Tensor,
    gan_loss_type: str = "bce"
) -> torch.Tensor:
    """
    Calculates GAN loss for the generator (ETP Sphere model).
    Revised based on ETP Roadmap (Section 4.3.3.i) - generator wants latents from *both*
    corpus A and corpus B to be classified as real by the discriminator.

    Args:
        d_output_source_A_for_generator: Logits from D for latent vectors from Corpus_A (not detached).
        d_output_source_B_for_generator: Logits from D for latent vectors from Corpus_B (not detached).
        gan_loss_type: String, e.g., "bce".

    Returns:
        The generator's GAN loss.
    """
    if gan_loss_type == "bce":
        # Generator wants discriminator to classify both sets of latents as "real"
        loss_A = F.binary_cross_entropy_with_logits(
            d_output_source_A_for_generator, torch.ones_like(d_output_source_A_for_generator)
        )
        loss_B = F.binary_cross_entropy_with_logits(
            d_output_source_B_for_generator, torch.ones_like(d_output_source_B_for_generator)
        )
        return (loss_A + loss_B) / 2
    else:
        logging.error(f"Unsupported GAN loss type for generator: {gan_loss_type}. Defaulting to BCE.")
        # Fallback to BCE logic
        loss_A = F.binary_cross_entropy_with_logits(
            d_output_source_A_for_generator, torch.ones_like(d_output_source_A_for_generator)
        )
        loss_B = F.binary_cross_entropy_with_logits(
            d_output_source_B_for_generator, torch.ones_like(d_output_source_B_for_generator)
        )
        return (loss_A + loss_B) / 2


if __name__ == '__main__':
    logging.info("Starting example usage of ETP loss functions.")
    
    batch_size = 16
    dim_source = 128
    dim_latent = 64

    # --- Test Reconstruction Loss ---
    logging.info("\n--- Testing Reconstruction Loss ---")
    original = torch.randn(batch_size, dim_source)
    reconstructed_good = original + torch.randn(batch_size, dim_source) * 0.1 # Close reconstruction
    reconstructed_bad = torch.randn(batch_size, dim_source) # Poor reconstruction
    
    loss_mse_good = calculate_reconstruction_loss(reconstructed_good, original, "mse")
    loss_mse_bad = calculate_reconstruction_loss(reconstructed_bad, original, "mse")
    logging.info(f"MSE Loss (good recon): {loss_mse_good.item()}")
    logging.info(f"MSE Loss (bad recon): {loss_mse_bad.item()}")
    assert loss_mse_good < loss_mse_bad

    loss_cosine_good = calculate_reconstruction_loss(reconstructed_good, original, "cosine")
    loss_cosine_bad = calculate_reconstruction_loss(reconstructed_bad, original, "cosine")
    logging.info(f"Cosine Loss (good recon): {loss_cosine_good.item()}")
    logging.info(f"Cosine Loss (bad recon): {loss_cosine_bad.item()}")
    assert loss_cosine_good < loss_cosine_bad
    
    # Test single vector case for cosine
    original_single = torch.randn(dim_source)
    reconstructed_single_good = original_single + torch.randn(dim_source) * 0.1
    loss_cosine_single = calculate_reconstruction_loss(reconstructed_single_good, original_single, "cosine")
    logging.info(f"Cosine Loss (single vector): {loss_cosine_single.item()}")
    assert isinstance(loss_cosine_single.item(), float)


    # --- Test Vector Space Preservation Loss ---
    logging.info("\n--- Testing Vector Space Preservation Loss ---")
    source_emb = torch.randn(batch_size, dim_source)
    # Case 1: Latent space perfectly preserves relative similarities (scaled version)
    latent_perfect = source_emb[:, :dim_latent] * 2.0 + 5.0 # Linear transform, should have high similarity of similarity matrices
    # Case 2: Latent space is random
    latent_random = torch.randn(batch_size, dim_latent)

    vsp_loss_perfect_cosine = calculate_vector_space_preservation_loss(source_emb, latent_perfect, "cosine", True)
    vsp_loss_random_cosine = calculate_vector_space_preservation_loss(source_emb, latent_random, "cosine", True)
    logging.info(f"VSP Loss (perfect, cosine, normalized): {vsp_loss_perfect_cosine.item()}")
    logging.info(f"VSP Loss (random, cosine, normalized): {vsp_loss_random_cosine.item()}")
    assert vsp_loss_perfect_cosine < vsp_loss_random_cosine

    vsp_loss_perfect_dot = calculate_vector_space_preservation_loss(source_emb, latent_perfect, "dot", True)
    vsp_loss_random_dot = calculate_vector_space_preservation_loss(source_emb, latent_random, "dot", True)
    logging.info(f"VSP Loss (perfect, dot, normalized): {vsp_loss_perfect_dot.item()}")
    logging.info(f"VSP Loss (random, dot, normalized): {vsp_loss_random_dot.item()}")
    # Dot product similarity is sensitive to magnitudes, so linear transform might not be as "perfect" as for cosine.
    # Still expect random to be worse.
    assert vsp_loss_perfect_dot < vsp_loss_random_dot 

    vsp_loss_perfect_cosine_unnorm = calculate_vector_space_preservation_loss(source_emb, latent_perfect, "cosine", False)
    logging.info(f"VSP Loss (perfect, cosine, unnormalized): {vsp_loss_perfect_cosine_unnorm.item()}")
    
    # Test VSP with batch size 1 and 0
    source_emb_bs1 = torch.randn(1, dim_source)
    latent_bs1 = torch.randn(1, dim_latent)
    vsp_loss_bs1 = calculate_vector_space_preservation_loss(source_emb_bs1, latent_bs1)
    logging.info(f"VSP Loss (batch_size=1): {vsp_loss_bs1.item()}")
    assert vsp_loss_bs1.item() == 0.0

    source_emb_bs0 = torch.randn(0, dim_source)
    latent_bs0 = torch.randn(0, dim_latent)
    vsp_loss_bs0 = calculate_vector_space_preservation_loss(source_emb_bs0, latent_bs0)
    logging.info(f"VSP Loss (batch_size=0): {vsp_loss_bs0.item()}")
    assert vsp_loss_bs0.item() == 0.0


    # --- Test Adversarial Latent Alignment Losses ---
    logging.info("\n--- Testing Adversarial Latent Alignment Losses ---")
    # Discriminator outputs (logits)
    # Scenario 1: Discriminator is doing well
    d_outs_A_good_D = torch.randn(batch_size, 1) + 2.0  # High logits for "real" (source A)
    d_outs_B_good_D_detached = torch.randn(batch_size, 1) - 2.0 # Low logits for "fake" (source B, detached)
    
    # Scenario 2: Discriminator is confused / Generator is doing well
    d_outs_A_bad_D = torch.randn(batch_size, 1) - 2.0 # Low logits for "real" (source A)
    d_outs_B_bad_D = torch.randn(batch_size, 1) + 2.0  # High logits for "fake" (source B)

    # Test Discriminator Loss
    loss_D_good = calculate_adversarial_latent_alignment_loss_discriminator(
        d_outs_A_good_D, d_outs_B_good_D_detached, "bce"
    )
    loss_D_bad = calculate_adversarial_latent_alignment_loss_discriminator(
        d_outs_A_bad_D, d_outs_B_good_D, "bce" # D still thinks B is fake here
    )
    logging.info(f"Discriminator Loss (D is good): {loss_D_good.item()}")
    logging.info(f"Discriminator Loss (D is confused about A): {loss_D_bad.item()}")
    # If D is good, its loss should be low. If D is bad (confused), its loss should be higher.
    assert loss_D_good < loss_D_bad


    # Test Generator Loss (using the revised function signature)
    # Generator wants D to output high logits (real) for both A and B latents.
    # d_output_source_A_for_generator and d_output_source_B_for_generator are outputs from D
    # when processing latents that the generator *wants* to be seen as real.
    
    # Scenario 1: Generator is failing (D sees its outputs as fake)
    d_A_for_G_fail = torch.randn(batch_size, 1) - 2.0 # D outputs low for A's latents
    d_B_for_G_fail = torch.randn(batch_size, 1) - 2.0 # D outputs low for B's latents (G wants high)
    
    # Scenario 2: Generator is succeeding (D sees its outputs as real)
    d_A_for_G_succeed = torch.randn(batch_size, 1) + 2.0 # D outputs high for A's latents
    d_B_for_G_succeed = torch.randn(batch_size, 1) + 2.0 # D outputs high for B's latents (G wants high)

    loss_G_fail = calculate_adversarial_latent_alignment_loss_generator(
        d_A_for_G_fail, d_B_for_G_fail, "bce"
    )
    loss_G_succeed = calculate_adversarial_latent_alignment_loss_generator(
        d_A_for_G_succeed, d_B_for_G_succeed, "bce"
    )
    logging.info(f"Generator Loss (G is failing): {loss_G_fail.item()}")
    logging.info(f"Generator Loss (G is succeeding): {loss_G_succeed.item()}")
    # If G is succeeding (D thinks its fakes are real), G's loss should be low.
    # If G is failing (D spots its fakes), G's loss should be high.
    assert loss_G_succeed < loss_G_fail
    
    logging.info("\nExample usage of ETP loss functions finished.")
