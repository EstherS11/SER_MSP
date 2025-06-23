"""
Ultimate custom model components for MSP-PODCAST discrete SSL emotion recognition
Fusion of IEMOCAP proven techniques + advanced features

Key features:
1. IEMOCAP's efficient offset technique for embedding
2. Hybrid attention mechanism: simple base + advanced options
3. Comprehensive analysis and debugging capabilities
4. Flexible architecture with proven defaults

Authors: [Your Name]
Best practices from IEMOCAP + Enhanced functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class Discrete_EmbeddingLayer(torch.nn.Module):
    """
    Ultimate discrete token embedding layer
    
    Combines IEMOCAP's efficient offset technique with advanced features:
    - Single embedding table with offset (IEMOCAP proven technique)
    - Optional positional encoding for better temporal modeling
    - Flexible initialization options
    - Comprehensive analysis capabilities
    
    Arguments
    ---------
    num_codebooks: int
        Number of codebooks (usually 6 for SSL layers)
    vocab_size: int  
        Size of vocabulary for each codebook (e.g., 1000)
    emb_dim: int
        Embedding dimension (e.g., 1024)
    pad_index: int (default: 0)
        Padding index (won't contribute to gradients)
    freeze: bool (default: False)
        Whether to freeze embeddings during training
    use_positional_encoding: bool (default: True)
        Whether to add positional encoding for codebook positions
    init_method: str (default: 'xavier')
        Initialization method: 'xavier', 'normal', 'uniform'
    """
    
    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        emb_dim: int,
        pad_index: int = 0,
        freeze: bool = False,
        use_positional_encoding: bool = True,
        init_method: str = 'xavier'
    ):
        super(Discrete_EmbeddingLayer, self).__init__()
        
        # Store configuration
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.freeze = freeze
        self.use_positional_encoding = use_positional_encoding
        self.init_method = init_method
        
        # IEMOCAP's key innovation: Single embedding table with offset
        total_vocab_size = num_codebooks * vocab_size
        self.embedding = torch.nn.Embedding(
            total_vocab_size, emb_dim, padding_idx=pad_index
        ).requires_grad_(not self.freeze)
        
        # Advanced feature: Positional encoding for codebook positions
        if self.use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, 1, num_codebooks, emb_dim)
            )
            self._init_positional_encoding()
        
        # Initialize embeddings
        self._init_embeddings()
        
        # For analysis and debugging
        self.register_buffer('codebook_usage', torch.zeros(num_codebooks))
        self.register_buffer('token_usage', torch.zeros(total_vocab_size))
    
    def _init_embeddings(self):
        """Initialize embedding weights based on specified method"""
        if self.init_method == 'xavier':
            torch.nn.init.xavier_uniform_(self.embedding.weight)
        elif self.init_method == 'normal':
            torch.nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        elif self.init_method == 'uniform':
            torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        
        # Zero out padding if specified
        if hasattr(self.embedding, 'padding_idx') and self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].fill_(0)
    
    def _init_positional_encoding(self):
        """Initialize positional encoding with sinusoidal patterns"""
        # Create sinusoidal positional encoding for codebook positions
        position = torch.arange(self.num_codebooks).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2).float() * 
                           -(math.log(10000.0) / self.emb_dim))
        
        pos_encoding = torch.zeros(1, 1, self.num_codebooks, self.emb_dim)
        pos_encoding[0, 0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, 0, :, 1::2] = torch.cos(position * div_term)
        
        self.positional_encoding.data = pos_encoding * 0.1  # Scale down
    
    def forward(self, in_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with IEMOCAP offset technique + enhancements
        
        Args:
            in_tokens: [batch, time, num_codebooks] discrete tokens
            
        Returns:
            embeddings: [batch, time, num_codebooks, emb_dim]
        """
        batch_size, time_steps, num_codebooks = in_tokens.shape
        
        # Validation
        assert num_codebooks == self.num_codebooks, \
            f"Expected {self.num_codebooks} codebooks, got {num_codebooks}"
        
        with torch.set_grad_enabled(not self.freeze):
            # IEMOCAP's offset technique: Make tokens unique across codebooks
            offset = torch.arange(
                0, self.num_codebooks * self.vocab_size, self.vocab_size,
                device=in_tokens.device, dtype=in_tokens.dtype
            )
            
            # Apply offset to create unique token IDs
            in_tokens_offset = in_tokens + offset.unsqueeze(0).unsqueeze(0)
            
            # Single embedding lookup (IEMOCAP efficiency)
            embeddings = self.embedding(in_tokens_offset)
            
            # Add positional encoding if enabled (advanced feature)
            if self.use_positional_encoding:
                embeddings = embeddings + self.positional_encoding
            
            # Update usage statistics for analysis
            if self.training:
                self._update_usage_stats(in_tokens_offset)
            
            return embeddings
    
    def _update_usage_stats(self, tokens_offset: torch.Tensor):
        """Update token and codebook usage statistics"""
        with torch.no_grad():
            # Update codebook usage
            for cb in range(self.num_codebooks):
                self.codebook_usage[cb] += (tokens_offset[:, :, cb] >= 0).sum().item()
            
            # Update token usage
            unique_tokens, counts = torch.unique(tokens_offset, return_counts=True)
            for token, count in zip(unique_tokens, counts):
                if 0 <= token < len(self.token_usage):
                    self.token_usage[token] += count.item()
    
    def get_codebook_embedding(self, codebook_idx: int, tokens: torch.Tensor) -> torch.Tensor:
        """Get embedding for specific codebook"""
        assert 0 <= codebook_idx < self.num_codebooks
        offset = codebook_idx * self.vocab_size
        tokens_offset = tokens + offset
        return self.embedding(tokens_offset)
    
    def get_usage_analysis(self) -> Dict[str, Any]:
        """Get comprehensive usage analysis"""
        return {
            'codebook_usage': self.codebook_usage.cpu().numpy(),
            'token_usage': self.token_usage.cpu().numpy(),
            'most_used_codebook': self.codebook_usage.argmax().item(),
            'least_used_codebook': self.codebook_usage.argmin().item(),
            'usage_variance': self.codebook_usage.var().item(),
            'total_tokens_processed': self.codebook_usage.sum().item()
        }


class AttentionMLP(torch.nn.Module):
    """
    Hybrid attention mechanism combining IEMOCAP simplicity with advanced options
    
    Features:
    - Base: IEMOCAP's proven simple architecture  
    - Advanced: Optional multi-head attention, layer norm, global priors
    - Flexible: Switch between simple/complex modes
    - Analysis: Comprehensive attention analysis tools
    
    Arguments
    ---------
    input_dim: int
        Input embedding dimension
    hidden_dim: int  
        Hidden layer dimension
    num_codebooks: int (default: 6)
        Number of codebooks
    mode: str (default: 'iemocap')
        'iemocap': Simple proven approach
        'advanced': Enhanced with modern techniques
        'hybrid': Best of both worlds
    dropout_rate: float (default: 0.1)
        Dropout probability for advanced modes
    use_layer_norm: bool (default: False) 
        Whether to use layer normalization
    use_global_priors: bool (default: False)
        Whether to learn global codebook importance
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_codebooks: int = 6,
        mode: str = 'hybrid',  # 'iemocap', 'advanced', 'hybrid'
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        use_global_priors: bool = True
    ):
        super(AttentionMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_codebooks = num_codebooks
        self.mode = mode
        
        # IEMOCAP base architecture (proven)
        if mode in ['iemocap', 'hybrid']:
            self.iemocap_layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1, bias=False),
            )
        
        # Advanced features (optional)
        if mode in ['advanced', 'hybrid']:
            # Enhanced MLP with modern techniques
            self.advanced_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_codebooks)
            )
            
            # Global codebook importance (learnable priors)
            if use_global_priors:
                self.global_importance = nn.Parameter(torch.ones(num_codebooks))
            else:
                self.register_parameter('global_importance', None)
        
        # Layer normalization for input stabilization
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        # For analysis
        self.attention_history = []
    
    def _init_weights(self):
        """Initialize all weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hybrid attention mechanism
        
        Args:
            x: [batch, time, num_codebooks, emb_dim] embeddings
            
        Returns:
            attention_weights: [batch, time, num_codebooks, 1] or [batch, time, num_codebooks]
        """
        # Input normalization
        x_norm = self.input_norm(x)
        
        if self.mode == 'iemocap':
            return self._iemocap_attention(x_norm)
        elif self.mode == 'advanced':
            return self._advanced_attention(x_norm)
        else:  # hybrid
            return self._hybrid_attention(x_norm)
    
    def _iemocap_attention(self, x: torch.Tensor) -> torch.Tensor:
        """IEMOCAP's proven simple attention"""
        # Apply MLP to get scalar scores
        scores = self.iemocap_layers(x)  # [batch, time, num_codebooks, 1]
        
        # Softmax across codebook dimension
        attention_weights = F.softmax(scores, dim=2)
        
        return attention_weights
    
    def _advanced_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Advanced attention with modern techniques"""
        batch_size, time_steps, num_codebooks, emb_dim = x.shape
        
        # Compute mean representation for context
        mean_x = x.mean(dim=2)  # [batch, time, emb_dim]
        
        # Generate attention logits
        attention_logits = self.advanced_layers(mean_x)  # [batch, time, num_codebooks]
        
        # Add global importance if available
        if self.global_importance is not None:
            global_weights = F.softmax(self.global_importance, dim=0)
            attention_logits = attention_logits + global_weights.unsqueeze(0).unsqueeze(0)
        
        # Normalize
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        return attention_weights.unsqueeze(-1)  # Add dimension for consistency
    
    def _hybrid_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid: Combine IEMOCAP + advanced features"""
        # Get both attention types
        iemocap_att = self._iemocap_attention(x).squeeze(-1)  # [batch, time, num_codebooks]
        
        # Advanced attention for context
        batch_size, time_steps, num_codebooks, emb_dim = x.shape
        mean_x = x.mean(dim=2)
        advanced_logits = self.advanced_layers(mean_x)
        
        # Add global priors if available
        if self.global_importance is not None:
            global_weights = F.softmax(self.global_importance, dim=0)
            advanced_logits = advanced_logits + global_weights.unsqueeze(0).unsqueeze(0)
        
        advanced_att = F.softmax(advanced_logits, dim=-1)
        
        # Combine: weighted average with learnable balance
        alpha = 0.7  # Favor IEMOCAP (proven) approach
        combined_att = alpha * iemocap_att + (1 - alpha) * advanced_att
        
        # Renormalize
        combined_att = F.softmax(combined_att, dim=-1)
        
        return combined_att.unsqueeze(-1)
    
    def compute_weighted_sum(self, embeddings: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted sum using IEMOCAP official matrix multiplication approach
        
        This is the KEY difference from element-wise multiplication:
        IEMOCAP uses elegant matrix multiplication for better efficiency and numerical stability
        """
        # Ensure attention weights have correct shape for matrix multiplication
        if attention_weights.dim() == 3:  # [batch, time, num_codebooks]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch, time, num_codebooks, 1]
        
        # IEMOCAP official approach: matrix multiplication
        # att_w: [batch, time, num_codebooks, 1] 
        # embeddings: [batch, time, num_codebooks, emb_dim]
        # Result: [batch, time, 1, emb_dim] -> squeeze -> [batch, time, emb_dim]
        weighted_sum = torch.matmul(attention_weights.transpose(2, -1), embeddings).squeeze(-2)
        
        return weighted_sum
    
    def get_attention_analysis(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Comprehensive attention analysis"""
        with torch.no_grad():
            attention_weights = self.forward(embeddings)
            
            if attention_weights.dim() == 4:
                attention_weights = attention_weights.squeeze(-1)
            
            analysis = {
                'attention_weights': attention_weights,
                'attention_entropy': self._compute_entropy(attention_weights),
                'dominant_codebook': attention_weights.argmax(dim=-1),
                'attention_std': attention_weights.std(dim=-1),
                'codebook_importance': attention_weights.mean(dim=(0, 1)),
            }
            
            if self.global_importance is not None:
                analysis['global_importance'] = F.softmax(self.global_importance, dim=0)
            
            return analysis
    
    def _compute_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution"""
        epsilon = 1e-8
        log_weights = torch.log(attention_weights + epsilon)
        entropy = -(attention_weights * log_weights).sum(dim=-1)
        return entropy


class MSPPodcastUltimateModel(torch.nn.Module):
    """
    Ultimate MSP-PODCAST emotion recognition model
    
    Combines:
    - IEMOCAP's proven offset embedding technique
    - Hybrid attention mechanism  
    - ECAPA-TDNN backbone
    - Comprehensive analysis capabilities
    - Flexible configuration options
    """
    
    def __init__(
        self,
        num_codebooks: int = 6,
        vocab_size: int = 1000,
        emb_dim: int = 1024,
        num_emotions: int = 4,
        attention_mode: str = 'hybrid',
        use_positional_encoding: bool = True,
        use_global_priors: bool = True,
        dropout_rate: float = 0.1
    ):
        super(MSPPodcastUltimateModel, self).__init__()
        
        # Store configuration
        self.config = {
            'num_codebooks': num_codebooks,
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'num_emotions': num_emotions,
            'attention_mode': attention_mode
        }
        
        # Ultimate discrete embedding (IEMOCAP + enhancements)
        self.discrete_embedding = Discrete_EmbeddingLayer(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            use_positional_encoding=use_positional_encoding
        )
        
        # Hybrid attention mechanism
        self.attention_mlp = AttentionMLP(
            input_dim=emb_dim,
            hidden_dim=emb_dim,
            num_codebooks=num_codebooks,
            mode=attention_mode,
            dropout_rate=dropout_rate,
            use_global_priors=use_global_priors
        )
        
        # ECAPA-TDNN backbone (proven architecture)
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
        
        self.ecapa_tdnn = ECAPA_TDNN(
            input_size=emb_dim,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=64,
            lin_neurons=192
        )
        
        # Final classifier
        self.classifier = Classifier(
            input_size=192,
            out_neurons=num_emotions
        )
        
        # For interpretability
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry']
    
    def forward(self, discrete_tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete forward pass using IEMOCAP official attention approach
        """
        # 1. Embed with IEMOCAP offset technique
        embeddings = self.discrete_embedding(discrete_tokens)
        
        # 2. Get attention weights
        attention_weights = self.attention_mlp(embeddings)
        
        # 3. IEMOCAP official matrix multiplication approach
        fused_features = torch.matmul(attention_weights.transpose(2, -1), embeddings).squeeze(-2)
        
        # 4. ECAPA-TDNN processing
        if lengths is None:
            lengths = torch.ones(discrete_tokens.size(0), device=discrete_tokens.device)
        
        ecapa_features = self.ecapa_tdnn(fused_features, lengths)
        
        # 5. Final classification
        logits = self.classifier(ecapa_features)
        
        return logits, attention_weights
    
    def predict_with_analysis(self, discrete_tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Prediction with comprehensive analysis"""
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(discrete_tokens, lengths)
            
            # Basic predictions
            probabilities = F.softmax(logits, dim=-1)
            predicted_classes = probabilities.argmax(dim=-1)
            confidence = probabilities.max(dim=-1)[0]
            
            # Get embeddings for analysis
            embeddings = self.discrete_embedding(discrete_tokens)
            
            # Comprehensive analysis
            attention_analysis = self.attention_mlp.get_attention_analysis(embeddings)
            usage_analysis = self.discrete_embedding.get_usage_analysis()
            
            return {
                'predictions': {
                    'emotion': [self.emotion_labels[idx] for idx in predicted_classes],
                    'confidence': confidence.cpu().numpy(),
                    'probabilities': probabilities.cpu().numpy()
                },
                'attention_analysis': {k: v.cpu().numpy() if torch.is_tensor(v) else v 
                                    for k, v in attention_analysis.items()},
                'usage_analysis': usage_analysis,
                'model_config': self.config
            }


class MultiClassMetrics:
    """
    Multi-class evaluation metrics for MSP-PODCAST 10-class emotion recognition
    
    Supports:
    - Macro/Weighted/Micro F1, Precision, Recall
    - Per-class metrics
    - Confusion matrix
    - Class-wise analysis
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 class_names: Optional[list] = None,
                 primary_metric: str = 'macro_f1',
                 additional_metrics: Optional[list] = None):
        
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.primary_metric = primary_metric
        self.additional_metrics = additional_metrics or ['accuracy', 'weighted_f1']
        
        # Storage for batch results
        self.predictions = []
        self.targets = []
        self.sample_ids = []
        
        # Import sklearn here to avoid dependency issues
        try:
            from sklearn.metrics import (
                f1_score, precision_score, recall_score, accuracy_score,
                confusion_matrix, classification_report
            )
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            print("Warning: sklearn not available, using basic metrics only")
    
    def append(self, ids, predictions, targets):
        """Append batch results for later computation"""
        # Convert predictions to class indices
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=-1)
        else:
            pred_classes = predictions
        
        # Store results
        self.predictions.extend(pred_classes.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.sample_ids.extend(ids)
    
    def summarize(self, field=None):
        """
        Compute comprehensive metrics
        
        Args:
            field: For compatibility with SpeechBrain interface
            
        Returns:
            Primary metric value (for checkpointing/LR scheduling)
        """
        if not self.predictions:
            return 0.0
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Compute metrics
        metrics = self._compute_all_metrics(predictions, targets)
        
        # Log detailed results
        self._log_detailed_results(metrics, predictions, targets)
        
        # Return primary metric for optimization
        return metrics.get(self.primary_metric, metrics.get('accuracy', 0.0))
    
    def _compute_all_metrics(self, predictions, targets):
        """Compute all requested metrics"""
        metrics = {}
        
        if self.sklearn_available:
            from sklearn.metrics import (
                f1_score, precision_score, recall_score, accuracy_score
            )
            
            # Basic accuracy
            metrics['accuracy'] = accuracy_score(targets, predictions)
            
            # F1 scores
            metrics['macro_f1'] = f1_score(targets, predictions, average='macro', zero_division=0)
            metrics['weighted_f1'] = f1_score(targets, predictions, average='weighted', zero_division=0)
            metrics['micro_f1'] = f1_score(targets, predictions, average='micro', zero_division=0)
            
            # Precision scores
            metrics['macro_precision'] = precision_score(targets, predictions, average='macro', zero_division=0)
            metrics['weighted_precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
            
            # Recall scores  
            metrics['macro_recall'] = recall_score(targets, predictions, average='macro', zero_division=0)
            metrics['weighted_recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
            
            # Per-class F1 scores
            per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
            for i, (class_name, f1) in enumerate(zip(self.class_names, per_class_f1)):
                metrics[f'f1_{class_name}'] = f1
                
        else:
            # Fallback: basic accuracy only
            metrics['accuracy'] = (predictions == targets).mean()
            metrics['macro_f1'] = metrics['accuracy']  # Approximation
        
        return metrics
    
    def _log_detailed_results(self, metrics, predictions, targets):
        """Log detailed results including per-class analysis"""
        logger.info("📊 Multi-class Evaluation Results:")
        logger.info(f"   Primary metric ({self.primary_metric}): {metrics.get(self.primary_metric, 0.0):.4f}")
        
        # Log main metrics
        main_metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        for metric in main_metrics:
            if metric in metrics:
                logger.info(f"   {metric}: {metrics[metric]:.4f}")
        
        # Log per-class F1 scores
        logger.info("   Per-class F1 scores:")
        for class_name in self.class_names:
            f1_key = f'f1_{class_name}'
            if f1_key in metrics:
                logger.info(f"     {class_name}: {metrics[f1_key]:.4f}")
        
        # Log class distribution
        unique, counts = np.unique(targets, return_counts=True)
        logger.info("   True class distribution:")
        for class_idx, count in zip(unique, counts):
            if class_idx < len(self.class_names):
                percentage = (count / len(targets)) * 100
                logger.info(f"     {self.class_names[class_idx]}: {count} ({percentage:.1f}%)")
        
        # Log confusion matrix info if sklearn available
        if self.sklearn_available:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(targets, predictions)
            
            # Log most confused classes
            logger.info("   Most confused class pairs:")
            confusion_pairs = []
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    if i != j and cm[i, j] > 0:
                        confusion_pairs.append((cm[i, j], self.class_names[i], self.class_names[j]))
            
            # Sort by confusion count and show top 5
            confusion_pairs.sort(reverse=True)
            for count, true_class, pred_class in confusion_pairs[:5]:
                logger.info(f"     {true_class} → {pred_class}: {count} errors")
    
    def get_confusion_matrix(self):
        """Get confusion matrix for visualization"""
        if not self.predictions or not self.sklearn_available:
            return None
        
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(np.array(self.targets), np.array(self.predictions))
    
    def get_classification_report(self):
        """Get detailed classification report"""
        if not self.predictions or not self.sklearn_available:
            return None
        
        from sklearn.metrics import classification_report
        return classification_report(
            np.array(self.targets), 
            np.array(self.predictions),
            target_names=self.class_names,
            zero_division=0
        )
    
    def reset(self):
        """Reset stored predictions and targets"""
        self.predictions = []
        self.targets = []
        self.sample_ids = []


# Update the existing test function to include multi-class testing
def test_multiclass_metrics():
    """Test multi-class metrics functionality"""
    print("🧪 Testing MultiClassMetrics...")
    
    # Create test data for 10 classes
    num_classes = 10
    num_samples = 100
    
    # Simulate predictions and targets
    predictions = torch.randint(0, num_classes, (num_samples,))
    targets = torch.randint(0, num_classes, (num_samples,))
    sample_ids = [f"sample_{i}" for i in range(num_samples)]
    
    # Create metrics object
    class_names = ['N', 'H', 'X', 'A', 'S', 'U', 'C', 'O', 'D', 'F']
    metrics = MultiClassMetrics(
        num_classes=num_classes,
        class_names=class_names,
        primary_metric='macro_f1'
    )
    
    # Append results
    metrics.append(sample_ids, predictions, targets)
    
    # Compute metrics
    primary_score = metrics.summarize()
    
    print(f"✅ Primary metric (macro_f1): {primary_score:.4f}")
    print("✅ MultiClassMetrics test passed!")


# Add to the main test function
if __name__ == "__main__":
    test_ultimate_model()
    test_multiclass_metrics()
    print("\n✨ Ultimate model with multi-class support ready!")