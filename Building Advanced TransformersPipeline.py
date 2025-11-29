import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (
    Embedding, MultiHeadAttention, Dense, 
    LayerNormalization, Dropout, TextVectorization
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
from tensorflow.keras.callbacks import EarlyStopping


class DataLoader:
    """Stage 1: Data Loading and Preprocessing"""
    
    def __init__(self, vocab_size=10000, seq_length=100):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.vectorizer = None
        self.text = None
        
    def load_data(self, url='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'):
        """Load text data from URL"""
        path_to_file = get_file('shakespeare.txt', url)
        self.text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        print(f"Loaded text with {len(self.text)} characters")
        print(f"Preview:\n{self.text[:500]}\n")
        return self
    
    def vectorize_text(self):
        """Create and adapt vectorizer"""
        self.vectorizer = TextVectorization(
            max_tokens=self.vocab_size, 
            output_mode='int'
        )
        text_ds = tf.data.Dataset.from_tensor_slices([self.text]).batch(1)
        self.vectorizer.adapt(text_ds)
        
        vectorized_text = self.vectorizer([self.text])[0]
        print(f"Vectorized text shape: {vectorized_text.shape}")
        print(f"First 10 tokens: {vectorized_text.numpy()[:10]}\n")
        return vectorized_text.numpy()


class SequenceGenerator:
    """Stage 2: Sequence Generation"""
    
    def __init__(self, seq_length=100):
        self.seq_length = seq_length
    
    def create_sequences(self, text):
        """Generate input-target sequence pairs"""
        input_seqs = []
        target_seqs = []
        
        for i in range(len(text) - self.seq_length):
            input_seq = text[i:i + self.seq_length]
            target_seq = text[i + 1:i + self.seq_length + 1]
            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
        
        X = np.array(input_seqs)
        Y = np.array(target_seqs)
        
        assert X.size > 0, "Input data X is empty"
        assert Y.size > 0, "Target data Y is empty"
        
        print(f"Generated {len(X)} sequences")
        print(f"Input shape: {X.shape}, Target shape: {Y.shape}\n")
        
        return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer building block"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerModel(Model):
    """Stage 3: Model Architecture"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_layers)
        ]
        self.dense = Dense(vocab_size)

    def positional_encoding(self, seq_length, embed_dim):
        """Generate positional encodings"""
        angle_rads = self.get_angles(
            np.arange(seq_length)[:, np.newaxis],
            np.arange(embed_dim)[np.newaxis, :],
            embed_dim
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        """Calculate angle rates for positional encoding"""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :seq_len, :]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        output = self.dense(x)
        return output


class ModelTrainer:
    """Stage 4: Model Training"""
    
    def __init__(self, model):
        self.model = model
        self.history = None
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        print("Model compiled\n")
        return self
    
    def train(self, X, Y, epochs=2, batch_size=32, patience=2):
        """Train the model with early stopping"""
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=patience,
            restore_best_weights=True
        )
        
        print(f"Training on {len(X)} samples...")
        self.history = self.model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        print("\nTraining complete\n")
        return self
    
    def plot_history(self):
        """Visualize training metrics"""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        if 'accuracy' in self.history.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['accuracy'])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy')
        
        plt.tight_layout()
        plt.show()


class TextGenerator:
    """Stage 5: Text Generation"""
    
    def __init__(self, model, vectorizer, seq_length):
        self.model = model
        self.vectorizer = vectorizer
        self.seq_length = seq_length
    
    def generate(self, start_string, num_generate=100, temperature=1.0):
        """Generate text from a seed string"""
        # Vectorize the start string
        input_eval = self.vectorizer([start_string]).numpy()
        
        # Pad or truncate to match expected sequence length
        if input_eval.shape[1] < self.seq_length:
            padding = np.zeros((1, self.seq_length - input_eval.shape[1]))
            input_eval = np.concatenate((padding, input_eval), axis=1)
        elif input_eval.shape[1] > self.seq_length:
            input_eval = input_eval[:, -self.seq_length:]
        
        input_eval = tf.convert_to_tensor(input_eval)
        text_generated = []
        
        # Generate tokens one by one
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = predictions[0, -1, :] / temperature
            
            predicted_id = tf.random.categorical(
                predictions[tf.newaxis, :], 
                num_samples=1
            )[0, 0].numpy()
            
            # Update input sequence
            input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)
            input_eval = input_eval[:, -self.seq_length:]
            input_eval = tf.convert_to_tensor(input_eval)
            
            # Get the word from vocabulary
            word = self.vectorizer.get_vocabulary()[predicted_id]
            text_generated.append(word)
        
        return start_string + ' ' + ' '.join(text_generated)


class TextGenerationPipeline:
    """Main Pipeline orchestrator"""
    
    def __init__(self, vocab_size=10000, seq_length=100, 
                 embed_dim=256, num_heads=4, ff_dim=512, num_layers=2):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        
        self.data_loader = None
        self.model = None
        self.trainer = None
        self.generator = None
    
    def run(self, max_samples=10000, epochs=2, batch_size=32):
        """Execute the complete pipeline"""
        print("="*60)
        print("STAGE 1: Data Loading and Preprocessing")
        print("="*60)
        self.data_loader = DataLoader(self.vocab_size, self.seq_length)
        self.data_loader.load_data()
        vectorized_text = self.data_loader.vectorize_text()
        
        print("="*60)
        print("STAGE 2: Sequence Generation")
        print("="*60)
        seq_gen = SequenceGenerator(self.seq_length)
        X, Y = seq_gen.create_sequences(vectorized_text)
        
        # Limit samples for faster training
        X = X[:max_samples]
        Y = Y[:max_samples]
        print(f"Using {len(X)} samples for training\n")
        
        print("="*60)
        print("STAGE 3: Model Building")
        print("="*60)
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_layers=self.num_layers,
            seq_length=self.seq_length
        )
        print("Model architecture created\n")
        
        print("="*60)
        print("STAGE 4: Model Training")
        print("="*60)
        self.trainer = ModelTrainer(self.model)
        self.trainer.compile_model().train(X, Y, epochs=epochs, batch_size=batch_size)
        self.trainer.plot_history()
        
        print("="*60)
        print("STAGE 5: Text Generation")
        print("="*60)
        self.generator = TextGenerator(
            self.model, 
            self.data_loader.vectorizer, 
            self.seq_length
        )
        
        return self
    
    def generate_text(self, start_string, num_generate=100, temperature=0.7):
        """Generate text using the trained model"""
        if self.generator is None:
            raise ValueError("Pipeline must be run before generating text")
        
        print(f"Generating text with seed: '{start_string}'")
        print(f"Temperature: {temperature}\n")
        
        generated = self.generator.generate(
            start_string, 
            num_generate=num_generate, 
            temperature=temperature
        )
        
        print("Generated Text:")
        print("-" * 60)
        print(generated)
        print("-" * 60)
        
        return generated


# Execute the pipeline
if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = TextGenerationPipeline(
        vocab_size=10000,
        seq_length=100,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        num_layers=2
    )
    
    # Run all stages
    pipeline.run(max_samples=10000, epochs=2, batch_size=32)
    
    # Generate text with different temperatures
    pipeline.generate_text("To be, or not to be", num_generate=100, temperature=0.7)
    pipeline.generate_text("Romeo and Juliet", num_generate=50, temperature=1.0)