#Transfomers : 

#steps to build a transformer model :
# 1- Embedding + Positional Encoding
# 2- Multi-Head Attention
# 3- Feed-Forward Network
# 4- Residual & Layer Norm
# 5- Encoder Block
# 6- Decoder Block
# 7- Final Linear & Softmax



import torch 
import torch.nn as nn
import math


#1- Embedding + Positional Encoding
#transformer doesnt keep track of the order alone so we need this class that gives it the position of the word in the sentence :

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        #a tensor where we store all the positional encodings for the maximum length of the input sequence
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0 , max_len, dtype=torch.float).unsqueeze(1) #this creates column vector of positions from 0 to max_len-1

        #create the denominator term for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) #this creates a vector of size d_model/2 where each element is calculated as exp(-i/d_model) where i is the index of the element

        pe[:, 0::2] = torch.sin(position * div_term) #assigns to even positions the sine of the product of position and div_term
        pe[:, 1::2] = torch.cos(position * div_term) #assigns to odd positions the cosine of the product of position and div_term

        #add batch dimension and register pe as a buffer (not a parameter but should be saved and moved with the model)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x): #gives to each info the position in the sentence
        #add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x
    

#---------------------------
#Scaled Dot-Product Attention :
    # Query x Key -> softmax -> value
    #the attention mechanism allows the model to focus on different parts of the input sequence when making
    #each words in the sentence will have a positional encoding added to its embedding, which allows the model to take into account the order of the words in the sentence. The positional encoding is calculated using sine and cosine functions of different frequencies, which allows the model to learn different patterns of word positions.
    #eahc words asks for the word before it and the word after it to understand the context of the sentence, so the positional encoding helps the model to understand the relationship between the words in the sentence and their positions.
def attention(q, k,v ,mask=None):
        scores = torch.matmul(q , k.transpose(-2, -1)) / math.sqrt(q.size(-1)) #calculate the attention scores by multiplying the query and key matrices and scaling by the square root of the dimension of the query/key vectors
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #apply the mask to the attention scores, setting masked positions to a very large negative value
        attn = torch.softmax(scores, dim=-1) #apply softmax to get the attention weights
        output = torch.matmul(attn, v) #calculate the output by multiplying the attention weights
        return output, attn

    


#2- Multi-Head Attention :
#enhancement of the attention mechanism that allows the model to jointly attend to information from different representation subspaces
#instead of performing a single attention function with d_model-dimensional keys, values and queries, we linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d_v-dimensional output values. These are concatenated and once again projected, resulting in the final values.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model , num_head):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model // num_head #dimension of each head
        self.q_linear = nn.Linear(d_model, d_model) #linear layer to project the input to query space
        self.k_linear = nn.Linear(d_model, d_model) #linear layer to project the input to key space
        self.v_linear = nn.Linear(d_model, d_model) #linear layer to project the input to value space
        self.out = nn.Linear(d_model, d_model) #linear layer to project the concatenated output of the attention heads back to d_model dimensions
        

    def forward(self, q, k , v , mask=None):
        batch_size = q.size(0)

        #perform linear projections and split into num_head heads
        q = self.q_linear(q).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) #project the input to query space and reshape for multi-head attention
        k = self.k_linear(k).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) #project the input to key space and reshape for multi-head attention
        v = self.v_linear(v).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2) #project the input to value space and reshape for multi-head attention

        #apply attention on all the projected vectors in batch
        scores , attn = attention(q , k , v , mask=mask)

        #concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) #concatenate the output of the attention heads
        output = self.out(concat) #project the concatenated output back to d_model dimensions

        return output
    



#3- Feed-Forward Network :
#position-wise feed-forward networks are applied to each position separately and identically. This consists of
#two linear transformations with a ReLU activation in between. The dimensionality of input and output is d_model, and the inner layer has dimensionality d_ff.
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) #first linear layer to project from d_model to d_ff dimensions
        self.linear2 = nn.Linear(d_ff, d_model) #second linear layer to project back from d_ff to d_model dimensions

    def forward(self, x):
        x = self.linear1(x) #apply first linear transformation
        x = torch.relu(x) #apply ReLU activation
        x = self.linear2(x) #apply second linear transformation
        return x
    



#Encoder Block :
#each encoder block consists of a multi-head attention layer followed by a feed-forward network, with residual connections and layer normalization applied to both sub-layers.
#the input to the encoder block is passed through the multi-head attention layer, and the output of this layer is added to the original input (residual connection) and then normalized using layer normalization. The output of this first sub-layer is then passed through the feed-forward network, and again a residual connection is applied followed by layer normalization.
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_head) #multi-head attention layer
        self.ffn = FeedForward(d_model, d_ff) #feed-forward network
        self.norm1 = nn.LayerNorm(d_model) #layer normalization for the first sub-layer
        self.norm2 = nn.LayerNorm(d_model) #layer normalization for the second sub-layer

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask=mask) #apply multi-head attention
        x = self.norm1(x + attn_output) #apply residual connection and layer normalization
        ffn_output = self.ffn(x) #apply feed-forward network
        x = self.norm2(x + ffn_output) #apply residual connection and layer normalization
        return x
    







#Transformer Encoder :
#the transformer encoder consists of a stack of N encoder blocks. The input to the encoder is passed through an embedding layer and a positional encoding layer before being fed into the first encoder block. The output of the last encoder block is the final output of the transformer encoder, which can be used for various downstream tasks such as machine translation, text classification, etc.
#the transformer encoder is designed to process sequential data, such as natural language, and can capture long-range dependencies in the input sequence through the use of self-attention mechanisms. The multi-head attention allows the model to attend to different parts of the input sequence simultaneously, while the feed-forward networks provide additional capacity for learning complex representations. The residual connections and layer normalization help to stabilize training and improve convergence.
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_head, d_ff, input_vocab_size, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model) #embedding layer to convert input tokens to embeddings
        self.positional_encoding = PositionalEncoding(d_model, max_len) #positional encoding layer to add positional information to the embeddings
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_head, d_ff) for _ in range(num_layers)]) #stack of encoder blocks

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model) #convert input tokens to embeddings and scale by sqrt(d_model)
        x = self.positional_encoding(x) #add positional encoding to the embeddings

        for layer in self.layers:
            x = layer(x, mask=mask) #pass through each encoder block

        return x #return the output of the last encoder block as the final output of the transformer encoder
    


# i need to seperate the embedding class from the positional encoding class because the embedding class is responsible for converting input tokens into dense vector representations, while the positional encoding class is responsible for adding positional information to these embeddings. By separating these two components, we can have more flexibility in how we use them. For example, we can use different types of embeddings (e.g., word embeddings, character embeddings) with the same positional encoding, or we can use the same embeddings with different positional encodings depending on the task at hand. Additionally, separating these components allows for better modularity and reusability of code, as we can easily swap out or modify one component without affecting the other.
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) #embedding layer to convert input tokens to embeddings

    def forward(self, x):
        return self.embedding(x) 
    


#Mask Functions :
def create_padding_mask(seq, pad_token =0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2) #create a mask for padding tokens (assuming 0 is the padding token)

def create_casual_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool() #create a mask to prevent attention to future tokens (causal mask)



#Decoder Block :
#each decoder block consists of three sub-layers: a masked multi-head attention layer, a multi-head attention layer that attends to the output of the encoder, and a feed-forward network. Similar to the encoder block, residual connections and layer normalization are applied to each sub-layer.
class DecoderBlock(nn.Module):
    def __init__(self, d_model ,num_head , d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_head) #masked multi-head attention layer for the decoder
        self.cross_attention = MultiHeadAttention(d_model, num_head) #multi-head attention layer that attends to the output of the encoder
        self.ffn = FeedForward(d_model, d_ff) #feed-forward network

        self.norm1 = nn.LayerNorm(d_model) #layer normalization for the first sub-layer
        self.norm2 = nn.LayerNorm(d_model) #layer normalization for the second sub-layer
        self.norm3 = nn.LayerNorm(d_model) #layer normalization for the third sub-layer

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attention1 = self.self_attention(x, x, x, mask=tgt_mask) #apply masked multi-head attention
        x = self.norm1(x + attention1) #apply residual connection and layer normalization
        attention2 = self.cross_attention(x, enc_output, enc_output, mask=src_mask) #apply multi-head attention that attends to the output of the encoder
        x = self.norm2(x + attention2) #apply residual connection and layer normalization

        #3 - feed forward 
        ffn = self.ffn(x) #apply feed-forward network
        x = self.norm3(x + ffn) #apply residual connection and layer normalization
        return x
    



#the Transformer Decoder Stack :
#the transformer decoder consists of a stack of N decoder blocks. The input to the decoder is passed through an embedding layer and a positional encoding layer before being fed into the first decoder block. The output of the last decoder block is the final output of the transformer decoder, which can be used for various downstream tasks such as machine translation, text generation, etc.

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers , d_model, num_head , d_ff , vocab_size , max_len = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) #embedding layer to convert input tokens to embeddings
        self.positional_encoding = PositionalEncoding(d_model, max_len) #positional encoding layer to add positional information to the embeddings
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_head, d_ff) for _ in range(num_layers)]) #stack of decoder blocks

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x) #add positional encoding to the embeddings
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask) #pass through each decoder block

        return x #return the output of the last decoder block as the final output of the transformer decoder
    



#Final Linear & Softmax :
#after the decoder stack, we typically have a final linear layer that projects the output of the
# you need to convert embedding to vocab probabilities 
class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size) #linear layer to project from d_model to vocab_size dimensions

    def forward(self, x):
        return self.linear(x) #project the output of the decoder to vocab_size dimensions
    



# the final Boss -> the Transformer class
class Transformer(nn.Module):
    def __init__(self , src_vocab_size , tgt_vocab_size , num_layers , d_model , num_head , d_ff , max_len = 5000):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_head, d_ff, src_vocab_size, max_len) #initialize the transformer encoder
        self.decoder = TransformerDecoder(num_layers, d_model, num_head, d_ff, tgt_vocab_size, max_len) #initialize the transformer decoder
        self.output_layer = OutputLayer(d_model, tgt_vocab_size) #initialize the final output layer

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, mask=src_mask) #pass the source input through the encoder
        dec_output = self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask) #pass the target input and encoder output through the decoder
        output = self.output_layer(dec_output) #project the output of the decoder to vocab_size dimensions
        return output #return the final output of the transformer model








if __name__ == "__main__":
    # Example usage
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    num_layers = 6
    d_model = 512
    num_head = 8
    d_ff = 2048
    max_len = 5000

    model = Transformer(src_vocab_size, tgt_vocab_size, num_layers, d_model, num_head, d_ff, max_len)

    src = torch.randint(0, src_vocab_size, (32, 20)) #batch of 32 sequences of length 20
    tgt = torch.randint(0, tgt_vocab_size, (32, 20)) #batch of 32 sequences of length 20

    src_mask = create_padding_mask(src) #create padding mask for source input
    tgt_mask = create_casual_mask(tgt.size(1)) #create causal mask for target input

    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask) #forward pass through the transformer model
    print(output.shape) #should be (32, 20, tgt_vocab_size)