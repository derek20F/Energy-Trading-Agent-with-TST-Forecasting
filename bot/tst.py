import torch
import numpy as np
import math
from torch import nn, Tensor
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple



device = "cpu"


def data_preprocess(in_df):
    '''
    This function is used to preprocess the data read from csv file.
    This function wil map all the features to [0,1]
    The input df is a pandas dataframe with following columns:
    ['price', 'demand', 'temp_air', 'pv_power', 'pv_power_forecast_1h',
    'pv_power_forecast_2h', 'pv_power_forecast_24h', 'pv_power_basic']
    '''
    import copy
    df = copy.deepcopy(in_df)
    price_data = df['price']
    price_min = -600
    price_max = 1000
    price_data[price_data<price_min] = price_min
    price_data[price_data>price_max] = price_max
    price_data = (price_data - price_min) / (price_max-price_min)
    df['price'] = price_data


    demand_data = df['demand']
    demand_min = -44.56
    demand_max = 2562.95
    demand_data = (demand_data - demand_min) / (demand_max-demand_min)
    df['demand'] = demand_data

    temp_air_data = df['temp_air']
    temp_air_max = 40
    temp_air_min = 0
    temp_air_data = (temp_air_data - temp_air_min) / (temp_air_max - temp_air_min)
    df['temp_air'] = temp_air_data

    pv_power_data = df['pv_power']
    pv_power_min = 0.0
    pv_power_max = 12
    pv_power_data = (pv_power_data - pv_power_min) / (pv_power_max - pv_power_min)
    df['pv_power'] = pv_power_data

    
    return df


#==============================================================================
#==============================================================================
#==============================================================================

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what
    the positional encoding layer does and why it is needed:

    "Since our model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we must inject some
    information about the relative or absolute position of the tokens in the
    sequence." (Vaswani et al, 2017)
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self,
        dropout: float=0.1,
        max_seq_len: int=360,
        d_model: int=512, #embedding dimension
        batch_first: bool=True
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len,dtype=torch.float).unsqueeze(1) #Column Vector

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model) #[batch,seq_len,embedding_dim]

            pe[0, :, 0::2] = torch.sin(position * div_term) #even

            pe[0, :, 1::2] = torch.cos(position * div_term) #odd
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)

            pe[:, 0, 0::2] = torch.sin(position * div_term)

            pe[:, 0, 1::2] = torch.cos(position * div_term) #[seq_len,batch,embedding_dim]

        self.register_buffer('pe', pe) #buffer of pytorch will not be trained (no gradient)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        if x.dim() == 2: #debug
            print("unbatch input to patch embedding layer") #debug
            if self.batch_first:
                x = x + self.pe[0,:x.size(1),:] #[:,sequence_len]
            else:
                x = x + self.pe[:x.size(0),0,:] #[sequence_len]
            return self.dropout(x)


        if self.batch_first:
            x = x + self.pe[:,:x.size(1)] #[:,sequence_len]

        else:
            x = x + self.pe[:x.size(0)] #[:sequence_len]

        return self.dropout(x)

#==============================================================================
#==============================================================================
#==============================================================================

class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".
    A detailed description of the code can be found in my article here:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.
    Unlike the paper, this class assumes that input layers, positional encoding
    layers and linear mapping layers are separate from the encoder and decoder,
    i.e. the encoder and decoder only do what is depicted as their sub-layers
    in the paper. For practical purposes, this assumption does not make a
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the
    Encoder() and Decoder() classes.
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020).
    'Deep Transformer Models for Time Series Forecasting:
    The Influenza Prevalence Case'.
    arXiv:2001.08317 [cs, stat] [Preprint].
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).
    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """

    def __init__(self,
        input_size: int, #equal to the length of the flatten image (which is 230)
        decoder_input_size: int, #the input features for the decoder (price, demand, pv_power)
        #dec_seq_len: int, #debug: this param is not used
        batch_first: bool=True, #if true, it mean batch is at index 0, if false, batch is at index 1
        out_seq_len: int=58, #predict how many frames in the future? #debug: aka dec_seq_length
        max_seq_len: int=600, #which is 5 mins #in our task, this number is equal to enc_seq_len. [Max length of the input sequence]
        dim_val: int=512, #sub-layer's output dimension. (embedding dimension) #debug: decrease this if model is slow
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2,
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048, #debug: decrease this if model is slow
        dim_feedforward_decoder: int=2048, #debug: decrease this if model is slow
        num_predicted_features: int=3 #Output features: [price, pv_power, demand]
        ):

        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len (out_seq_len): int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.

        """

        super().__init__()

        self.dec_seq_len = out_seq_len
        self.out_seq_len = out_seq_len
        self.max_seq_len = max_seq_len
        self.enc_seq_len = max_seq_len #encoder sequence length


        #print("input_size is: {}".format(input_size))
        #print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        # Linear transformation. Linear mapping.
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val #embedding dimension
            )

        self.decoder_input_layer = nn.Linear(
            #debug: I think this should equal to the encoder input size# in_features=num_predicted_features,
            in_features=decoder_input_size,
            out_features=dim_val
            )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=max_seq_len,
            batch_first=batch_first
            )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, #embedding dimension
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder, #debug: can I change this?
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None #normalisation is set to None because in the nn.TransformerEncoderLayer, the normalisation is already done.
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
            )
        #self.outputmapping = nn.Tanh() #map the output data range to [-1,1]
        self.outputmapping = nn.Sigmoid() #map the output data range to [0,1]
    '''
    forward function need src and tgt, which is not in the model. they can have any size if not well defined.
    '''
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
                tgt_mask: Tensor=None) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        #debug: what is batch first?

        Args:
            src: the encoder's input sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate, 230 for our task)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """
        #Don't need to check if the input size to encoder and decoder are correct.
        #Because transformer can actually take arbitrary length of input.

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        ##print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src) # src shape: [batch_size, src length, dim_val] regardless of number of input features
        ##print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
            )

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt) # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        #print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        #if src_mask is not None:
            #print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        #if tgt_mask is not None:
            #print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )

        #print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output) # shape [batch_size, target seq len]
        #print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        #output shape = [batch_size, out_seq_len, number_of_feature]
        #Mapping the output of each image to range [-1,1]

        decoder_output = self.outputmapping(decoder_output)

        return decoder_output

#==============================================================================
#==============================================================================
#==============================================================================

class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.

    """
    def __init__(self,
        #data: torch.tensor,
        step: int, #set it as the enc_seq_len
        enc_seq_len: int,
        #debug: not used at all #dec_seq_len: int,
        target_seq_len: int,
        data_file_path: str
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence
                        before any slicing. If univariate, data.size() will be
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.
                Derek: we are using image, so we should not have this value

            step (int): control the gap of each sample. (how many sample to skip)
                if step is 1, no frame is skipped. it will sample (0,n), (1,n+1), (2,n+2)
                if step is 2, it will sample (0,n), (2,n+2), (4,n+4), (6,n+6)

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence.
                     The sub-sequence is split into src, trg and trg_y later.

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            [deprecated] arget_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """

        super().__init__()
        self.data = pd.read_csv(data_file_path,index_col=False)
        #self.data = self.data[['price','demand','temp_air', 'pv_power', 'pv_power_forecast_1h', 'pv_power_forecast_2h', 'pv_power_forecast_24h', 'pv_power_basic']]
        self.data = self.data[['price','demand','temp_air', 'pv_power']]
        self.data = self.data.dropna()
        self.data = data_preprocess(self.data)
        #print(self.data)
        ##[GBH] how to normalize the data?

        
        #self.re_wid = re_wid
        #self.re_hei = re_hei
        #self.npy_path = npy_path



        self.indices = self.get_indices_entire_sequence(
            num_obs=len(self.data),
            window_size=(enc_seq_len+target_seq_len),
            step_size=step
            )

        #debug #self.data = data

        ##print("From get_src_trg: data size = {}".format(len(img_list)))

        self.enc_seq_len = enc_seq_len

        ##self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len




    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input), i.e., x1-x10
        2) trg (the decoder input), i.e., x10-x13
        3) trg_y (the target), i.e., x11-x14
        """
        # Get the first element of the i'th tuple in the list self.indices
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx] #the image names of current step.
        
        #for img in seq_imgs:
        #
        #    '''
        #    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.
        #    FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        #    '''
        #    sequence.append(temp)
        sequence = np.array(sequence) #convert to np array
        sequence = torch.from_numpy(sequence).float() #need to convert uint8 to float for gradient computation
        #print(f"sequece after converted to float: {sequence}")

        #sequence = sequence.flatten(start_dim=1)
        #print(f"sequece after flatten: {sequence}")
        #The pixel range of frames in sequence is 0-255
        #sequence = sequence / 255.0 #normalise the data to [0,1]
        #We need to transform the range to -1~1. #We are not normalising images independently, we normalise the whole sequence together.
        #sequence = sequence / sequence.max() #range from 0~1
        #sequence  = sequence - 0.5 #range -0.5~0.5
        #sequence = sequence * 2 #range -1~1

        ##print(sequence.shape)

        #print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            ##dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y

    


    def get_src_trg(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        ##dec_seq_len: int,
        target_seq_len: int
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.

        Args:

            sequence: tensor, a 1D tensor of length n where
                    n = encoder input length + target sequence length

            enc_seq_len: int, the desired length of the input to the transformer encoder

            target_seq_len: int, the desired length of the target sequence (the
                            one against which the model output is compared)

        Return:

            src: tensor, 1D, used as input to the transformer model

            trg: tensor, 1D, used as input to the transformer model

            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss.

        """
        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]

        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]
        
        #[todo] Do we need to only extract the 3 input?
        #trg = trg['price','demand','pv_power']
        #trg_y = trg_y['price','demand','pv_power']
        #print(trg)
        #print("=====")
        #print(trg_y)
        #print("=====")

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"
        # print(src,trg,trg_y)
        return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]

    # Generate indices for the dataset class.
    def get_indices_entire_sequence(self, num_obs, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences.
        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences.
        When access the data by the indices, the end_idx is exclusive while the start_idx is inclusive.

        Args:
            num_obs (int): Number of observations (time steps) in the entire
                           dataset for which indices must be generated, e.g.
                           len(data)
            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50
                               time steps, window_size = 100+50 = 150
            step_size (int): Size of each step as the data sequence is traversed
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size],
                             and the next will be [1:window_size].
        Return:
            indices: a list of tuples
        """

        #stop_position = len(data)-1 # 1- because of 0 indexing
        stop_position = num_obs

        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0

        subseq_last_idx = window_size

        indices = []

        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))

            subseq_first_idx += step_size

            subseq_last_idx += step_size

        return indices
        
#==============================================================================
#==============================================================================
#==============================================================================

def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    In the original paper "Attention is all you need", it masked out (setting to −∞)
        all values in the input of the softmax which correspond to
        illegal connections.

    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

#==============================================================================
#==============================================================================
#==============================================================================

def run_encoder_decoder_inference(
    model: nn.Module,
    src: torch.Tensor,
    forecast_window: int,
    batch_size: int,
    device,
    batch_first: bool=True
    ) -> torch.Tensor:

    """
    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.  #Derek: how, the tgt size is not fixed.

    This function returns a prediction of length forecast_window for each batch in src

    NB! If you want the inference to be done without gradient calculation,
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()

    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be
    backpropagated during training and gradient calculation hence is required.

    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.

    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before
               passed to this function.

        src: The input to the model (encoder input)

        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 frames

        batch_size: batch size

        batch_first: If true, the shape of the model input should be
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]

    """

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    # Take the last value of the target variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, :] if batch_first == False else src[:, -1, :] # shape [batch_size, number of features] #the last img in the sequence


    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False:
        tgt = tgt.unsqueeze(0)

    if batch_first == True:
        tgt = tgt.unsqueeze(1) #change shape from [batch_size, number of features] to [batch_size, 1, number of features]
    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window-1):

        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        dim_b = src.shape[1] if batch_first == True else src.shape[0]

        tgt_mask = generate_square_subsequent_mask(
            dim1=dim_a, #dim_a should be the output sequence length
            dim2=dim_a
            )

        src_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b #dim_b should be the enc_seq_len
            )

        # Make prediction
        prediction = model(src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device))

        # If statement simply makes sure that the predicted value is
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step
            # represented in tgt
            last_predicted_value = prediction[-1, :, :]

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [out_sequence_length, number_of_feature] --> [out_sequence_length, 1 ,number_of_feature]
            last_predicted_value = last_predicted_value.unsqueeze(1)

        # Detach the predicted element from the graph and concatenate with
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)

    # Create masks
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a
        )

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b
        )

    # Make final prediction
    final_prediction = model(src.to(device), tgt.to(device), src_mask.to(device), tgt_mask.to(device))

    return final_prediction


#==============================================================================
#==============================================================================
#==============================================================================


#==============================================================================
#==============================================================================
#==============================================================================


#==============================================================================
#==============================================================================
#==============================================================================


#==============================================================================
#==============================================================================
#==============================================================================


#==============================================================================
#==============================================================================
#==============================================================================