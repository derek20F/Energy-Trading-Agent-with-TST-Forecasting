import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy
import sys ## Derek
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tst import TimeSeriesTransformer, run_encoder_decoder_inference
from utility import *
import copy


class TstPolicy(Policy):
    def __init__(self, window_size=5):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        
        
        self.reward = 0
        self.max_charge_rate = 5 #kW
        self.max_discharge_rate = 5 #kW
        self.interval = 5 #min
        self.max_cap = 13 #Max battery capacity, unit: kWH

        self.max_charge_energy_each_step = self.max_charge_rate*(self.interval/60)
        self.max_discharge_energy_each_step = self.max_discharge_rate*(self.interval/60)
        
        #initial blank history for storing the past 24 hr data
        #self.history = pd.DataFrame(columns=['timestamp','price','demand','temp_air','pv_power'])
        self.past1hrDischargeProfit = deque(maxlen=12)
        self.history = deque(maxlen=288) #['timestamp','price','demand','temp_air','pv_power']
        self.processed_history = deque(maxlen=288) #['price','demand','temp_air','pv_power']
        #self.processed_history = pd.DataFrame(columns=['price','demand','temp_air','pv_power'])
        #self.history = pd.DataFrame(columns=['price','demand','temp_air','pv_power'])
        #initial the TST model
        self.checkpoint_path = r"./bot/checkpoints/epoch-67-modelstatedictHalf.tar"
        
        self.device = 'cpu'
        #self.device = 'cuda'
        dim_val = 512
        n_heads = 8
        n_decoder_layers = 4 
        n_encoder_layers = 4 
        batch_first = True
        dec_seq_len = 12 # length of input given to decoder. We want to predict next 1 hr = 1*60/5=12
        enc_seq_len = 288 # past 24 hr == 24*60/5 == 288
        output_sequence_length = dec_seq_len 
        max_seq_len = enc_seq_len 
        num_predicted_features = 4 #['price','demand','temp_air','pv_power']
        input_size = 4 #The number of input variables to the endocder. 1 if univariate forecasting.
        decoder_input_size = input_size
        dim_feedforward_encoder = 2048
        dim_feedforward_decoder = 2048
        dropout_encoder = 0.2
        dropout_decoder = 0.2
        dropout_pos_enc = 0.1
        self.model = TimeSeriesTransformer(
            dim_val=dim_val,
            batch_first=batch_first,
            input_size=input_size, #the dimension of each input
            decoder_input_size = decoder_input_size,
            max_seq_len=max_seq_len,#derek: equal to encoder sequence length
            out_seq_len=output_sequence_length, #derek: should equal to dec_seq_length
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            dim_feedforward_decoder=dim_feedforward_decoder,
            dim_feedforward_encoder=dim_feedforward_encoder,
            dropout_encoder=dropout_encoder,
            dropout_decoder=dropout_decoder,
            dropout_pos_enc=dropout_pos_enc,
            num_predicted_features=num_predicted_features).to(self.device)
        
        '''
        # Make src mask for decoder with size:
        self.tgt_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=output_sequence_length
        )

        self.src_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len
            )
        '''
        
        self.loadcheckpoint(self.checkpoint_path,device=self.device)
        
    def loadcheckpoint(self, checkpoint,device=None):
        '''
        checkpoint (str): the path where the checkpoint was saved.
        device: decide where the load tensor should be stored.
            Can be 'cpu', 'cuda:1', ect.
            if None, then it will stored on the same device where the tensor was saved.

        '''
        self.model.eval()
        checkpoint = torch.load(checkpoint,map_location=device)
        self.model.load_state_dict(checkpoint)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    def tst_predict(self,input):
        self.model.eval()

        '''
        TST model take input with shape [batch_size, input_sequence_length, features].
        For our case, the batch_size = 1, input_sequecne_length should be the length of the history, features = 4.
        '''
        input_array = copy.deepcopy(input)
        #input_array = input.values
        #print(f"input array is:\n{input_array}")
        input_tensor = torch.as_tensor(input_array)
        #input_tensor = torch.from_numpy(input_array)
        input_tensor = input_tensor.float()
        input_tensor = torch.unsqueeze(input_tensor, dim=0).to(self.device)
        #print(f"input tensor is:\n{input_tensor}")
        #print(f"input tensor shape is:\n{input_tensor.shape}")

        with torch.no_grad():
            output = run_encoder_decoder_inference(
                model=self.model,
                src=input_tensor,
                forecast_window=12,
                batch_size = 1,
                device = self.device,
                batch_first = True
                )
        return output
            


    def act(self, external_state, internal_state):
        #print("===========act funciton is running=================")
        #print(f"external state: {external_state}")
        #print(f"internal state before act: {internal_state}")
        panel2battery_kW = 0 #solar panel to battery
        grid2battery_kW = 0 #Amount chaged from Grid

        #current state
        #{'total_profit': 0, 'profit_delta': 0, 'battery_soc': 7.5, 'max_charge_rate': 5, 'remaining_steps': 15767} internal state
        current_battery_cap = internal_state['battery_soc']
        max_charge_rate = internal_state['max_charge_rate']
        remaining_steps = internal_state['remaining_steps']
        if remaining_steps%100 == 0:
            print(f"{remaining_steps} left...")

        import datetime
        timestamp = external_state['timestamp']
        temp = external_state['temp_air']
        market_price = external_state['price']
        pv_power_kw = external_state['pv_power'] #power to solar panel
        demand = external_state['demand']
        #print(external_state)
        self.history.append([timestamp,market_price,demand,temp,pv_power_kw])
        
        '''
        new_row = pd.DataFrame(
            {
                'timestamp': [timestamp],
                'price': [market_price],
                'demand': [demand],
                'temp_air': [temp],
                'pv_power': [pv_power_kw]
            }
        )'''
        #self.history = pd.concat([self.history,new_row],ignore_index=True) #not efficient.
        #self.history = self.history.tail(288) #only keep upto 24hr (24*12) == 288 rows.
        self.processed_history.append([process_price(market_price),process_demand(demand),process_temp_air(temp),process_pv_power(pv_power_kw)])
        '''
        new_row_processed = pd.DataFrame(
            {
                'price': [process_price(market_price)],
                'demand': [process_demand(demand)],
                'temp_air': [process_temp_air(temp)],
                'pv_power': [process_pv_power(pv_power_kw)]
            }
        )
        '''
        #self.processed_history = pd.concat(
        #    [self.processed_history,new_row_processed],ignore_index=True) #not efficient.
        #self.processed_history = self.processed_history.tail(288) #only keep upto 24hr (24*12) == 288 rows.


        output = self.tst_predict(self.processed_history)
        
        #print(f"output is {output.shape}")

        #Convert [0,1] output back to value.
        output = torch.squeeze(output,dim=0)
        output = output.numpy(force=True)
        
        timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S%z')
        #print(output)
        
        future_price = output[:,0]
        future_demand = output[:,1]
        future_pv_power = output[:,3]

        future_price = unprocess_price(future_price)
        future_demand = unprocess_demand(future_demand)
        future_pv_power = unporcess_pv_power(future_pv_power)
        
        max_fprice = future_price.max()
        max_fdemand = future_demand.max()
        max_fpv = future_pv_power.max()

        #print(f"future price: {future_price}")
        #print(f"future demand: {future_demand}")
        #print(f"future pv_power: {future_pv_power}")

        #print(f"Updated History: {self.history}")

        #timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S%z')
        #hr_min = datetime.time(hour=timestamp.hour,minute=timestamp.minute)
        
        current_discharge_profit, current_charging_cost = compute_net_profit_cost(timestamp, market_price)
        self.past1hrDischargeProfit.append(market_price)
        

        future_discharge_profits, future_charging_costs = compute_nextN_net_profit_cost(timestamp, future_price)
            
        
        #print(internal_state)
        #sell at peak
        #if current cost is way higher than the previous 5 avg cost and the predicted future price is decreasing --> must sell now

        if market_price > 1000:
            panel2battery_kW = 0
            grid2battery_kW = -max_charge_rate

        elif ((current_discharge_profit > np.mean(future_discharge_profits)) 
            and (np.mean(self.past1hrDischargeProfit) < current_discharge_profit)):
            panel2battery_kW = 0
            grid2battery_kW = -max_charge_rate

        elif (current_discharge_profit < np.mean(future_discharge_profits)) and \
            (np.mean(self.past1hrDischargeProfit) > current_discharge_profit):
            panel2battery_kW = min(pv_power_kw, max_charge_rate)
            grid2battery_kW = max_charge_rate - panel2battery_kW
        elif np.mean(future_discharge_profits) > current_discharge_profit:
            panel2battery_kW = min(pv_power_kw, max_charge_rate)
            if current_discharge_profit > 0:
                #charge somem power, to make sure we always has power in the battery
                grid2battery_kW = max_charge_rate * (np.mean(future_discharge_profits) - current_discharge_profit) / (current_discharge_profit)
            else: #to prevent loss money
                grid2battery_kW = max_charge_rate

        elif market_price <= 0:
            #avoiding discharge as much as possible.
            panel2battery_kW = min(pv_power_kw, max_charge_rate)
            grid2battery_kW = max_charge_rate - panel2battery_kW

        else:# market_price > 0:
            #min = 0
            #max = 1000
            prob = 1 - (current_charging_cost / 100)
            if prob < 0:
                prob = 0 

            power2battery = max_charge_rate*prob

            panel2battery_kW = min(power2battery, pv_power_kw) #make profit as long as the price is positive
            #should consider to charge some battery when price is really low.
            grid2battery_kW = power2battery - panel2battery_kW

        
        return panel2battery_kW, grid2battery_kW #[solar_panel to battery, grid to battery]

    def load_historical(self, external_states: pd.DataFrame):   
        print('load_historical funciton is running')
        print(f'external state: {external_states}')
        #for price in external_states['price'].values:
        #    self.price_history.append(price)