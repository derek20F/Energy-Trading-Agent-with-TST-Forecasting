import datetime
def process_price(price):
    '''
    This funciton is used to process the price
    Args: price (float)
    '''
    price_min = -600.0
    price_max = 1000.0
    if price > price_max:
        price = price_max
    elif price < price_min:
        price = price_min

    price = (price-price_min) / (price_max-price_min)
        
    return price
    
def unprocess_price(price):
    '''
    This function is used to convert the normalized price back to the correct value.
    Args:
        price (float): in [0,1]
    '''
    
    price_min = -600.0
    price_max = 1000.0
    
    price = price_min + price*(price_max - price_min)
    return price
    
def process_demand(demand):
    demand_min = -44.56
    demand_max = 2562.95
    demand = (demand - demand_min) / (demand_max-demand_min)
    return demand
    
def unprocess_demand(demand):
    demand_min = -44.56
    demand_max = 2562.95
    demand = demand_min + demand*(demand_max - demand_min)
    return demand
    
def process_temp_air(temp_air):
    temp_air_max = 40.0
    temp_air_min = 0.0
    temp_air = (temp_air - temp_air_min) / (temp_air_max - temp_air_min)
    return temp_air
    
def unprocess_temp_air(temp_air):
    temp_air_max = 40.0
    temp_air_min = 0.0
    
    temp_air = temp_air_min + temp_air*(temp_air_max - temp_air_min)
    return temp_air
    
def process_pv_power(pv_power):
    pv_power_min = 0.0
    pv_power_max = 12.0
    pv_power = (pv_power - pv_power_min) / (pv_power_max - pv_power_min)
    return pv_power
    
def unporcess_pv_power(pv_power):
    pv_power_min = 0.0
    pv_power_max = 12.0
    
    pv_power = pv_power_min + pv_power * (pv_power_max - pv_power_min)
    
    return pv_power

def compute_net_profit_cost(timestamp, market_price):
    '''
    To compute the net discharge profit and charging cost by considering tariffs
    and if the time is at peak of off-peak
    Args:
        timestamp (datetime object), in UTC
        market_price (float)
    '''
    is_peak = timestamp.hour >= 7 and timestamp.hour < 11 #UTC time

    if is_peak:
        #during peak, you will earn extra 30% when you discharge
        #it will also cost you 40% more for charging.
        discharge_profit = market_price + abs(market_price * 0.30) #positive
        charging_cost = market_price + abs(market_price * 0.40) #positive
        
    else:
        #during off-peak, you will earn less when discharge
        #but it only cost you 5% for charging
        discharge_profit = market_price - abs(market_price * 0.15) #positive
        charging_cost = market_price + abs(market_price * 0.05) #positive #the large the worser

    return discharge_profit, charging_cost

def compute_nextN_net_profit_cost(current_timestamp, nextN_market_price):
    '''
    To compute the net discharge profit and charging cost by considering tariffs
    and if the time is at peak of off-peak
    Args:
        timestamp (datetime object), in UTC
        market_price (float)
    '''
    discharge_profits = []
    charging_costs = []
    for i in range(len(nextN_market_price)):
        timestamp = current_timestamp+datetime.timedelta(minutes=(5*(i+1)))
        is_peak = timestamp.hour >= 7 and timestamp.hour < 11 #UTC time

        if is_peak:
            #during peak, you will earn extra 30% when you discharge
            #it will also cost you 40% more for charging.
            discharge_profit = nextN_market_price[i] + abs(nextN_market_price[i] * 0.30) #positive
            charging_cost = nextN_market_price[i] + abs(nextN_market_price[i] * 0.40) #positive
            
        else:
            #during off-peak, you will earn less when discharge
            #but it only cost you 5% for charging
            discharge_profit = nextN_market_price[i] - abs(nextN_market_price[i] * 0.15) #positive
            charging_cost = nextN_market_price[i] + abs(nextN_market_price[i] * 0.05) #positive #the large the worser
    discharge_profits.append(discharge_profit)
    charging_costs.append(charging_cost)


    return discharge_profits, charging_costs