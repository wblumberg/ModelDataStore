import math

# Frontogenesis (to convert from [temperature units]/m/s to [temperature units/100 km/3h multiply by 1.08e9])
# Fosberg FWI
# 

def fosberg_fwi(temperature, relative_humidity, wind_speed):
    """
    Calculate the Fosberg Fire Weather Index (FWI).
    
    Args:
        temperature: Temperature in Celsius
        relative_humidity: Relative humidity as a percentage (0-100)
        wind_speed: Wind speed in m/s
        
    AI Generated - Need to Check
    
    Returns:
        Fosberg FWI index value
    """
    
    # Equilibrium moisture content (EMC)
    emc = 0.03674 * relative_humidity * math.exp(-0.06692 * temperature)
    
    # Drying rate
    ed = 0.924 * relative_humidity**0.679 - 0.000345 * relative_humidity**2 + 0.02821 * relative_humidity - 2.3085
    
    # Fuel moisture content at which rate of spread is maximum
    m_f = 2.76 - 20.87 * math.exp(-0.023 * ed)
    
    # Rate of fire spread adjustment factor
    r = wind_speed * (m_f - emc) / (m_f + emc)
    
    # Fosberg FWI
    fwi = r * math.exp(-0.01 * (ed - emc))
    
    return fwi