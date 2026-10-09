def calculate_reward(delta_pdr, delta_sinr, delta_throughput, packet_loss, energy,
                     alpha, beta, delta, lambda_, mu):
    r_t = (alpha * delta_pdr + 
           beta * delta_sinr + 
           delta * delta_throughput - 
           lambda_ * packet_loss - 
           mu * energy)
    return r_t
