//inventory
import numpy as np

def simulate_newsstand(num_days):
    buy_price = 0.30
    sell_price = 0.45
    scrap_price = 0.05

    # Demand cumulative distributions
    demand_dist = {
        'Good': {40: 0.03, 50: 0.08, 60: 0.23, 70: 0.43, 80: 0.78, 90: 0.93, 100: 1.00},
        'Fair': {40: 0.10, 50: 0.28, 60: 0.68, 70: 0.88, 80: 0.96, 90: 1.00, 100: 1.00},
        'Poor': {40: 0.44, 50: 0.66, 60: 0.82, 70: 0.94, 80: 1.00, 90: 1.00, 100: 1.00}
    }

    total_revenue, total_loss_from_excess_demand, total_salvage, total_profit = 0, 0, 0, 0

    for _ in range(num_days):
        # Choose news type
        news_type_rand = np.random.uniform(0, 100)
        if news_type_rand <= 35:
            news_type = 'Good'
        elif news_type_rand <= 80:
            news_type = 'Fair'
        else:
            news_type = 'Poor'

        # Choose demand
        demand_rand = np.random.uniform(0, 100)
        demand = None
        for d, cum_prob in demand_dist[news_type].items():
            if demand_rand <= cum_prob * 100:
                demand = d
                break

        # Fixed order quantity
        order_quantity = 100
        sold_papers = min(order_quantity, demand)
        scrapped_papers = order_quantity - sold_papers
        loss_from_excess_demand = max(0, demand - order_quantity) * (sell_price - buy_price)

        daily_revenue = sold_papers * sell_price
        daily_cost = order_quantity * buy_price
        daily_salvage = scrapped_papers * scrap_price
        daily_profit = daily_revenue + daily_salvage - daily_cost - loss_from_excess_demand

        total_revenue += daily_revenue
        total_loss_from_excess_demand += loss_from_excess_demand
        total_salvage += daily_salvage
        total_profit += daily_profit

    avg_revenue = total_revenue / num_days
    avg_loss_from_excess_demand = total_loss_from_excess_demand / num_days
    avg_salvage = total_salvage / num_days
    avg_profit = total_profit / num_days

    print(f"\nSimulation over {num_days} days:")
    print(f"Average Daily Revenue from Sales: ${avg_revenue:.2f}")
    print(f"Average Daily Loss of Profit from Excess Demand: ${avg_loss_from_excess_demand:.2f}")
    print(f"Average Daily Salvage from Scrap: ${avg_salvage:.2f}")
    print(f"Average Daily Profit: ${avg_profit:.2f}")


# Run for different day counts
simulation_days = [200, 500, 1000, 10000]
for days in simulation_days:
    simulate_newsstand(days)

# Part a: Generate 100 news types
print("\nPart a: News Type Simulation (100 days)")
news_types_sim = []
for _ in range(100):
    rand_num = np.random.uniform(0, 100)
    if rand_num <= 35:
        news_types_sim.append('Good')
    elif rand_num <= 80:
        news_types_sim.append('Fair')
    else:
        news_types_sim.append('Poor')
print(news_types_sim)

# Part b: Generate 100 random demand numbers using distributions
num_samples_b = 100
print("\nPart b: Demand generation using distributions (100 samples each):")

good_demand_rand_exp = np.random.exponential(scale=50, size=num_samples_b)
fair_demand_rand_norm = np.random.normal(loc=50, scale=10, size=num_samples_b)
poor_demand_rand_poisson = np.random.poisson(lam=50, size=num_samples_b)

print("Good News Demand (Exponential, mean 50):", good_demand_rand_exp)
print("Fair News Demand (Normal, mean 50, SD 10):", fair_demand_rand_norm)
print("Poor News Demand (Poisson, mean 50):", poor_demand_rand_poisson)


//queue
import numpy as np
import matplotlib.pyplot as plt

SIM_TIME = 1000
MEAN_INTER_ARRIVAL = 10
SERVICE_TIME_MIN = 8
SERVICE_TIME_MAX = 12

current_time = 0
customers_waiting = 0
queue = []
server_busy = False

total_wait_time = 0
num_customers_served = 0
customers_in_queue_history = []
server_utilization_history = []
time_points = []

def generate_inter_arrival_time():
    return np.random.exponential(MEAN_INTER_ARRIVAL)

def generate_service_time():
    return np.random.randint(SERVICE_TIME_MIN, SERVICE_TIME_MAX + 1)

# event list: (time, type, id)
event_list = []
next_arrival_time = generate_inter_arrival_time()
event_list.append((next_arrival_time, 'arrival', 1))
event_list.sort()

customer_counter = 1

while current_time < SIM_TIME:
    if not event_list:
        break

    event_time, event_type, customer_id = event_list.pop(0)
    current_time = event_time

    time_points.append(current_time)
    customers_in_queue_history.append(customers_waiting)
    server_utilization_history.append(1 if server_busy else 0)

    if event_type == 'arrival':
        if current_time < SIM_TIME:
            customer_counter += 1
            next_arrival_time = current_time + generate_inter_arrival_time()
            if next_arrival_time < SIM_TIME:
                event_list.append((next_arrival_time, 'arrival', customer_counter))
                event_list.sort()

        if server_busy:
            customers_waiting += 1
            queue.append(current_time)
        else:
            server_busy = True
            service_duration = generate_service_time()
            server_finish_time = current_time + service_duration
            event_list.append((server_finish_time, 'departure', customer_id))
            event_list.sort()

    elif event_type == 'departure':
        num_customers_served += 1
        server_busy = False

        if queue:
            arrival_time_of_next = queue.pop(0)
            wait_time = current_time - arrival_time_of_next
            total_wait_time += wait_time
            customers_waiting -= 1

            server_busy = True
            service_duration = generate_service_time()
            server_finish_time = current_time + service_duration
            event_list.append((server_finish_time, 'departure', None))
            event_list.sort()

# capture last state
time_points.append(SIM_TIME)
customers_in_queue_history.append(customers_waiting)
server_utilization_history.append(1 if server_busy else 0)

# averages
average_wait_time = total_wait_time / num_customers_served if num_customers_served > 0 else 0
queue_area = sum((time_points[i+1] - time_points[i]) * customers_in_queue_history[i]
                 for i in range(len(time_points)-1))
average_customers_waiting = queue_area / SIM_TIME

utilization_area = sum((time_points[i+1] - time_points[i]) * server_utilization_history[i]
                       for i in range(len(time_points)-1))
average_server_utilization = utilization_area / SIM_TIME

print(f"Simulation Results (Time Units: {SIM_TIME})")
print(f"Average time customer waits in a queue: {average_wait_time:.2f} min")
print(f"Average number of customers waiting: {average_customers_waiting:.2f}")
print(f"Average utilization of the booking station: {average_server_utilization:.2%}")
print(f"Customers served: {num_customers_served}")

# plots
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.step(time_points, customers_in_queue_history, where='post')
plt.xlabel("Time")
plt.ylabel("Number of Customers in Queue")
plt.title("Sample Path: Customers in Queue Over Time")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.step(time_points, server_utilization_history, where='post')
plt.xlabel("Time")
plt.ylabel("Server Utilization (1=Busy, 0=Idle)")
plt.title("Sample Path: Server Utilization Over Time")
plt.yticks([0, 1])
plt.grid(True)

plt.tight_layout()
plt.show()


//dynamical system
import numpy as np
import matplotlib.pyplot as plt

num_terms = 100
a0_values = np.random.uniform(0.5, 1, 100)

def dynamical_system(a0, r, num_terms):
    sequence = []
    for i in range(len(a0)):
        seq = [a0[i]]
        for n in range(1, num_terms):
            next_term = (r[i]**(n-1)) * a0[i]
            seq.append(next_term)
        sequence.append(seq)
    return np.array(sequence)

# Case i) r = 0
r_values = np.zeros(100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('Plot for r=0')
plt.grid(True)
sequences = dynamical_system(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()


# Case ii) 0 < r < 1
r_values = np.random.uniform(0.001, 1, 100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('Plot for 0 < r < 1')
plt.grid(True)
sequences = dynamical_system(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()


# Case iii) -1 < r < 0
r_values = np.random.uniform(-1, -0.001, 100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('Plot for -1 < r < 0')
plt.grid(True)
sequences = dynamical_system(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()


# Case iv) |r| > 1 (can be r > 1 or r < -1)
r_values = np.random.choice(np.concatenate((np.random.uniform(1.001, 2, 100), np.random.uniform(-2, -1.001, 100))), size=100)
plt.figure(figsize=(10, 6))
plt.xlabel('n')
plt.ylabel('a(n)')
plt.title('Plot for |r| > 1')
plt.grid(True)
sequences = dynamical_system(a0_values, r_values, num_terms)
for seq in sequences:
    plt.plot(range(num_terms), seq, alpha=0.5)
plt.show()

import matplotlib.pyplot as plt

def digoxin_decay(initial_concentration, daily_dosage, n_days):
    concentration = [initial_concentration]
    for i in range(n_days):
        decayed_concentration = concentration[-1] * 0.5
        new_concentration = decayed_concentration + daily_dosage
        concentration.append(new_concentration)
    return concentration

initial_concentration = 0
n_days = 20
dosages = [0.1, 0.2, 0.3]
plt.figure(figsize=(10, 6))

for dosage in dosages:
    concentrations = digoxin_decay(initial_concentration, dosage, n_days)
    days = list(range(n_days + 1))
    plt.plot(days, concentrations, marker='o', linestyle='-', label=f'Dosage: {dosage} mg')

plt.xlabel('Day')
plt.ylabel('Digoxin Concentration (mg)')
plt.title('Digoxin Concentration in Bloodstream Over Time')
plt.legend()
plt.grid(True)
plt.show()
